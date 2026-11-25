//! Cross-encoder reranking recipe (`CrossEncoderRerankRecipe`) for `(query, document)` pairs.
//!
//! Executes joint bidirectional cross-attention across concatenated `(query, document)` tokens,
//! projects the pooled representation through the classification head (supporting both single-logit
//! and two-logit relevance heads), and calibrates bounded scores in `[0.0, 1.0]` via
//! `jero_kernels_rvv::rerank_sigmoid_scores_f32`.

use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelMetadata, ModelWeights, QuantType};

use crate::kernels_compat::{pool_sequence_compat, rerank_sigmoid_scores_compat};
use crate::modernbert::ModernBertRecipe;
use crate::weights::{maybe_finalize_embedding_output, project_classifier_logits_strict};

/// Logit-to-relevance calibration mode for [`CrossEncoderRerankRecipe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RerankScoringMode {
    /// Automatically detects `1`-logit sigmoid vs `2`-logit `[neg, pos]` softmax-positive heads.
    #[default]
    Auto,
    /// Expects a single-logit head (`[score]`), applying `sigmoid(score * sigmoid_scale)`.
    SingleLogitSigmoid,
    /// Expects a two-logit head (`[neg, pos]`), applying `sigmoid((pos - neg) * sigmoid_scale)`
    /// (exact 2-class softmax positive probability at `sigmoid_scale == 1.0`).
    TwoLogitPositive,
}

/// Zero-KV cross-encoder reranking recipe scoring `(query, document)` inputs with calibrated
/// sigmoid probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossEncoderRerankRecipe {
    /// Underlying bidirectional encoder backbone (`bert`, `bge`, `jina-bert`, `modernbert`, etc.).
    pub encoder: ModernBertRecipe,
    /// Optional `[CLS]` / `<s>` prefix token ID prepended by [`Self::pack_pair_tokens`].
    pub cls_token_id: Option<u32>,
    /// Optional `[SEP]` / `</s>` separator token ID inserted between query and document.
    pub sep_token_id: Option<u32>,
    /// Positive finite logit temperature multiplier passed to `rerank_sigmoid_scores_f32`.
    pub sigmoid_scale: f32,
    /// Logit interpretation mode (`Auto`, `SingleLogitSigmoid`, or `TwoLogitPositive`).
    pub scoring_mode: RerankScoringMode,
}

impl Default for CrossEncoderRerankRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossEncoderRerankRecipe {
    /// Creates a default zero-KV cross-encoder reranker recipe (`"cross-encoder"`, `num_labels = 1`).
    #[must_use]
    pub const fn new() -> Self {
        let mut encoder = ModernBertRecipe::new_bge();
        encoder.architecture = "cross-encoder";
        encoder.num_labels = 1;
        Self {
            encoder,
            cls_token_id: None,
            sep_token_id: None,
            sigmoid_scale: 1.0,
            scoring_mode: RerankScoringMode::Auto,
        }
    }

    /// Creates a `BGE-Reranker` style cross-encoder recipe (`"bge-reranker"`, `[CLS]` pooling).
    #[must_use]
    pub const fn new_bge_reranker() -> Self {
        let mut encoder = ModernBertRecipe::new_bge();
        encoder.architecture = "bge-reranker";
        encoder.num_labels = 1;
        Self {
            encoder,
            cls_token_id: None,
            sep_token_id: None,
            sigmoid_scale: 1.0,
            scoring_mode: RerankScoringMode::Auto,
        }
    }

    /// Creates a `Jina-Reranker` style cross-encoder recipe (`"jina-reranker"`, `ALiBi` + `GeGLU`).
    #[must_use]
    pub const fn new_jina_reranker() -> Self {
        let mut encoder = ModernBertRecipe::new_jina_bert();
        encoder.architecture = "jina-reranker";
        encoder.num_labels = 1;
        Self {
            encoder,
            cls_token_id: None,
            sep_token_id: None,
            sigmoid_scale: 1.0,
            scoring_mode: RerankScoringMode::Auto,
        }
    }

    /// Creates a `ModernBERT` cross-encoder reranker recipe (`"modernbert-reranker"`).
    #[must_use]
    pub const fn new_modernbert_reranker() -> Self {
        let mut encoder = ModernBertRecipe::new_modernbert();
        encoder.architecture = "modernbert-reranker";
        encoder.num_labels = 1;
        Self {
            encoder,
            cls_token_id: None,
            sep_token_id: None,
            sigmoid_scale: 1.0,
            scoring_mode: RerankScoringMode::Auto,
        }
    }

    /// Wraps an existing [`ModernBertRecipe`] backbone as a cross-encoder reranker.
    #[must_use]
    pub const fn from_encoder(encoder: ModernBertRecipe) -> Self {
        Self {
            encoder,
            cls_token_id: None,
            sep_token_id: None,
            sigmoid_scale: 1.0,
            scoring_mode: RerankScoringMode::Auto,
        }
    }

    /// Hydrates a [`CrossEncoderRerankRecipe`] from `arch` and optional [`ModelMetadata`].
    #[must_use]
    pub fn from_metadata(arch: &str, metadata: Option<&ModelMetadata>) -> Option<Self> {
        let base_arch = match arch {
            "cross-encoder" | "reranker" | "bge-reranker" => "bge",
            "jina-reranker" => "jina-bert",
            "modernbert-reranker" => "modernbert",
            other => other,
        };
        let mut encoder = ModernBertRecipe::from_metadata(base_arch, metadata)?;
        encoder.architecture = match arch {
            "cross-encoder" => "cross-encoder",
            "reranker" => "reranker",
            "bge-reranker" => "bge-reranker",
            "jina-reranker" => "jina-reranker",
            "modernbert-reranker" => "modernbert-reranker",
            _ => encoder.architecture,
        };
        if metadata.is_none() {
            encoder.num_labels = 1;
        }
        Some(Self {
            encoder,
            cls_token_id: None,
            sep_token_id: None,
            sigmoid_scale: 1.0,
            scoring_mode: RerankScoringMode::Auto,
        })
    }

    /// Configures optional `[CLS]` and `[SEP]` boundary tokens injected around `(query, document)`.
    #[must_use]
    pub const fn with_special_tokens(
        mut self,
        cls_token_id: Option<u32>,
        sep_token_id: Option<u32>,
    ) -> Self {
        self.cls_token_id = cls_token_id;
        self.sep_token_id = sep_token_id;
        self
    }

    /// Configures the positive sigmoid calibration temperature multiplier (`scale > 0.0`).
    #[must_use]
    pub const fn with_sigmoid_scale(mut self, scale: f32) -> Self {
        self.sigmoid_scale = scale;
        self
    }

    /// Configures how classifier head logits are reduced prior to sigmoid calibration.
    #[must_use]
    pub const fn with_scoring_mode(mut self, mode: RerankScoringMode) -> Self {
        self.scoring_mode = mode;
        self
    }

    /// Packs `(query_tokens, doc_tokens)` into a joint bidirectional cross-encoder token stream
    /// and matching segment-type IDs (`0` = query segment, `1` = document segment), checking `n_ctx`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `query_tokens` or `doc_tokens` is empty, or
    /// [`K3Error::ContextLengthExceeded`] if the packed sequence length exceeds `n_ctx`.
    pub fn pack_pair_tokens(
        &self,
        query_tokens: &[u32],
        doc_tokens: &[u32],
        n_ctx: usize,
    ) -> Result<(Vec<u32>, Vec<u32>), K3Error> {
        if query_tokens.is_empty() || doc_tokens.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.encoder.architecture.to_owned(),
                reason: "cross-encoder query_tokens and doc_tokens must both be non-empty"
                    .to_owned(),
            });
        }

        let extra_tokens = usize::from(self.cls_token_id.is_some())
            + if self.sep_token_id.is_some() { 2 } else { 0 };
        let total_len = query_tokens
            .len()
            .checked_add(doc_tokens.len())
            .and_then(|n| n.checked_add(extra_tokens))
            .ok_or(K3Error::ContextLengthExceeded {
                requested_tokens: usize::MAX,
                max_context_tokens: n_ctx,
            })?;
        if n_ctx == 0 || total_len > n_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: total_len,
                max_context_tokens: n_ctx,
            });
        }

        let mut tokens = Vec::with_capacity(total_len);
        let mut segments = Vec::with_capacity(total_len);

        if let Some(cls) = self.cls_token_id {
            tokens.push(cls);
            segments.push(0);
        }
        tokens.extend_from_slice(query_tokens);
        segments.extend(core::iter::repeat_n(0_u32, query_tokens.len()));
        if let Some(sep) = self.sep_token_id {
            tokens.push(sep);
            segments.push(0);
        }
        tokens.extend_from_slice(doc_tokens);
        segments.extend(core::iter::repeat_n(1_u32, doc_tokens.len()));
        if let Some(sep) = self.sep_token_id {
            tokens.push(sep);
            segments.push(1);
        }

        Ok((tokens, segments))
    }

    fn validate_scale(&self) -> Result<(), K3Error> {
        if !self.sigmoid_scale.is_finite() || self.sigmoid_scale <= 0.0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "rerank_scale".to_owned(),
                offset: 0,
                byte_len: size_of::<f32>(),
                buffer_len: size_of::<f32>(),
                reason: format!(
                    "cross-encoder sigmoid_scale ({}) must be finite and strictly positive",
                    self.sigmoid_scale
                ),
            });
        }
        Ok(())
    }

    fn evaluate_raw_pair_logit(
        &self,
        tokens: &[u32],
        segments: Option<&[u32]>,
        n_ctx: usize,
        weights: &dyn ModelWeights,
    ) -> Result<f32, K3Error> {
        let seq_hidden = self
            .encoder
            .encode_sequence(tokens, segments, n_ctx, weights)?;
        let mut pooled = vec![0.0_f32; self.encoder.hidden_dim];
        pool_sequence_compat(
            &mut pooled,
            &seq_hidden,
            tokens.len(),
            self.encoder.hidden_dim,
            self.encoder.mean_pool,
        );

        let raw_logits = project_classifier_logits_strict(weights, &pooled, None)?;
        self.reduce_head_logits(&raw_logits)
    }

    fn reduce_head_logits(&self, logits: &[f32]) -> Result<f32, K3Error> {
        for (idx, &v) in logits.iter().enumerate() {
            if !v.is_finite() {
                return Err(K3Error::CapabilityMismatch {
                    architecture: self.encoder.architecture.to_owned(),
                    reason: format!("non-finite cross-encoder head logit {v} at index {idx}"),
                });
            }
        }
        match (self.scoring_mode, logits.len()) {
            (RerankScoringMode::Auto | RerankScoringMode::SingleLogitSigmoid, 1) => Ok(logits[0]),
            (RerankScoringMode::Auto | RerankScoringMode::TwoLogitPositive, 2) => {
                let diff = logits[1] - logits[0];
                if !diff.is_finite() {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: self.encoder.architecture.to_owned(),
                        reason: format!(
                            "non-finite two-logit margin ({} - {})",
                            logits[1], logits[0]
                        ),
                    });
                }
                Ok(diff)
            }
            (mode, len) => Err(K3Error::CapabilityMismatch {
                architecture: self.encoder.architecture.to_owned(),
                reason: format!(
                    "cross-encoder head output dimension {len} incompatible with scoring mode {mode:?}"
                ),
            }),
        }
    }

    /// Evaluates a single `(query_tokens, doc_tokens)` pair and returns its calibrated relevance
    /// score in `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if inputs are empty/exceed `n_ctx`, required encoder or classifier head
    /// weights are missing or corrupted, or `sigmoid_scale` is invalid.
    pub fn rerank_pair(
        &self,
        query_tokens: &[u32],
        doc_tokens: &[u32],
        n_ctx: usize,
        weights: &dyn ModelWeights,
    ) -> Result<f32, K3Error> {
        self.validate_scale()?;
        let (tokens, segments) = self.pack_pair_tokens(query_tokens, doc_tokens, n_ctx)?;
        let raw_logit = self.evaluate_raw_pair_logit(&tokens, Some(&segments), n_ctx, weights)?;
        let mut score = [0.0_f32; 1];
        rerank_sigmoid_scores_compat(&mut score, &[raw_logit], self.sigmoid_scale)?;
        Ok(score[0])
    }

    /// Scores each document in `documents` against `query_tokens` and applies vectorized sigmoid
    /// calibration via `rerank_sigmoid_scores_f32`, preserving input document order.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if `documents` is empty, any pair fails bounds/weight validation, or
    /// `sigmoid_scale` is invalid.
    pub fn rerank_documents(
        &self,
        query_tokens: &[u32],
        documents: &[&[u32]],
        n_ctx: usize,
        weights: &dyn ModelWeights,
    ) -> Result<Vec<f32>, K3Error> {
        self.validate_scale()?;
        if documents.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.encoder.architecture.to_owned(),
                reason: "cross-encoder documents slice cannot be empty".to_owned(),
            });
        }

        let mut raw_logits = Vec::with_capacity(documents.len());
        for &doc in documents {
            let (tokens, segments) = self.pack_pair_tokens(query_tokens, doc, n_ctx)?;
            let logit = self.evaluate_raw_pair_logit(&tokens, Some(&segments), n_ctx, weights)?;
            raw_logits.push(logit);
        }

        let mut scores = vec![0.0_f32; raw_logits.len()];
        rerank_sigmoid_scores_compat(&mut scores, &raw_logits, self.sigmoid_scale)?;
        Ok(scores)
    }

    /// Scores and sorts `documents` by descending calibrated relevance score, tie-breaking
    /// deterministically by ascending document index.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if [`Self::rerank_documents`] fails.
    pub fn rank_documents(
        &self,
        query_tokens: &[u32],
        documents: &[&[u32]],
        n_ctx: usize,
        weights: &dyn ModelWeights,
    ) -> Result<Vec<(usize, f32)>, K3Error> {
        let scores = self.rerank_documents(query_tokens, documents, n_ctx, weights)?;
        let mut ranked: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(ranked)
    }
}

impl ArchRecipe for CrossEncoderRerankRecipe {
    fn architecture_id(&self) -> &'static str {
        self.encoder.architecture
    }

    fn requires_kv_cache(&self) -> bool {
        false
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        self.encoder.supports_quant(quant)
    }

    fn forward_classify(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.validate_scale()?;
        let prompt = ctx.prompt_tokens().to_vec();
        let seq_len = prompt.len();
        let seq_hidden = self
            .encoder
            .encode_sequence(&prompt, None, ctx.n_ctx(), weights)?;

        let handled_embedding = maybe_finalize_embedding_output(
            ctx,
            &seq_hidden,
            seq_len,
            self.encoder.hidden_dim,
            self.encoder.effective_default_pooling(),
            weights,
        )?;
        if handled_embedding {
            if ctx.position() == 0 {
                ctx.advance_position(seq_len)?;
            }
            return Ok(());
        }

        if ctx.logits().is_empty() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "rerank_logits".to_owned(),
                offset: 0,
                byte_len: size_of::<f32>(),
                buffer_len: 0,
                reason: "cross-encoder output logits buffer cannot be empty".to_owned(),
            });
        }

        let mut pooled = vec![0.0_f32; self.encoder.hidden_dim];
        pool_sequence_compat(
            &mut pooled,
            &seq_hidden,
            seq_len,
            self.encoder.hidden_dim,
            self.encoder.mean_pool,
        );
        let raw_logits = project_classifier_logits_strict(weights, &pooled, None)?;
        let reduced = self.reduce_head_logits(&raw_logits)?;
        let mut calibrated = [0.0_f32; 1];
        rerank_sigmoid_scores_compat(&mut calibrated, &[reduced], self.sigmoid_scale)?;

        let out_logits = ctx.logits_mut();
        out_logits.fill(0.0);
        out_logits[0] = calibrated[0];

        if ctx.position() == 0 {
            ctx.advance_position(seq_len)?;
        }
        Ok(())
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.forward_classify(ctx, weights)
    }

    fn decode_step(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        Err(K3Error::CapabilityMismatch {
            architecture: self.encoder.architecture.to_owned(),
            reason: format!(
                "cross-encoder reranker '{}' is single-pass and does not support decode_step",
                self.encoder.architecture
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modernbert::test_fixtures::{
        build_complete_gguf_encoder, build_complete_safetensors_encoder,
    };
    use jero_core::test_helpers::create_synthetic_gguf_v3;
    use jero_core::{GgufFile, SafeTensorsFile};

    fn configure_test_reranker(
        mut recipe: CrossEncoderRerankRecipe,
        num_layers: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        n_heads: usize,
    ) -> CrossEncoderRerankRecipe {
        recipe.encoder.num_layers = num_layers;
        recipe.encoder.hidden_dim = hidden_dim;
        recipe.encoder.ffn_dim = ffn_dim;
        recipe.encoder.n_heads = n_heads;
        recipe.encoder.head_dim = hidden_dim / n_heads;
        recipe
    }

    #[test]
    fn test_cross_encoder_single_and_two_logit_rerank_and_order() {
        for num_labels in [1_usize, 2] {
            let gguf_bytes =
                build_complete_gguf_encoder("bge", 1, 24, 8, 16, Some(num_labels), false);
            let gguf = GgufFile::parse(&gguf_bytes).expect("parse reranker gguf");

            let recipe = configure_test_reranker(
                CrossEncoderRerankRecipe::new_bge_reranker().with_special_tokens(Some(0), Some(1)),
                1,
                8,
                16,
                2,
            );
            assert!(!recipe.requires_kv_cache());

            let query = [2_u32, 4, 6];
            let docs: [&[u32]; 3] = [&[3_u32, 5, 7], &[8_u32, 10, 12], &[14_u32, 18, 22]];

            let scores = recipe
                .rerank_documents(&query, &docs, 64, &gguf)
                .expect("rerank_documents succeeds");
            assert_eq!(scores.len(), 3);
            for &s in &scores {
                assert!(s.is_finite());
                assert!(
                    s > 0.0 && s < 1.0,
                    "expected calibrated (0, 1) score, got {s}"
                );
            }

            let ranked = recipe
                .rank_documents(&query, &docs, 64, &gguf)
                .expect("rank_documents succeeds");
            assert_eq!(ranked.len(), 3);
            assert!(ranked[0].1 >= ranked[1].1);
            assert!(ranked[1].1 >= ranked[2].1);

            // ArchRecipe::forward_classify writes exact same calibrated score at logits[0]
            let (packed, _) = recipe.pack_pair_tokens(&query, docs[0], 64).expect("pack");
            let mut logits = [0.0_f32; 2];
            let mut ctx =
                ExecutionContext::new(&packed, 64, &mut logits, &mut []).expect("rerank ctx");
            recipe
                .forward_classify(&mut ctx, &gguf)
                .expect("forward_classify rerank succeeds");
            assert!((ctx.logits()[0] - scores[0]).abs() < 1e-5);
            assert!(matches!(
                recipe.decode_step(&mut ctx, &gguf),
                Err(K3Error::CapabilityMismatch { .. })
            ));
        }
    }

    #[test]
    fn test_cross_encoder_gguf_vs_safetensors_parity() {
        let gguf_bytes = build_complete_gguf_encoder("bge", 1, 20, 8, 16, Some(1), false);
        let st_bytes = build_complete_safetensors_encoder("bge", 1, 20, 8, 16, 2, Some(1), false);

        let gguf = GgufFile::parse(&gguf_bytes).expect("parse gguf");
        let st = SafeTensorsFile::parse(&st_bytes).expect("parse safetensors");

        let recipe = configure_test_reranker(
            CrossEncoderRerankRecipe::new_bge_reranker().with_sigmoid_scale(1.25),
            1,
            8,
            16,
            2,
        );
        let query = [2_u32, 5];
        let docs: [&[u32]; 2] = [&[3_u32, 9], &[7_u32, 15]];

        let scores_g = recipe
            .rerank_documents(&query, &docs, 64, &gguf)
            .expect("gguf rerank");
        let scores_s = recipe
            .rerank_documents(&query, &docs, 64, &st)
            .expect("safetensors rerank");

        for i in 0..2 {
            assert!(
                (scores_g[i] - scores_s[i]).abs() < 1e-5,
                "GGUF vs SafeTensors rerank mismatch at {i}: {} vs {}",
                scores_g[i],
                scores_s[i]
            );
        }
    }

    #[test]
    fn test_cross_encoder_fail_closed_validation() {
        // Missing classifier head fails closed even when all encoder layer weights exist
        let no_head_bytes = build_complete_gguf_encoder("bge", 1, 16, 8, 16, None, false);
        let no_head_gguf = GgufFile::parse(&no_head_bytes).expect("parse no-head gguf");
        let recipe = configure_test_reranker(CrossEncoderRerankRecipe::new(), 1, 8, 16, 2);

        let err = recipe
            .rerank_pair(&[1_u32, 2], &[3_u32, 4], 64, &no_head_gguf)
            .expect_err("missing classifier head must fail closed");
        assert!(matches!(err, K3Error::TensorBoundsError { .. }));

        // Empty GGUF fails closed
        let empty_bytes = create_synthetic_gguf_v3("cross-encoder", 64, &[]);
        let empty_gguf = GgufFile::parse(&empty_bytes).expect("parse empty gguf");
        let err_empty = recipe
            .rerank_pair(&[1_u32], &[2_u32], 64, &empty_gguf)
            .expect_err("empty gguf must fail closed");
        assert!(matches!(err_empty, K3Error::TensorBoundsError { .. }));

        // 3-class head incompatible with Auto rerank mode fails closed
        let three_class_bytes = build_complete_gguf_encoder("bge", 1, 16, 8, 16, Some(3), false);
        let three_class_gguf = GgufFile::parse(&three_class_bytes).expect("parse 3-class gguf");
        let err_3 = recipe
            .rerank_pair(&[1_u32, 2], &[3_u32, 4], 64, &three_class_gguf)
            .expect_err("3-logit head must fail closed in Auto rerank mode");
        assert!(matches!(err_3, K3Error::CapabilityMismatch { .. }));

        // Invalid scale fails closed
        let valid_bytes = build_complete_gguf_encoder("bge", 1, 16, 8, 16, Some(1), false);
        let valid_gguf = GgufFile::parse(&valid_bytes).expect("parse valid gguf");
        let bad_scale = recipe.clone().with_sigmoid_scale(0.0);
        assert!(matches!(
            bad_scale.rerank_pair(&[1_u32], &[2_u32], 64, &valid_gguf),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // Empty query / document fails closed
        assert!(matches!(
            recipe.rerank_pair(&[], &[2_u32], 64, &valid_gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));
        assert!(matches!(
            recipe.rerank_pair(&[1_u32], &[], 64, &valid_gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }
}
