//! Zero-KV `DeBERTa-v3` disentangled-attention encoder recipe (`deberta-v3`).
//!
//! Computes single-pass bidirectional attention with log-bucketed relative position bias
//! (`c2c + c2p + p2c`) via `deberta_disentangled_attention_f32` and strict ADR-045 fail-closed
//! weight hydration (`token_embd.weight`, `rel_embeddings.weight`, per-layer norms/Q/K/V/O/FFN,
//! and classifier head) without KV cache allocation.

use jero_core::{
    ArchRecipe, EmbeddingPooling, ExecutionContext, K3Error, ModelMetadata, ModelWeights,
    QuantType, X100_CLUSTER_MASK,
};

use crate::kernels_compat::{
    compute_deberta_rel_pos_bucket_compat, deberta_disentangled_attention_compat,
    decode_norm_weights, dequantize_embedding_row, gelu_compat, layernorm_compat, matvec_any,
};
use crate::weights::{
    add_optional_bias, apply_optional_final_norm, attn_k_candidates, attn_norm_candidates,
    attn_output_candidates, attn_q_candidates, attn_v_candidates, embed_encoder_sequence,
    ffn_down_candidates, ffn_norm_candidates, ffn_up_candidates, find_tensor_desc,
    infer_num_labels_from_metadata, maybe_finalize_embedding_output,
    project_classifier_logits_strict, resolve_optional_bias, resolve_optional_tensor,
    resolve_required_norm, resolve_required_tensor,
};

const DEBERTA_NORM_EPS: f32 = 1e-7;

/// Zero-KV `DeBERTa-v3` disentangled-attention encoder classification recipe.
///
/// Enforces a `Q8_0` / `F16` / `BF16` / `F32` precision floor and produces 3-way `[CLS]` softmax
/// probabilities (`BENIGN`, `INJECTION`, `JAILBREAK`) or dense `[CLS]` embeddings when an
/// `embedding_output` buffer is attached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebertaV3Recipe {
    /// Number of encoder layers.
    pub num_layers: usize,
    /// Model hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Per-head dimension (`hidden_dim / n_heads`).
    pub head_dim: usize,
    /// Maximum relative position bucket distance `k` (`2 * k` buckets).
    pub max_relative_positions: usize,
    /// Number of classification labels (`3` by default: `BENIGN`, `INJECTION`, `JAILBREAK`).
    pub num_labels: usize,
}

impl Default for DebertaV3Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl DebertaV3Recipe {
    /// Creates a default `DeBERTa-v3` guardrail classification recipe (`12` layers, `768` hidden,
    /// `12` heads, `3` `[CLS]` classes).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_layers: 12,
            hidden_dim: 768,
            n_heads: 12,
            head_dim: 64,
            max_relative_positions: 256,
            num_labels: 3,
        }
    }

    /// Returns the default `X100`-only cluster affinity bitmask (`0x00FF`) for `DeBERTa-v3` guards.
    #[must_use]
    pub const fn default_cluster_mask() -> u64 {
        X100_CLUSTER_MASK
    }

    /// Hydrates a [`DebertaV3Recipe`] from optional `GGUF` or `SafeTensors` [`ModelMetadata`].
    #[must_use]
    pub fn from_metadata(metadata: Option<&ModelMetadata>) -> Self {
        let mut recipe = Self::new();
        let Some(meta) = metadata else {
            return recipe;
        };

        if meta.layer_count() > 0 {
            recipe.num_layers = meta.layer_count();
        }
        if meta.embedding_dim() > 0 {
            recipe.hidden_dim = meta.embedding_dim();
        } else if let Some(embd) = find_tensor_desc(meta.tensors(), "token_embd.weight")
            && embd.shape().len() == 2
            && embd.shape()[0] > 0
        {
            recipe.hidden_dim = embd.shape()[0];
        } else {
            for hf_name in [
                "deberta.embeddings.word_embeddings.weight",
                "embeddings.word_embeddings.weight",
            ] {
                if let Some(embd) = find_tensor_desc(meta.tensors(), hf_name)
                    && embd.shape().len() == 2
                    && embd.shape()[1] > 0
                {
                    recipe.hidden_dim = embd.shape()[1];
                    break;
                }
            }
        }

        if meta.head_count() > 0 {
            recipe.n_heads = meta.head_count().min(recipe.hidden_dim).max(1);
        } else if recipe.hidden_dim % recipe.n_heads != 0 {
            recipe.n_heads = if recipe.hidden_dim % 4 == 0 { 4 } else { 1 };
        }
        recipe.head_dim = (recipe.hidden_dim / recipe.n_heads).max(1);

        for rel_name in rel_embeddings_candidates() {
            if let Some(rel) = find_tensor_desc(meta.tensors(), &rel_name)
                && rel.shape().len() == 2
            {
                let buckets = if rel.shape()[1] == recipe.hidden_dim {
                    rel.shape()[0]
                } else {
                    rel.shape()[1]
                };
                if buckets >= 2 {
                    recipe.max_relative_positions = buckets / 2;
                    break;
                }
            }
        }

        if let Some(labels) = infer_num_labels_from_metadata(meta, recipe.hidden_dim) {
            recipe.num_labels = labels;
        }

        recipe
    }
}

/// Precomputed disentangled relative-position index table and hydrated position embedding tables
/// for `DeBERTa-v3`.
#[derive(Debug, Clone, PartialEq)]
pub struct DebertaRelPosBuffers {
    /// Flattened `[seq_len, seq_len]` relative position bucket index map.
    pub rel_pos_buckets: Vec<usize>,
    /// Query-side relative position embeddings `[2 * max_rel, attn_dim]`.
    pub q_pos_buf: Vec<f32>,
    /// Key-side relative position embeddings `[2 * max_rel, attn_dim]`.
    pub k_pos_buf: Vec<f32>,
}

fn rel_embeddings_candidates() -> Vec<String> {
    [
        "rel_embeddings.weight",
        "deberta.encoder.rel_embeddings.weight",
        "encoder.rel_embeddings.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

fn rel_embeddings_norm_candidates() -> Vec<String> {
    [
        "rel_embeddings_norm.weight",
        "deberta.encoder.RelEmbeddings.LayerNorm.weight",
        "encoder.RelEmbeddings.LayerNorm.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

impl DebertaRelPosBuffers {
    /// Hydrates relative position bucket indices and `rel_embeddings.weight` strictly from
    /// `weights`, failing closed with [`K3Error::TensorBoundsError`] when absent or undersized.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `seq_len == 0`, `attn_dim == 0`, `max_rel == 0`,
    /// `rel_embeddings.weight` is missing/truncated/corrupted, or its row count is smaller than
    /// `2 * max_rel`.
    pub fn build(
        weights: &dyn ModelWeights,
        seq_len: usize,
        attn_dim: usize,
        max_rel: usize,
    ) -> Result<Self, K3Error> {
        if seq_len == 0 || attn_dim == 0 || max_rel == 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "rel_embeddings.weight".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: format!(
                    "DeBERTa-v3 relative position dimensions must be non-zero (seq={seq_len}, attn_dim={attn_dim}, max_rel={max_rel})"
                ),
            });
        }

        let num_buckets = max_rel * 2;
        let mut rel_pos_buckets = vec![0_usize; seq_len * seq_len];
        for q_idx in 0..seq_len {
            for k_idx in 0..seq_len {
                let bucket = compute_deberta_rel_pos_bucket_compat(q_idx, k_idx, max_rel);
                rel_pos_buckets[q_idx * seq_len + k_idx] = bucket.min(num_buckets - 1);
            }
        }

        let rel_view = resolve_required_tensor(
            weights,
            &rel_embeddings_candidates(),
            "rel_embeddings.weight",
        )?;
        let pos_len =
            num_buckets
                .checked_mul(attn_dim)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: rel_view.name().to_owned(),
                    offset: rel_view.offset(),
                    byte_len: rel_view.byte_len(),
                    buffer_len: rel_view.as_bytes().len(),
                    reason: "rel_embeddings buffer dimension overflowed usize".to_owned(),
                })?;
        if rel_view.num_elements() < pos_len || rel_view.num_elements() % attn_dim != 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: rel_view.name().to_owned(),
                offset: rel_view.offset(),
                byte_len: rel_view.byte_len(),
                buffer_len: rel_view.as_bytes().len(),
                reason: format!(
                    "rel_embeddings elements ({}) must be a multiple of attn_dim ({attn_dim}) and >= {pos_len}",
                    rel_view.num_elements()
                ),
            });
        }

        let mut q_pos_buf = vec![0.0_f32; pos_len];
        let mut row = vec![0.0_f32; attn_dim];
        for b in 0..num_buckets {
            dequantize_embedding_row(&rel_view, b, &mut row)?;
            q_pos_buf[b * attn_dim..(b + 1) * attn_dim].copy_from_slice(&row);
        }

        if let Some(norm_view) =
            resolve_optional_tensor(weights, &rel_embeddings_norm_candidates())?
        {
            let mut norm_w = vec![0.0_f32; attn_dim];
            decode_norm_weights(&norm_view, &mut norm_w)?;
            let norm_b = resolve_optional_bias(weights, norm_view.name(), &[], attn_dim)?;
            let mut normed = vec![0.0_f32; attn_dim];
            for b in 0..num_buckets {
                let dst = &mut q_pos_buf[b * attn_dim..(b + 1) * attn_dim];
                layernorm_compat(
                    &mut normed,
                    dst,
                    &norm_w,
                    norm_b.as_deref(),
                    DEBERTA_NORM_EPS,
                )?;
                dst.copy_from_slice(&normed);
            }
        }

        let k_pos_buf = q_pos_buf.clone();
        Ok(Self {
            rel_pos_buckets,
            q_pos_buf,
            k_pos_buf,
        })
    }
}

fn softmax_probabilities_inplace(logits: &mut [f32]) -> Result<(), K3Error> {
    if logits.is_empty() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "classifier_logits".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: "cannot apply softmax to empty logits slice".to_owned(),
        });
    }
    let mut max_val = f64::NEG_INFINITY;
    for &v in logits.iter() {
        if !v.is_finite() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "deberta-v3".to_owned(),
                reason: format!("non-finite DeBERTa-v3 classification logit: {v}"),
            });
        }
        max_val = max_val.max(f64::from(v));
    }
    let mut sum_exp = 0.0_f64;
    let mut exps = Vec::with_capacity(logits.len());
    for &v in logits.iter() {
        let e = (f64::from(v) - max_val).exp();
        sum_exp += e;
        exps.push(e);
    }
    let inv_sum = if sum_exp > 0.0 && sum_exp.is_finite() {
        1.0_f64 / sum_exp
    } else {
        #[expect(clippy::cast_precision_loss, reason = "small label count fits f64")]
        {
            1.0_f64 / (logits.len() as f64)
        }
    };
    for (slot, e) in logits.iter_mut().zip(exps) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "softmax probability is strictly inside [0.0, 1.0]"
        )]
        {
            *slot = (e * inv_sum) as f32;
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "DeBERTa layer execution passes precomputed disentangled relative position buffers"
)]
fn deberta_run_layer(
    layer: usize,
    seq_len: usize,
    hidden_dim: usize,
    n_heads: usize,
    head_dim: usize,
    rel_pos: &DebertaRelPosBuffers,
    seq_hidden: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    deberta_run_attention_sublayer(
        layer, seq_len, hidden_dim, n_heads, head_dim, rel_pos, seq_hidden, weights,
    )?;
    deberta_run_ffn_sublayer(layer, seq_len, hidden_dim, seq_hidden, weights)
}

#[expect(
    clippy::too_many_arguments,
    reason = "DeBERTa disentangled attention sublayer takes geometry and relative position buffers"
)]
fn deberta_run_attention_sublayer(
    layer: usize,
    seq_len: usize,
    hidden_dim: usize,
    n_heads: usize,
    head_dim: usize,
    rel_pos: &DebertaRelPosBuffers,
    seq_hidden: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let attn_dim = n_heads * head_dim;
    let mut q_buf = vec![0.0_f32; seq_len * attn_dim];
    let mut k_buf = vec![0.0_f32; seq_len * attn_dim];
    let mut v_buf = vec![0.0_f32; seq_len * attn_dim];
    let mut attn_out_buf = vec![0.0_f32; seq_len * attn_dim];
    let mut norm_tok = vec![0.0_f32; hidden_dim];

    let (attn_norm_w, attn_norm_b) = resolve_required_norm(
        weights,
        &attn_norm_candidates(layer),
        &format!("blk.{layer}.attn_norm.weight"),
        hidden_dim,
    )?;
    let q_w = resolve_required_tensor(
        weights,
        &attn_q_candidates(layer),
        &format!("blk.{layer}.attn_q.weight"),
    )?;
    let k_w = resolve_required_tensor(
        weights,
        &attn_k_candidates(layer),
        &format!("blk.{layer}.attn_k.weight"),
    )?;
    let v_w = resolve_required_tensor(
        weights,
        &attn_v_candidates(layer),
        &format!("blk.{layer}.attn_v.weight"),
    )?;
    let q_b = resolve_optional_bias(weights, q_w.name(), &[], attn_dim)?;
    let k_b = resolve_optional_bias(weights, k_w.name(), &[], attn_dim)?;
    let v_b = resolve_optional_bias(weights, v_w.name(), &[], attn_dim)?;

    for pos in 0..seq_len {
        let tok_in = &seq_hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
        layernorm_compat(
            &mut norm_tok,
            tok_in,
            &attn_norm_w,
            attn_norm_b.as_deref(),
            DEBERTA_NORM_EPS,
        )?;
        let dst = pos * attn_dim..(pos + 1) * attn_dim;
        matvec_any(&mut q_buf[dst.clone()], &q_w, &norm_tok)?;
        add_optional_bias(&mut q_buf[dst.clone()], q_b.as_deref(), q_w.name())?;
        matvec_any(&mut k_buf[dst.clone()], &k_w, &norm_tok)?;
        add_optional_bias(&mut k_buf[dst.clone()], k_b.as_deref(), k_w.name())?;
        matvec_any(&mut v_buf[dst.clone()], &v_w, &norm_tok)?;
        add_optional_bias(&mut v_buf[dst], v_b.as_deref(), v_w.name())?;
    }

    deberta_disentangled_attention_compat(
        &mut attn_out_buf,
        &q_buf,
        &k_buf,
        &v_buf,
        &rel_pos.q_pos_buf,
        &rel_pos.k_pos_buf,
        &rel_pos.rel_pos_buckets,
        seq_len,
        n_heads,
        head_dim,
    )?;

    let attn_out_w = resolve_required_tensor(
        weights,
        &attn_output_candidates(layer),
        &format!("blk.{layer}.attn_output.weight"),
    )?;
    let attn_out_b = resolve_optional_bias(weights, attn_out_w.name(), &[], hidden_dim)?;
    let mut attn_proj = vec![0.0_f32; hidden_dim];
    for pos in 0..seq_len {
        let attn_tok = &attn_out_buf[pos * attn_dim..(pos + 1) * attn_dim];
        matvec_any(&mut attn_proj, &attn_out_w, attn_tok)?;
        add_optional_bias(&mut attn_proj, attn_out_b.as_deref(), attn_out_w.name())?;
        let tok_slice = &mut seq_hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
        for (dst, &delta) in tok_slice.iter_mut().zip(attn_proj.iter()) {
            *dst += delta;
        }
    }
    Ok(())
}

fn deberta_run_ffn_sublayer(
    layer: usize,
    seq_len: usize,
    hidden_dim: usize,
    seq_hidden: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let ffn_dim = hidden_dim.saturating_mul(4);
    let (ffn_norm_w, ffn_norm_b) = resolve_required_norm(
        weights,
        &ffn_norm_candidates(layer),
        &format!("blk.{layer}.ffn_norm.weight"),
        hidden_dim,
    )?;
    let ffn_up_w = resolve_required_tensor(
        weights,
        &ffn_up_candidates(layer),
        &format!("blk.{layer}.ffn_up.weight"),
    )?;
    let actual_ffn_dim = ffn_up_w
        .num_elements()
        .checked_div(hidden_dim)
        .filter(|&d| d > 0)
        .unwrap_or(ffn_dim);
    let ffn_up_b = resolve_optional_bias(weights, ffn_up_w.name(), &[], actual_ffn_dim)?;
    let ffn_down_w = resolve_required_tensor(
        weights,
        &ffn_down_candidates(layer),
        &format!("blk.{layer}.ffn_down.weight"),
    )?;
    let ffn_down_b = resolve_optional_bias(weights, ffn_down_w.name(), &[], hidden_dim)?;

    let mut norm_tok = vec![0.0_f32; hidden_dim];
    let mut ffn_up = vec![0.0_f32; actual_ffn_dim];
    let mut ffn_act = vec![0.0_f32; actual_ffn_dim];
    let mut ffn_down = vec![0.0_f32; hidden_dim];

    for pos in 0..seq_len {
        let tok_slice = &mut seq_hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
        layernorm_compat(
            &mut norm_tok,
            tok_slice,
            &ffn_norm_w,
            ffn_norm_b.as_deref(),
            DEBERTA_NORM_EPS,
        )?;
        matvec_any(&mut ffn_up, &ffn_up_w, &norm_tok)?;
        add_optional_bias(&mut ffn_up, ffn_up_b.as_deref(), ffn_up_w.name())?;
        gelu_compat(&mut ffn_act, &ffn_up)?;
        matvec_any(&mut ffn_down, &ffn_down_w, &ffn_act)?;
        add_optional_bias(&mut ffn_down, ffn_down_b.as_deref(), ffn_down_w.name())?;

        for (dst, &delta) in tok_slice.iter_mut().zip(ffn_down.iter()) {
            *dst += delta;
        }
    }

    Ok(())
}

impl ArchRecipe for DebertaV3Recipe {
    fn architecture_id(&self) -> &'static str {
        "deberta-v3"
    }

    fn requires_kv_cache(&self) -> bool {
        false
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::Q8_0 | QuantType::F16 | QuantType::BF16 | QuantType::F32
        )
    }

    fn forward_classify(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let prompt = ctx.prompt_tokens().to_vec();
        if prompt.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "deberta-v3".to_owned(),
                reason: "DeBERTa-v3 classification prompt_tokens cannot be empty".to_owned(),
            });
        }

        let seq_len = prompt.len();
        if seq_len > ctx.n_ctx() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: seq_len,
                max_context_tokens: ctx.n_ctx(),
            });
        }

        if self.num_layers == 0
            || self.hidden_dim == 0
            || self.n_heads == 0
            || self.head_dim == 0
            || self.max_relative_positions == 0
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "deberta-v3".to_owned(),
                reason: format!(
                    "invalid DeBERTa-v3 geometry: layers={}, hidden={}, heads={}, head_dim={}, max_rel={}",
                    self.num_layers,
                    self.hidden_dim,
                    self.n_heads,
                    self.head_dim,
                    self.max_relative_positions
                ),
            });
        }

        let hidden_dim = self.hidden_dim;
        let attn_dim = self.n_heads * self.head_dim;

        let mut seq_hidden = embed_encoder_sequence(weights, &prompt, None, hidden_dim, false)?;
        let rel_pos =
            DebertaRelPosBuffers::build(weights, seq_len, attn_dim, self.max_relative_positions)?;

        for layer in 0..self.num_layers {
            deberta_run_layer(
                layer,
                seq_len,
                hidden_dim,
                self.n_heads,
                self.head_dim,
                &rel_pos,
                &mut seq_hidden,
                weights,
            )?;
        }

        apply_optional_final_norm(
            weights,
            &mut seq_hidden,
            seq_len,
            hidden_dim,
            DEBERTA_NORM_EPS,
        )?;

        let handled_embedding = maybe_finalize_embedding_output(
            ctx,
            &seq_hidden,
            seq_len,
            hidden_dim,
            EmbeddingPooling::Cls,
            weights,
        )?;
        if handled_embedding {
            if ctx.position() == 0 {
                ctx.advance_position(seq_len)?;
            }
            return Ok(());
        }

        let out_labels = self.num_labels.max(1);
        let logits_len = ctx.logits().len();
        if logits_len < out_labels {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "classifier_logits".to_owned(),
                offset: 0,
                byte_len: out_labels.saturating_mul(size_of::<f32>()),
                buffer_len: logits_len.saturating_mul(size_of::<f32>()),
                reason: format!(
                    "DeBERTa-v3 logits buffer length {logits_len} < required num_labels ({out_labels})"
                ),
            });
        }

        let cls_state = &seq_hidden[..hidden_dim];
        let mut class_probs =
            project_classifier_logits_strict(weights, cls_state, Some(out_labels))?;
        softmax_probabilities_inplace(&mut class_probs)?;

        let out_logits = ctx.logits_mut();
        out_logits.fill(0.0);
        out_logits[..out_labels].copy_from_slice(&class_probs);

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
            architecture: "deberta-v3".to_owned(),
            reason:
                "DeBERTa-v3 encoder is single-pass and does not support autoregressive decode_step"
                    .to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modernbert::test_fixtures::{deterministic_f32_bytes, norm_ones_f32_bytes};
    use jero_core::safetensors::test_helpers::create_synthetic_safetensors;
    use jero_core::test_helpers::create_synthetic_gguf_v3;
    use jero_core::{GgufFile, SafeTensorsFile};

    fn build_complete_deberta_gguf(
        num_layers: usize,
        vocab_size: usize,
        hidden_dim: usize,
        max_rel: usize,
        num_labels: usize,
    ) -> Vec<u8> {
        let ffn_dim = hidden_dim * 4;
        let buckets = max_rel * 2;
        let mut specs: Vec<(String, QuantType, Vec<usize>)> = vec![
            (
                "token_embd.weight".to_owned(),
                QuantType::F32,
                vec![hidden_dim, vocab_size],
            ),
            (
                "rel_embeddings.weight".to_owned(),
                QuantType::F32,
                vec![hidden_dim, buckets],
            ),
        ];
        for l in 0..num_layers {
            specs.push((
                format!("blk.{l}.attn_norm.weight"),
                QuantType::F32,
                vec![hidden_dim],
            ));
            specs.push((
                format!("blk.{l}.attn_q.weight"),
                QuantType::F32,
                vec![hidden_dim, hidden_dim],
            ));
            specs.push((
                format!("blk.{l}.attn_k.weight"),
                QuantType::F32,
                vec![hidden_dim, hidden_dim],
            ));
            specs.push((
                format!("blk.{l}.attn_v.weight"),
                QuantType::F32,
                vec![hidden_dim, hidden_dim],
            ));
            specs.push((
                format!("blk.{l}.attn_output.weight"),
                QuantType::F32,
                vec![hidden_dim, hidden_dim],
            ));
            specs.push((
                format!("blk.{l}.ffn_norm.weight"),
                QuantType::F32,
                vec![hidden_dim],
            ));
            specs.push((
                format!("blk.{l}.ffn_up.weight"),
                QuantType::F32,
                vec![hidden_dim, ffn_dim],
            ));
            specs.push((
                format!("blk.{l}.ffn_down.weight"),
                QuantType::F32,
                vec![ffn_dim, hidden_dim],
            ));
        }
        specs.push((
            "classifier.weight".to_owned(),
            QuantType::F32,
            vec![hidden_dim, num_labels],
        ));
        specs.push((
            "classifier.bias".to_owned(),
            QuantType::F32,
            vec![num_labels],
        ));

        let mut chunks = Vec::with_capacity(specs.len());
        for (idx, (name, _, shape)) in specs.iter().enumerate() {
            let elems: usize = shape.iter().product();
            if name.ends_with("norm.weight") {
                chunks.push(norm_ones_f32_bytes(elems));
            } else {
                chunks.push(deterministic_f32_bytes(elems, idx + 1));
            }
        }

        let borrowed: Vec<(&str, QuantType, &[usize])> = specs
            .iter()
            .map(|(n, q, s)| (n.as_str(), *q, s.as_slice()))
            .collect();
        let mut bytes = create_synthetic_gguf_v3("deberta-v3", 64, &borrowed);
        let total_bytes: usize = chunks.iter().map(Vec::len).sum();
        let mut cursor = bytes.len() - total_bytes;
        for chunk in chunks {
            bytes[cursor..cursor + chunk.len()].copy_from_slice(&chunk);
            cursor += chunk.len();
        }
        bytes
    }

    #[test]
    fn test_deberta_v3_recipe_cls_logits_and_quant_floor() {
        let bytes = build_complete_deberta_gguf(1, 16, 8, 8, 3);
        let gguf = GgufFile::parse(&bytes).expect("parse complete deberta-v3 gguf");
        let mut recipe = DebertaV3Recipe::from_metadata(Some(gguf.metadata()));
        recipe.num_layers = 1;
        recipe.hidden_dim = 8;
        recipe.n_heads = 2;
        recipe.head_dim = 4;
        recipe.max_relative_positions = 8;
        recipe.num_labels = 3;

        assert!(!recipe.requires_kv_cache());
        assert_eq!(recipe.architecture_id(), "deberta-v3");
        assert_eq!(DebertaV3Recipe::default_cluster_mask(), X100_CLUSTER_MASK);
        assert!(recipe.supports_quant(QuantType::Q8_0));
        assert!(recipe.supports_quant(QuantType::F16));
        assert!(recipe.supports_quant(QuantType::BF16));
        assert!(recipe.supports_quant(QuantType::F32));
        assert!(!recipe.supports_quant(QuantType::Q4_0));
        assert!(!recipe.supports_quant(QuantType::Q4_K_M));
        assert!(!recipe.supports_quant(QuantType::Q5_K));
        assert!(!recipe.supports_quant(QuantType::Q6_K));

        let prompt_a = [1_u32, 3, 5, 7];
        let prompt_b = [2_u32, 4, 8, 12];
        let mut logits_a = [0.0_f32; 3];
        let mut logits_b = [0.0_f32; 3];

        let mut ctx_a =
            ExecutionContext::new(&prompt_a, 64, &mut logits_a, &mut []).expect("ctx_a");
        recipe
            .forward_classify(&mut ctx_a, &gguf)
            .expect("CLS classification pass A succeeds");
        assert_eq!(ctx_a.position(), prompt_a.len());
        assert!(matches!(
            recipe.decode_step(&mut ctx_a, &gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));
        let sum_a: f32 = logits_a.iter().sum();
        assert!((sum_a - 1.0).abs() < 1e-4);
        assert!(
            logits_a
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        );

        let mut ctx_b =
            ExecutionContext::new(&prompt_b, 64, &mut logits_b, &mut []).expect("ctx_b");
        recipe
            .forward_classify(&mut ctx_b, &gguf)
            .expect("CLS classification pass B succeeds");
        let sum_b: f32 = logits_b.iter().sum();
        assert!((sum_b - 1.0).abs() < 1e-4);

        let diff = logits_a
            .iter()
            .zip(logits_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(diff > 1e-6);
    }

    #[test]
    fn test_deberta_v3_adr045_missing_rel_embeddings_fails_closed() {
        // Empty GGUF fails closed
        let empty = create_synthetic_gguf_v3("deberta-v3", 64, &[]);
        let empty_gguf = GgufFile::parse(&empty).expect("parse empty deberta");
        let mut recipe = DebertaV3Recipe::new();
        recipe.num_layers = 1;
        recipe.hidden_dim = 8;
        recipe.n_heads = 2;
        recipe.head_dim = 4;
        recipe.max_relative_positions = 8;
        let mut logits = [0.0_f32; 3];
        let mut ctx = ExecutionContext::new(&[1_u32, 2], 64, &mut logits, &mut []).expect("ctx");
        assert!(matches!(
            recipe.forward_classify(&mut ctx, &empty_gguf),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // Token embedding present but missing rel_embeddings.weight fails closed
        let embd_only = create_synthetic_gguf_v3(
            "deberta-v3",
            64,
            &[("token_embd.weight", QuantType::F32, &[8, 16][..])],
        );
        let embd_gguf = GgufFile::parse(&embd_only).expect("parse embd-only deberta");
        let err = recipe
            .forward_classify(&mut ctx, &embd_gguf)
            .expect_err("ADR-045: missing rel_embeddings.weight must fail closed");
        assert!(matches!(err, K3Error::TensorBoundsError { .. }));
    }

    #[test]
    fn test_deberta_v3_safetensors_hydration() {
        let hidden_dim = 8_usize;
        let ffn_dim = hidden_dim * 4;
        let buckets = 16_usize;
        let payloads = [
            (
                "deberta.embeddings.word_embeddings.weight",
                vec![16, hidden_dim],
                deterministic_f32_bytes(16 * hidden_dim, 1),
            ),
            (
                "deberta.encoder.rel_embeddings.weight",
                vec![buckets, hidden_dim],
                deterministic_f32_bytes(buckets * hidden_dim, 2),
            ),
            (
                "deberta.encoder.layer.0.attention.output.LayerNorm.weight",
                vec![hidden_dim],
                norm_ones_f32_bytes(hidden_dim),
            ),
            (
                "deberta.encoder.layer.0.attention.self.query_proj.weight",
                vec![hidden_dim, hidden_dim],
                deterministic_f32_bytes(hidden_dim * hidden_dim, 3),
            ),
            (
                "deberta.encoder.layer.0.attention.self.key_proj.weight",
                vec![hidden_dim, hidden_dim],
                deterministic_f32_bytes(hidden_dim * hidden_dim, 4),
            ),
            (
                "deberta.encoder.layer.0.attention.self.value_proj.weight",
                vec![hidden_dim, hidden_dim],
                deterministic_f32_bytes(hidden_dim * hidden_dim, 5),
            ),
            (
                "deberta.encoder.layer.0.attention.output.dense.weight",
                vec![hidden_dim, hidden_dim],
                deterministic_f32_bytes(hidden_dim * hidden_dim, 6),
            ),
            (
                "deberta.encoder.layer.0.output.LayerNorm.weight",
                vec![hidden_dim],
                norm_ones_f32_bytes(hidden_dim),
            ),
            (
                "deberta.encoder.layer.0.intermediate.dense.weight",
                vec![ffn_dim, hidden_dim],
                deterministic_f32_bytes(ffn_dim * hidden_dim, 7),
            ),
            (
                "deberta.encoder.layer.0.output.dense.weight",
                vec![hidden_dim, ffn_dim],
                deterministic_f32_bytes(hidden_dim * ffn_dim, 8),
            ),
            (
                "classifier.weight",
                vec![3, hidden_dim],
                deterministic_f32_bytes(3 * hidden_dim, 9),
            ),
            ("classifier.bias", vec![3], deterministic_f32_bytes(3, 10)),
        ];
        let refs: Vec<(&str, QuantType, &[usize], &[u8])> = payloads
            .iter()
            .map(|(n, s, b)| (*n, QuantType::F32, s.as_slice(), b.as_slice()))
            .collect();
        let st_bytes = create_synthetic_safetensors(
            &[
                ("general.architecture", "deberta-v3"),
                ("num_hidden_layers", "1"),
                ("hidden_size", "8"),
                ("num_attention_heads", "2"),
            ],
            &refs,
        );
        let st = SafeTensorsFile::parse(&st_bytes).expect("parse deberta safetensors");
        let recipe = DebertaV3Recipe::from_metadata(Some(st.metadata()));
        assert_eq!(recipe.num_layers, 1);
        assert_eq!(recipe.hidden_dim, 8);
        assert_eq!(recipe.n_heads, 2);
        assert_eq!(recipe.head_dim, 4);
        assert_eq!(recipe.max_relative_positions, 8);
        assert_eq!(recipe.num_labels, 3);

        let mut logits = [0.0_f32; 3];
        let mut ctx = ExecutionContext::new(&[1_u32, 3, 5], 64, &mut logits, &mut []).expect("ctx");
        recipe
            .forward_classify(&mut ctx, &st)
            .expect("SafeTensors deberta-v3 forward_classify succeeds");
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }
}
