//! Zero-KV bidirectional encoder and dense embedding recipes (`modernbert`, `bert`, `nomic-bert`,
//! `bge`, `jina-bert`).
//!
//! Executes non-autoregressive single-pass sequence encoding without allocating persistent KV cache
//! pages, supporting alternating local/global `RoPE` attention (`ModernBERT`), full global `RoPE` +
//! `SwiGLU` (`nomic-bert`), `[CLS]` pooling (`bert`, `bge`), and `ALiBi` + `GeGLU` (`jina-bert`),
//! with strict ADR-045 fail-closed weight hydration and `SafeTensors` / `GGUF` parity.

use jero_core::{
    ArchRecipe, EmbeddingPooling, ExecutionContext, K3Error, ModelMetadata, ModelWeights,
    QuantType, TensorView,
};

use crate::kernels_compat::{
    apply_rope_compat, bidirectional_attention_compat, geglu_compat, gelu_compat, layernorm_compat,
    matvec_any, pool_sequence_compat, swiglu_compat,
};
use crate::weights::{
    add_optional_bias, apply_optional_final_norm, attn_k_candidates, attn_norm_candidates,
    attn_output_candidates, attn_q_candidates, attn_qkv_candidates, attn_v_candidates,
    embed_encoder_sequence, ffn_down_candidates, ffn_gate_candidates, ffn_norm_candidates,
    ffn_up_candidates, find_tensor_desc, infer_linear_out_dim, infer_num_labels_from_metadata,
    maybe_finalize_embedding_output, project_classifier_logits_strict, resolve_optional_bias,
    resolve_optional_norm, resolve_optional_tensor, resolve_required_norm, resolve_required_tensor,
};

/// Feed-forward non-linearity used by bidirectional encoder blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncoderFfnActivation {
    /// Standard Gaussian Error Linear Unit (`GELU_tanh`).
    Gelu,
    /// Gated `SwiGLU` activation (`silu(gate) * up`).
    SwiGlu,
    /// Gated `GeGLU` activation (`gelu(gate) * up`).
    GeGlu,
}

/// Zero-KV bidirectional encoder and dense embedding recipe (`modernbert`, `bert`, `nomic-bert`,
/// `bge`, `jina-bert`, `deberta-v3`).
#[derive(Debug, Clone, PartialEq)]
pub struct ModernBertRecipe {
    /// Canonical `general.architecture` identifier.
    pub architecture: &'static str,
    /// Number of encoder layers.
    pub num_layers: usize,
    /// Model hidden embedding dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Per-head attention dimension (`hidden_dim / n_heads`).
    pub head_dim: usize,
    /// Classification head output class count (`num_labels`).
    pub num_labels: usize,
    /// Symmetric local attention window radius (`None` = global bidirectional on all layers).
    pub local_window: Option<usize>,
    /// Sequence pooling mode for classification: `true` for mean pooling, `false` for `[CLS]`.
    pub mean_pool: bool,
    /// Layer normalization epsilon.
    pub norm_eps: f32,
    /// Feed-forward activation variant (`Gelu`, `SwiGlu`, or `GeGlu`).
    pub ffn_activation: EncoderFfnActivation,
    /// Whether `NeoX` rotary position embeddings are applied to `Q` and `K`.
    pub use_rope: bool,
    /// Base `RoPE` frequency (`theta`).
    pub rope_theta: f32,
    /// Whether `ALiBi` linear attention distance biases are applied (`jina-bert`).
    pub use_alibi: bool,
    /// Default pooling mode when `embedding_output` is attached with [`EmbeddingPooling::None`].
    pub default_pooling: EmbeddingPooling,
}

impl Default for ModernBertRecipe {
    fn default() -> Self {
        Self::new_modernbert()
    }
}

impl ModernBertRecipe {
    /// Creates a default `ModernBERT-base` recipe (22 layers, `768` hidden, alternating `128`-token
    /// local and every-3rd-layer global `RoPE` bidirectional attention, `GeGLU`, mean pooling).
    #[must_use]
    pub const fn new_modernbert() -> Self {
        Self {
            architecture: "modernbert",
            num_layers: 22,
            hidden_dim: 768,
            ffn_dim: 1152,
            n_heads: 12,
            head_dim: 64,
            num_labels: 2,
            local_window: Some(128),
            mean_pool: true,
            norm_eps: 1e-5,
            ffn_activation: EncoderFfnActivation::GeGlu,
            use_rope: true,
            rope_theta: 160_000.0,
            use_alibi: false,
            default_pooling: EmbeddingPooling::Mean,
        }
    }

    /// Creates a default `BERT` encoder classification recipe (`[CLS]` pooling, global attention).
    #[must_use]
    pub const fn new_bert() -> Self {
        Self {
            architecture: "bert",
            num_layers: 12,
            hidden_dim: 768,
            ffn_dim: 3072,
            n_heads: 12,
            head_dim: 64,
            num_labels: 2,
            local_window: None,
            mean_pool: false,
            norm_eps: 1e-12,
            ffn_activation: EncoderFfnActivation::Gelu,
            use_rope: false,
            rope_theta: 10_000.0,
            use_alibi: false,
            default_pooling: EmbeddingPooling::Cls,
        }
    }

    /// Creates a default `DeBERTa-v3` encoder configuration (`[CLS]` pooling, global attention).
    #[must_use]
    pub const fn new_deberta_v3() -> Self {
        Self {
            architecture: "deberta-v3",
            num_layers: 12,
            hidden_dim: 768,
            ffn_dim: 3072,
            n_heads: 12,
            head_dim: 64,
            num_labels: 2,
            local_window: None,
            mean_pool: false,
            norm_eps: 1e-7,
            ffn_activation: EncoderFfnActivation::Gelu,
            use_rope: false,
            rope_theta: 10_000.0,
            use_alibi: false,
            default_pooling: EmbeddingPooling::Cls,
        }
    }

    /// Creates a default `Nomic-BERT` dense embedding recipe (`RoPE` + `SwiGLU` + `Mean` pooling).
    #[must_use]
    pub const fn new_nomic_bert() -> Self {
        Self {
            architecture: "nomic-bert",
            num_layers: 12,
            hidden_dim: 768,
            ffn_dim: 3072,
            n_heads: 12,
            head_dim: 64,
            num_labels: 2,
            local_window: None,
            mean_pool: true,
            norm_eps: 1e-12,
            ffn_activation: EncoderFfnActivation::SwiGlu,
            use_rope: true,
            rope_theta: 10_000.0,
            use_alibi: false,
            default_pooling: EmbeddingPooling::Mean,
        }
    }

    /// Creates a default `BGE` dense embedding recipe (global bidirectional attention + `[CLS]` pooling).
    #[must_use]
    pub const fn new_bge() -> Self {
        Self {
            architecture: "bge",
            num_layers: 12,
            hidden_dim: 768,
            ffn_dim: 3072,
            n_heads: 12,
            head_dim: 64,
            num_labels: 2,
            local_window: None,
            mean_pool: false,
            norm_eps: 1e-12,
            ffn_activation: EncoderFfnActivation::Gelu,
            use_rope: false,
            rope_theta: 10_000.0,
            use_alibi: false,
            default_pooling: EmbeddingPooling::Cls,
        }
    }

    /// Creates a default `Jina-BERT` dense embedding recipe (`ALiBi` + `GeGLU` + `Mean` pooling).
    #[must_use]
    pub const fn new_jina_bert() -> Self {
        Self {
            architecture: "jina-bert",
            num_layers: 12,
            hidden_dim: 768,
            ffn_dim: 3072,
            n_heads: 12,
            head_dim: 64,
            num_labels: 2,
            local_window: None,
            mean_pool: true,
            norm_eps: 1e-12,
            ffn_activation: EncoderFfnActivation::GeGlu,
            use_rope: false,
            rope_theta: 10_000.0,
            use_alibi: true,
            default_pooling: EmbeddingPooling::Mean,
        }
    }

    /// Hydrates a [`ModernBertRecipe`] from `arch` and optional `GGUF` or `SafeTensors`
    /// [`ModelMetadata`].
    #[must_use]
    pub fn from_metadata(arch: &str, metadata: Option<&ModelMetadata>) -> Option<Self> {
        let mut recipe = match arch {
            "modernbert" => Self::new_modernbert(),
            "bert" => Self::new_bert(),
            "deberta-v3" => Self::new_deberta_v3(),
            "nomic-bert" => Self::new_nomic_bert(),
            "bge" => Self::new_bge(),
            "jina-bert" => Self::new_jina_bert(),
            _ => return None,
        };

        let Some(meta) = metadata else {
            return Some(recipe);
        };

        if meta.layer_count() > 0 {
            recipe.num_layers = meta.layer_count();
        }
        recipe.hidden_dim = resolve_hidden_dim_from_metadata(meta, recipe.hidden_dim);

        if meta.head_count() > 0 {
            recipe.n_heads = meta.head_count().min(recipe.hidden_dim).max(1);
        } else if recipe.hidden_dim % recipe.n_heads != 0 {
            recipe.n_heads = if recipe.hidden_dim % 4 == 0 { 4 } else { 1 };
        }
        recipe.head_dim = (recipe.hidden_dim / recipe.n_heads).max(1);

        if let Some(ffn) = resolve_ffn_dim_from_metadata(meta, recipe.hidden_dim) {
            recipe.ffn_dim = ffn;
        } else if meta.embedding_dim() > 0 {
            recipe.ffn_dim = recipe.hidden_dim.saturating_mul(2);
        }

        if let Some(theta) = meta.rope_freq_base()
            && theta.is_finite()
            && theta > 0.0
        {
            recipe.rope_theta = theta;
        }
        if let Some(eps) = meta.rms_norm_eps()
            && eps.is_finite()
            && eps > 0.0
        {
            recipe.norm_eps = eps;
        }
        if let Some(labels) = infer_num_labels_from_metadata(meta, recipe.hidden_dim) {
            recipe.num_labels = labels;
        }

        Some(recipe)
    }

    /// Returns the symmetric bidirectional attention window for `layer_idx` (`None` = global).
    #[must_use]
    pub const fn layer_window(&self, layer_idx: usize) -> Option<usize> {
        if matches!(self.architecture.as_bytes(), b"modernbert") && layer_idx % 3 == 0 {
            None
        } else {
            self.local_window
        }
    }

    /// Returns the effective default [`EmbeddingPooling`] aligned with `mean_pool` and `default_pooling`.
    #[must_use]
    pub const fn effective_default_pooling(&self) -> EmbeddingPooling {
        match self.default_pooling {
            EmbeddingPooling::LastToken => EmbeddingPooling::LastToken,
            EmbeddingPooling::Mean | EmbeddingPooling::Cls | EmbeddingPooling::None => {
                if self.mean_pool {
                    EmbeddingPooling::Mean
                } else {
                    EmbeddingPooling::Cls
                }
            }
        }
    }

    const fn layer_rope_theta(&self, layer_idx: usize) -> f32 {
        if matches!(self.architecture.as_bytes(), b"modernbert")
            && layer_idx % 3 != 0
            && self.rope_theta > 10_000.0
        {
            10_000.0
        } else {
            self.rope_theta
        }
    }

    /// Executes the zero-KV bidirectional encoder stack over `tokens` (with optional `segment_ids`),
    /// returning the final hidden sequence `[tokens.len() * hidden_dim]`.
    pub(crate) fn encode_sequence(
        &self,
        tokens: &[u32],
        segment_ids: Option<&[u32]>,
        n_ctx: usize,
        weights: &dyn ModelWeights,
    ) -> Result<Vec<f32>, K3Error> {
        if tokens.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: "encoder prompt_tokens cannot be empty".to_owned(),
            });
        }
        let seq_len = tokens.len();
        if seq_len > n_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: seq_len,
                max_context_tokens: n_ctx,
            });
        }
        if self.num_layers == 0
            || self.hidden_dim == 0
            || self.ffn_dim == 0
            || self.n_heads == 0
            || self.head_dim == 0
            || !self.norm_eps.is_finite()
            || self.norm_eps <= 0.0
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "invalid encoder geometry: layers={}, hidden={}, ffn={}, heads={}, head_dim={}, eps={}",
                    self.num_layers,
                    self.hidden_dim,
                    self.ffn_dim,
                    self.n_heads,
                    self.head_dim,
                    self.norm_eps
                ),
            });
        }

        let add_sinusoidal_pos = !self.use_rope && !self.use_alibi;
        let mut seq_hidden = embed_encoder_sequence(
            weights,
            tokens,
            segment_ids,
            self.hidden_dim,
            add_sinusoidal_pos,
        )?;

        for layer in 0..self.num_layers {
            self.run_encoder_layer(layer, seq_len, &mut seq_hidden, weights)?;
        }

        apply_optional_final_norm(
            weights,
            &mut seq_hidden,
            seq_len,
            self.hidden_dim,
            self.norm_eps,
        )?;

        Ok(seq_hidden)
    }

    fn run_encoder_layer(
        &self,
        layer: usize,
        seq_len: usize,
        seq_hidden: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.run_encoder_attention_sublayer(layer, seq_len, seq_hidden, weights)?;
        self.run_encoder_ffn_sublayer(layer, seq_len, seq_hidden, weights)
    }

    fn run_encoder_attention_sublayer(
        &self,
        layer: usize,
        seq_len: usize,
        seq_hidden: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = self.hidden_dim;
        let attn_dim = self.n_heads * self.head_dim;
        let mut q_buf = vec![0.0_f32; seq_len * attn_dim];
        let mut k_buf = vec![0.0_f32; seq_len * attn_dim];
        let mut v_buf = vec![0.0_f32; seq_len * attn_dim];
        let mut attn_out_buf = vec![0.0_f32; seq_len * attn_dim];
        let mut norm_tok = vec![0.0_f32; hidden_dim];

        let attn_norm =
            if matches!(self.architecture, "modernbert" | "modernbert-reranker") && layer == 0 {
                resolve_optional_norm(weights, &attn_norm_candidates(layer), hidden_dim)?
            } else {
                Some(resolve_required_norm(
                    weights,
                    &attn_norm_candidates(layer),
                    &format!("blk.{layer}.attn_norm.weight"),
                    hidden_dim,
                )?)
            };
        let qkv_weights = resolve_layer_qkv_weights(weights, layer, attn_dim)?;

        for pos in 0..seq_len {
            let tok_in = &seq_hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
            if let Some((attn_norm_w, attn_norm_b)) = &attn_norm {
                layernorm_compat(
                    &mut norm_tok,
                    tok_in,
                    attn_norm_w,
                    attn_norm_b.as_deref(),
                    self.norm_eps,
                )?;
            } else {
                norm_tok.copy_from_slice(tok_in);
            }

            let dst = pos * attn_dim..(pos + 1) * attn_dim;
            project_qkv_token(
                &qkv_weights,
                &norm_tok,
                &mut q_buf[dst.clone()],
                &mut k_buf[dst.clone()],
                &mut v_buf[dst.clone()],
            )?;

            if self.use_rope {
                let theta = self.layer_rope_theta(layer);
                apply_rope_compat(&mut q_buf[dst.clone()], self.head_dim, pos, seq_len, theta)?;
                apply_rope_compat(&mut k_buf[dst], self.head_dim, pos, seq_len, theta)?;
            }
        }

        #[expect(
            clippy::cast_precision_loss,
            reason = "head_dim fits comfortably in f32"
        )]
        let attn_scale = 1.0_f32 / (self.head_dim as f32).sqrt();
        bidirectional_attention_compat(
            &mut attn_out_buf,
            &q_buf,
            &k_buf,
            &v_buf,
            seq_len,
            self.n_heads,
            self.head_dim,
            self.layer_window(layer),
            self.use_alibi,
            attn_scale,
        )?;

        let out_w = resolve_required_tensor(
            weights,
            &attn_output_candidates(layer),
            &format!("blk.{layer}.attn_output.weight"),
        )?;
        let out_b = resolve_optional_bias(weights, out_w.name(), &[], hidden_dim)?;
        let mut attn_proj = vec![0.0_f32; hidden_dim];
        for pos in 0..seq_len {
            let attn_tok = &attn_out_buf[pos * attn_dim..(pos + 1) * attn_dim];
            matvec_any(&mut attn_proj, &out_w, attn_tok)?;
            add_optional_bias(&mut attn_proj, out_b.as_deref(), out_w.name())?;

            let tok_slice = &mut seq_hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
            for (dst, &delta) in tok_slice.iter_mut().zip(attn_proj.iter()) {
                *dst += delta;
            }
        }
        Ok(())
    }

    fn run_encoder_ffn_sublayer(
        &self,
        layer: usize,
        seq_len: usize,
        seq_hidden: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = self.hidden_dim;
        let (ffn_norm_w, ffn_norm_b) = resolve_required_norm(
            weights,
            &ffn_norm_candidates(layer),
            &format!("blk.{layer}.ffn_norm.weight"),
            hidden_dim,
        )?;
        let ffn_weights = resolve_layer_ffn_weights(
            weights,
            layer,
            self.architecture,
            self.ffn_activation,
            self.ffn_dim,
        )?;

        let mut norm_tok = vec![0.0_f32; hidden_dim];
        let mut ffn_act = vec![0.0_f32; self.ffn_dim];
        let mut ffn_down = vec![0.0_f32; hidden_dim];
        let mut ffn_scratch = FfnScratch::new(self.ffn_dim);

        for pos in 0..seq_len {
            let tok_slice = &mut seq_hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
            layernorm_compat(
                &mut norm_tok,
                tok_slice,
                &ffn_norm_w,
                ffn_norm_b.as_deref(),
                self.norm_eps,
            )?;

            eval_layer_ffn(
                &ffn_weights,
                &norm_tok,
                self.ffn_activation,
                &mut ffn_scratch,
                &mut ffn_act,
                &mut ffn_down,
            )?;

            for (dst, &delta) in tok_slice.iter_mut().zip(ffn_down.iter()) {
                *dst += delta;
            }
        }
        Ok(())
    }
}

struct SeparateQkvWeights<'w> {
    q: TensorView<'w>,
    q_bias: Option<Vec<f32>>,
    k: TensorView<'w>,
    k_bias: Option<Vec<f32>>,
    v: TensorView<'w>,
    v_bias: Option<Vec<f32>>,
}

enum LayerQkvWeights<'w> {
    Separate(Box<SeparateQkvWeights<'w>>),
    Fused {
        qkv: TensorView<'w>,
        qkv_bias: Option<Vec<f32>>,
        attn_dim: usize,
    },
}

fn resolve_layer_qkv_weights(
    weights: &dyn ModelWeights,
    layer: usize,
    attn_dim: usize,
) -> Result<LayerQkvWeights<'_>, K3Error> {
    if let Some(qkv) = resolve_optional_tensor(weights, &attn_qkv_candidates(layer))?
        && !weights.contains_tensor(&format!("blk.{layer}.attn_q.weight"))
    {
        let qkv_bias = resolve_optional_bias(weights, qkv.name(), &[], attn_dim * 3)?;
        return Ok(LayerQkvWeights::Fused {
            qkv,
            qkv_bias,
            attn_dim,
        });
    }

    let q = resolve_required_tensor(
        weights,
        &attn_q_candidates(layer),
        &format!("blk.{layer}.attn_q.weight"),
    )?;
    let k = resolve_required_tensor(
        weights,
        &attn_k_candidates(layer),
        &format!("blk.{layer}.attn_k.weight"),
    )?;
    let v = resolve_required_tensor(
        weights,
        &attn_v_candidates(layer),
        &format!("blk.{layer}.attn_v.weight"),
    )?;
    let q_bias = resolve_optional_bias(weights, q.name(), &[], attn_dim)?;
    let k_bias = resolve_optional_bias(weights, k.name(), &[], attn_dim)?;
    let v_bias = resolve_optional_bias(weights, v.name(), &[], attn_dim)?;

    Ok(LayerQkvWeights::Separate(Box::new(SeparateQkvWeights {
        q,
        q_bias,
        k,
        k_bias,
        v,
        v_bias,
    })))
}

fn project_qkv_token(
    qkv_weights: &LayerQkvWeights<'_>,
    norm_tok: &[f32],
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) -> Result<(), K3Error> {
    match qkv_weights {
        LayerQkvWeights::Separate(sep) => {
            matvec_any(q_out, &sep.q, norm_tok)?;
            add_optional_bias(q_out, sep.q_bias.as_deref(), sep.q.name())?;
            matvec_any(k_out, &sep.k, norm_tok)?;
            add_optional_bias(k_out, sep.k_bias.as_deref(), sep.k.name())?;
            matvec_any(v_out, &sep.v, norm_tok)?;
            add_optional_bias(v_out, sep.v_bias.as_deref(), sep.v.name())?;
        }
        LayerQkvWeights::Fused {
            qkv,
            qkv_bias,
            attn_dim,
        } => {
            let mut fused = vec![0.0_f32; attn_dim * 3];
            matvec_any(&mut fused, qkv, norm_tok)?;
            add_optional_bias(&mut fused, qkv_bias.as_deref(), qkv.name())?;
            q_out.copy_from_slice(&fused[..*attn_dim]);
            k_out.copy_from_slice(&fused[*attn_dim..2 * *attn_dim]);
            v_out.copy_from_slice(&fused[2 * *attn_dim..]);
        }
    }
    Ok(())
}

struct LayerFfnWeights<'w> {
    up: TensorView<'w>,
    up_bias: Option<Vec<f32>>,
    gate: Option<(TensorView<'w>, Option<Vec<f32>>)>,
    down: TensorView<'w>,
    down_bias: Option<Vec<f32>>,
    fused_gate_up: bool,
    ffn_dim: usize,
}

struct FfnScratch {
    up: Vec<f32>,
    gate: Vec<f32>,
    fused: Vec<f32>,
}

impl FfnScratch {
    fn new(ffn_dim: usize) -> Self {
        Self {
            up: vec![0.0_f32; ffn_dim],
            gate: vec![0.0_f32; ffn_dim],
            fused: vec![0.0_f32; ffn_dim * 2],
        }
    }
}

fn resolve_layer_ffn_weights<'w>(
    weights: &'w dyn ModelWeights,
    layer: usize,
    architecture: &str,
    activation: EncoderFfnActivation,
    ffn_dim: usize,
) -> Result<LayerFfnWeights<'w>, K3Error> {
    let up = resolve_required_tensor(
        weights,
        &ffn_up_candidates(layer),
        &format!("blk.{layer}.ffn_up.weight"),
    )?;
    let down = resolve_required_tensor(
        weights,
        &ffn_down_candidates(layer),
        &format!("blk.{layer}.ffn_down.weight"),
    )?;

    let hidden_dim = up
        .num_elements()
        .checked_div(ffn_dim)
        .filter(|&d| d > 0)
        .unwrap_or(1);
    let fused_gate_up = up.num_elements() == (2 * ffn_dim).saturating_mul(hidden_dim)
        || up.name().ends_with("Wi.weight")
        || up.name().ends_with("ffn_gate_up.weight");

    let up_bias = if fused_gate_up {
        resolve_optional_bias(weights, up.name(), &[], 2 * ffn_dim)?
    } else {
        resolve_optional_bias(weights, up.name(), &[], ffn_dim)?
    };
    let down_out_len = down.num_elements().checked_div(ffn_dim).unwrap_or(0);
    let down_bias = if down_out_len > 0 {
        resolve_optional_bias(weights, down.name(), &[], down_out_len)?
    } else {
        None
    };

    let gate = if fused_gate_up {
        None
    } else if let Some(gate_view) = resolve_optional_tensor(weights, &ffn_gate_candidates(layer))? {
        let gate_bias = resolve_optional_bias(weights, gate_view.name(), &[], ffn_dim)?;
        Some((gate_view, gate_bias))
    } else if matches!(activation, EncoderFfnActivation::SwiGlu)
        || (matches!(activation, EncoderFfnActivation::GeGlu) && architecture != "modernbert")
    {
        let missing = resolve_required_tensor(
            weights,
            &ffn_gate_candidates(layer),
            &format!("blk.{layer}.ffn_gate.weight"),
        )?;
        let gate_bias = resolve_optional_bias(weights, missing.name(), &[], ffn_dim)?;
        Some((missing, gate_bias))
    } else {
        None
    };

    Ok(LayerFfnWeights {
        up,
        up_bias,
        gate,
        down,
        down_bias,
        fused_gate_up,
        ffn_dim,
    })
}

fn eval_layer_ffn(
    ffn: &LayerFfnWeights<'_>,
    norm_tok: &[f32],
    activation: EncoderFfnActivation,
    scratch: &mut FfnScratch,
    act_out: &mut [f32],
    down_out: &mut [f32],
) -> Result<(), K3Error> {
    if ffn.fused_gate_up {
        matvec_any(&mut scratch.fused, &ffn.up, norm_tok)?;
        add_optional_bias(&mut scratch.fused, ffn.up_bias.as_deref(), ffn.up.name())?;
        let (first, second) = scratch.fused.split_at(ffn.ffn_dim);
        match activation {
            EncoderFfnActivation::SwiGlu => swiglu_compat(act_out, first, second)?,
            EncoderFfnActivation::GeGlu | EncoderFfnActivation::Gelu => {
                geglu_compat(act_out, first, second)?;
            }
        }
    } else if let Some((gate_view, gate_bias)) = &ffn.gate {
        matvec_any(&mut scratch.gate, gate_view, norm_tok)?;
        add_optional_bias(&mut scratch.gate, gate_bias.as_deref(), gate_view.name())?;
        matvec_any(&mut scratch.up, &ffn.up, norm_tok)?;
        add_optional_bias(&mut scratch.up, ffn.up_bias.as_deref(), ffn.up.name())?;
        match activation {
            EncoderFfnActivation::SwiGlu => {
                swiglu_compat(act_out, &scratch.gate, &scratch.up)?;
            }
            EncoderFfnActivation::GeGlu | EncoderFfnActivation::Gelu => {
                geglu_compat(act_out, &scratch.gate, &scratch.up)?;
            }
        }
    } else {
        matvec_any(&mut scratch.up, &ffn.up, norm_tok)?;
        add_optional_bias(&mut scratch.up, ffn.up_bias.as_deref(), ffn.up.name())?;
        gelu_compat(act_out, &scratch.up)?;
    }

    matvec_any(down_out, &ffn.down, act_out)?;
    add_optional_bias(down_out, ffn.down_bias.as_deref(), ffn.down.name())?;
    Ok(())
}

fn resolve_hidden_dim_from_metadata(meta: &ModelMetadata, default_dim: usize) -> usize {
    if meta.embedding_dim() > 0 {
        return meta.embedding_dim();
    }
    if let Some(embd) = find_tensor_desc(meta.tensors(), "token_embd.weight")
        && embd.shape().len() == 2
        && embd.shape()[0] > 0
    {
        return embd.shape()[0];
    }
    for hf_name in [
        "embeddings.word_embeddings.weight",
        "bert.embeddings.word_embeddings.weight",
        "deberta.embeddings.word_embeddings.weight",
        "roberta.embeddings.word_embeddings.weight",
        "model.embeddings.tok_embeddings.weight",
        "model.embed_tokens.weight",
    ] {
        if let Some(embd) = find_tensor_desc(meta.tensors(), hf_name)
            && embd.shape().len() == 2
            && embd.shape()[1] > 0
        {
            return embd.shape()[1];
        }
    }
    default_dim
}

fn resolve_ffn_dim_from_metadata(meta: &ModelMetadata, hidden_dim: usize) -> Option<usize> {
    if let Some(ffn) = meta.feed_forward_length()
        && ffn > 0
    {
        return Some(ffn);
    }
    for name in [
        "blk.0.ffn_up.weight",
        "blk.0.ffn_gate.weight",
        "encoder.layer.0.intermediate.dense.weight",
        "bert.encoder.layer.0.intermediate.dense.weight",
        "deberta.encoder.layer.0.intermediate.dense.weight",
        "roberta.encoder.layer.0.intermediate.dense.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "encoder.layers.0.mlp.fc11.weight",
        "encoder.layers.0.mlp.fc12.weight",
        "layers.0.mlp.fc1.weight",
    ] {
        if let Some(desc) = find_tensor_desc(meta.tensors(), name)
            && let Some(dim) = infer_linear_out_dim(desc, hidden_dim)
        {
            return Some(dim);
        }
    }
    if let Some(wi) = find_tensor_desc(meta.tensors(), "model.layers.0.mlp.Wi.weight")
        && let Some(fused_dim) = infer_linear_out_dim(wi, hidden_dim)
        && fused_dim >= 2
    {
        return Some(fused_dim / 2);
    }
    None
}

impl ArchRecipe for ModernBertRecipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn requires_kv_cache(&self) -> bool {
        false
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q4_0
                | QuantType::Q4_1
                | QuantType::Q5_0
                | QuantType::Q5_1
                | QuantType::Q8_0
                | QuantType::Q2_K
                | QuantType::Q3_K
                | QuantType::Q4_K_M
                | QuantType::Q5_K
                | QuantType::Q6_K
                | QuantType::IQ4_NL
        )
    }

    fn forward_classify(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let prompt = ctx.prompt_tokens().to_vec();
        let seq_len = prompt.len();
        let seq_hidden = self.encode_sequence(&prompt, None, ctx.n_ctx(), weights)?;

        let handled_embedding = maybe_finalize_embedding_output(
            ctx,
            &seq_hidden,
            seq_len,
            self.hidden_dim,
            self.effective_default_pooling(),
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
                    "classification logits buffer length {logits_len} < required num_labels ({out_labels})"
                ),
            });
        }

        let mut pooled = vec![0.0_f32; self.hidden_dim];
        pool_sequence_compat(
            &mut pooled,
            &seq_hidden,
            seq_len,
            self.hidden_dim,
            self.mean_pool,
        );

        let class_logits = project_classifier_logits_strict(weights, &pooled, Some(out_labels))?;
        let out_logits = ctx.logits_mut();
        out_logits.fill(0.0);
        out_logits[..out_labels].copy_from_slice(&class_logits);

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
            architecture: self.architecture.to_owned(),
            reason: format!(
                "encoder architecture '{}' is single-pass and does not support autoregressive decode_step",
                self.architecture
            ),
        })
    }
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    use jero_core::QuantType;
    use jero_core::safetensors::test_helpers::create_synthetic_safetensors;
    use jero_core::test_helpers::create_synthetic_gguf_v3;

    pub(crate) fn deterministic_f32_bytes(len: usize, seed: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len * 4);
        for idx in 0..len {
            #[expect(clippy::cast_precision_loss, reason = "bounded test index fits f32")]
            let val = ((((idx * 7 + seed * 3 + 5) % 23) as f32) - 11.0) * 0.05;
            out.extend_from_slice(&val.to_le_bytes());
        }
        out
    }

    pub(crate) fn norm_ones_f32_bytes(len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len * 4);
        for idx in 0..len {
            #[expect(clippy::cast_precision_loss, reason = "small offset fits f32")]
            let val = 1.0_f32 + ((idx % 3) as f32) * 0.05;
            out.extend_from_slice(&val.to_le_bytes());
        }
        out
    }

    pub(crate) fn build_complete_gguf_encoder(
        arch: &str,
        num_layers: usize,
        vocab_size: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        num_labels: Option<usize>,
        include_gate: bool,
    ) -> Vec<u8> {
        let mut specs: Vec<(String, QuantType, Vec<usize>)> = Vec::new();
        specs.push((
            "token_embd.weight".to_owned(),
            QuantType::F32,
            vec![hidden_dim, vocab_size],
        ));
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
            if include_gate {
                specs.push((
                    format!("blk.{l}.ffn_gate.weight"),
                    QuantType::F32,
                    vec![hidden_dim, ffn_dim],
                ));
            }
            specs.push((
                format!("blk.{l}.ffn_down.weight"),
                QuantType::F32,
                vec![ffn_dim, hidden_dim],
            ));
        }
        if let Some(labels) = num_labels {
            specs.push((
                "classifier.weight".to_owned(),
                QuantType::F32,
                vec![hidden_dim, labels],
            ));
            specs.push(("classifier.bias".to_owned(), QuantType::F32, vec![labels]));
        }

        let mut payload_chunks: Vec<Vec<u8>> = Vec::with_capacity(specs.len());
        for (idx, (name, _, shape)) in specs.iter().enumerate() {
            let elems: usize = shape.iter().product();
            if name.ends_with("norm.weight") {
                payload_chunks.push(norm_ones_f32_bytes(elems));
            } else {
                payload_chunks.push(deterministic_f32_bytes(elems, idx + 1));
            }
        }

        let borrowed: Vec<(&str, QuantType, &[usize])> = specs
            .iter()
            .map(|(n, q, s)| (n.as_str(), *q, s.as_slice()))
            .collect();
        let mut bytes = create_synthetic_gguf_v3(arch, 128, &borrowed);
        let total_bytes: usize = payload_chunks.iter().map(Vec::len).sum();
        let data_start = bytes.len() - total_bytes;
        let mut cursor = data_start;
        for chunk in payload_chunks {
            bytes[cursor..cursor + chunk.len()].copy_from_slice(&chunk);
            cursor += chunk.len();
        }
        bytes
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "synthetic SafeTensors fixture builder takes explicit geometry parameters"
    )]
    pub(crate) fn build_complete_safetensors_encoder(
        arch: &str,
        num_layers: usize,
        vocab_size: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        n_heads: usize,
        num_labels: Option<usize>,
        include_gate: bool,
    ) -> Vec<u8> {
        let mut payloads: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        payloads.push((
            "embeddings.word_embeddings.weight".to_owned(),
            vec![vocab_size, hidden_dim],
            deterministic_f32_bytes(vocab_size * hidden_dim, 1),
        ));
        let mut seed = 2_usize;
        for l in 0..num_layers {
            payloads.push((
                format!("encoder.layer.{l}.attention.output.LayerNorm.weight"),
                vec![hidden_dim],
                norm_ones_f32_bytes(hidden_dim),
            ));
            seed += 1;
            payloads.push((
                format!("encoder.layer.{l}.attention.self.query.weight"),
                vec![hidden_dim, hidden_dim],
                deterministic_f32_bytes(hidden_dim * hidden_dim, seed),
            ));
            seed += 1;
            payloads.push((
                format!("encoder.layer.{l}.attention.self.key.weight"),
                vec![hidden_dim, hidden_dim],
                deterministic_f32_bytes(hidden_dim * hidden_dim, seed),
            ));
            seed += 1;
            payloads.push((
                format!("encoder.layer.{l}.attention.self.value.weight"),
                vec![hidden_dim, hidden_dim],
                deterministic_f32_bytes(hidden_dim * hidden_dim, seed),
            ));
            seed += 1;
            payloads.push((
                format!("encoder.layer.{l}.attention.output.dense.weight"),
                vec![hidden_dim, hidden_dim],
                deterministic_f32_bytes(hidden_dim * hidden_dim, seed),
            ));
            seed += 1;
            payloads.push((
                format!("encoder.layer.{l}.output.LayerNorm.weight"),
                vec![hidden_dim],
                norm_ones_f32_bytes(hidden_dim),
            ));
            seed += 1;
            payloads.push((
                format!("encoder.layer.{l}.intermediate.dense.weight"),
                vec![ffn_dim, hidden_dim],
                deterministic_f32_bytes(ffn_dim * hidden_dim, seed),
            ));
            seed += 1;
            if include_gate {
                payloads.push((
                    format!("model.layers.{l}.mlp.gate_proj.weight"),
                    vec![ffn_dim, hidden_dim],
                    deterministic_f32_bytes(ffn_dim * hidden_dim, seed),
                ));
                seed += 1;
            }
            payloads.push((
                format!("encoder.layer.{l}.output.dense.weight"),
                vec![hidden_dim, ffn_dim],
                deterministic_f32_bytes(hidden_dim * ffn_dim, seed),
            ));
            seed += 1;
        }
        if let Some(labels) = num_labels {
            payloads.push((
                "classifier.weight".to_owned(),
                vec![labels, hidden_dim],
                deterministic_f32_bytes(labels * hidden_dim, seed),
            ));
            seed += 1;
            payloads.push((
                "classifier.bias".to_owned(),
                vec![labels],
                deterministic_f32_bytes(labels, seed),
            ));
        }

        let layers_str = num_layers.to_string();
        let hidden_str = hidden_dim.to_string();
        let ffn_str = ffn_dim.to_string();
        let heads_str = n_heads.to_string();
        let meta = [
            ("general.architecture", arch),
            ("num_hidden_layers", layers_str.as_str()),
            ("hidden_size", hidden_str.as_str()),
            ("intermediate_size", ffn_str.as_str()),
            ("num_attention_heads", heads_str.as_str()),
        ];
        let borrowed: Vec<(&str, QuantType, &[usize], &[u8])> = payloads
            .iter()
            .map(|(n, s, b)| (n.as_str(), QuantType::F32, s.as_slice(), b.as_slice()))
            .collect();
        create_synthetic_safetensors(&meta, &borrowed)
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::{build_complete_gguf_encoder, build_complete_safetensors_encoder};
    use super::*;
    use jero_core::test_helpers::create_synthetic_gguf_v3;
    use jero_core::{GgufFile, SafeTensorsFile};

    #[test]
    fn test_modernbert_forward_classify_strict_weights() {
        let bytes = build_complete_gguf_encoder("modernbert", 3, 24, 16, 32, Some(4), false);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic modernbert gguf");
        let meta = gguf.metadata();

        let mut recipe =
            ModernBertRecipe::from_metadata("modernbert", Some(meta)).expect("modernbert recipe");
        recipe.num_layers = 3;
        recipe.hidden_dim = 16;
        recipe.ffn_dim = 32;
        recipe.n_heads = 2;
        recipe.head_dim = 8;
        recipe.num_labels = 4;

        assert!(!recipe.requires_kv_cache());
        assert_eq!(recipe.layer_window(0), None);
        assert_eq!(recipe.layer_window(1), Some(128));
        assert_eq!(recipe.layer_window(2), Some(128));

        let prompt_a = [1_u32, 4, 7, 11];
        let prompt_b = [2_u32, 9, 13, 21];
        let mut logits_a = [0.0_f32; 4];
        let mut logits_b = [0.0_f32; 4];

        let mut ctx_a =
            ExecutionContext::new(&prompt_a, 64, &mut logits_a, &mut []).expect("ctx_a");
        recipe
            .forward_classify(&mut ctx_a, &gguf)
            .expect("forward_classify A succeeds with 0 KV bytes");
        assert_eq!(ctx_a.position(), prompt_a.len());
        let decode_err = recipe
            .decode_step(&mut ctx_a, &gguf)
            .expect_err("encoder decode_step must fail");
        assert!(matches!(decode_err, K3Error::CapabilityMismatch { .. }));
        assert!(logits_a.iter().all(|v| v.is_finite()));

        let mut ctx_b =
            ExecutionContext::new(&prompt_b, 64, &mut logits_b, &mut []).expect("ctx_b");
        recipe
            .forward_classify(&mut ctx_b, &gguf)
            .expect("forward_classify B succeeds with 0 KV bytes");
        assert!(logits_b.iter().all(|v| v.is_finite()));

        let max_diff = logits_a
            .iter()
            .zip(logits_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "distinct prompts must yield distinct classification logits: A={logits_a:?}, B={logits_b:?}"
        );
    }

    #[test]
    fn test_adr045_empty_and_partial_gguf_fail_closed() {
        let empty_bytes = create_synthetic_gguf_v3("bert", 64, &[]);
        let empty_gguf = GgufFile::parse(&empty_bytes).expect("parse empty bert gguf");
        let mut bert_recipe = ModernBertRecipe::new_bert();
        bert_recipe.num_layers = 1;
        bert_recipe.hidden_dim = 8;
        bert_recipe.ffn_dim = 16;
        bert_recipe.n_heads = 2;
        bert_recipe.head_dim = 4;
        bert_recipe.num_labels = 3;

        let mut logits = [0.0_f32; 3];
        let mut ctx =
            ExecutionContext::new(&[1_u32, 2, 3], 64, &mut logits, &mut []).expect("bert ctx");
        let err = bert_recipe
            .forward_classify(&mut ctx, &empty_gguf)
            .expect_err("ADR-045: empty GGUF must fail closed");
        assert!(matches!(err, K3Error::TensorBoundsError { .. }));

        // Partial GGUF with only token_embd.weight must still fail closed on missing blk.0.attn_norm.weight
        let partial_bytes = create_synthetic_gguf_v3(
            "bert",
            64,
            &[("token_embd.weight", QuantType::F32, &[8, 16][..])],
        );
        let partial_gguf = GgufFile::parse(&partial_bytes).expect("parse partial bert gguf");
        let mut ctx2 =
            ExecutionContext::new(&[1_u32, 2, 3], 64, &mut logits, &mut []).expect("bert ctx 2");
        let err2 = bert_recipe
            .forward_classify(&mut ctx2, &partial_gguf)
            .expect_err("ADR-045: partial GGUF missing layer norms/weights must fail closed");
        assert!(matches!(err2, K3Error::TensorBoundsError { .. }));

        // Out-of-bounds token ID fails closed
        let complete_bytes = build_complete_gguf_encoder("bert", 1, 16, 8, 16, Some(3), false);
        let complete_gguf = GgufFile::parse(&complete_bytes).expect("parse complete bert gguf");
        let mut ctx_oob =
            ExecutionContext::new(&[1_u32, 99], 64, &mut logits, &mut []).expect("ctx oob");
        let err_oob = bert_recipe
            .forward_classify(&mut ctx_oob, &complete_gguf)
            .expect_err("ADR-045: out-of-range token_id must fail closed");
        assert!(matches!(err_oob, K3Error::TensorBoundsError { .. }));
    }

    #[test]
    fn test_dense_embedding_nomic_bge_jina_and_matryoshka() {
        for (arch, pooling, include_gate) in [
            ("nomic-bert", EmbeddingPooling::Mean, true),
            ("bge", EmbeddingPooling::Cls, false),
            ("jina-bert", EmbeddingPooling::Mean, true),
            ("modernbert", EmbeddingPooling::Mean, true),
            ("bert", EmbeddingPooling::LastToken, false),
        ] {
            let bytes = build_complete_gguf_encoder(arch, 2, 24, 16, 32, None, include_gate);
            let gguf = GgufFile::parse(&bytes).expect("parse embedding gguf");
            let mut recipe =
                ModernBertRecipe::from_metadata(arch, Some(gguf.metadata())).expect("recipe");
            recipe.num_layers = 2;
            recipe.hidden_dim = 16;
            recipe.ffn_dim = 32;
            recipe.n_heads = 2;
            recipe.head_dim = 8;

            // Full 16-dim L2-normalized embedding
            let mut embed_full = [0.0_f32; 16];
            let mut ctx_full = ExecutionContext::new(&[1_u32, 5, 9, 12], 64, &mut [], &mut [])
                .expect("ctx")
                .with_embedding_output(&mut embed_full, pooling, None)
                .expect("attach full embedding");
            recipe
                .forward_classify(&mut ctx_full, &gguf)
                .expect("dense embedding pass succeeds without classifier head");
            let norm_sq: f32 = embed_full.iter().map(|v| v * v).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-4,
                "{arch} full embedding must have unit L2 norm, got sq={norm_sq}"
            );

            // Matryoshka 8-dim truncated L2-normalized embedding
            let mut embed_m8 = [0.0_f32; 8];
            let mut ctx_m8 = ExecutionContext::new(&[1_u32, 5, 9, 12], 64, &mut [], &mut [])
                .expect("ctx m8")
                .with_embedding_output(&mut embed_m8, pooling, Some(8))
                .expect("attach matryoshka embedding");
            recipe
                .forward_classify(&mut ctx_m8, &gguf)
                .expect("matryoshka embedding pass succeeds");
            let norm_sq_m8: f32 = embed_m8.iter().map(|v| v * v).sum();
            assert!(
                (norm_sq_m8 - 1.0).abs() < 1e-4,
                "{arch} Matryoshka-8 embedding must have unit L2 norm, got sq={norm_sq_m8}"
            );

            // Cosine direction alignment between Matryoshka prefix and full prefix
            let full_prefix = &embed_full[..8];
            let prefix_norm: f32 = full_prefix.iter().map(|v| v * v).sum::<f32>().sqrt();
            if prefix_norm > 1e-5 {
                for i in 0..8 {
                    assert!(
                        (embed_m8[i] - full_prefix[i] / prefix_norm).abs() < 1e-4,
                        "{arch} Matryoshka prefix mismatch at dim {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_gguf_vs_safetensors_encoder_and_embedding_parity() {
        for (arch, include_gate) in [
            ("modernbert", false),
            ("nomic-bert", true),
            ("bge", false),
            ("jina-bert", true),
        ] {
            let gguf_bytes = build_complete_gguf_encoder(arch, 1, 16, 8, 16, Some(3), include_gate);
            let st_bytes =
                build_complete_safetensors_encoder(arch, 1, 16, 8, 16, 2, Some(3), include_gate);

            let gguf = GgufFile::parse(&gguf_bytes).expect("parse gguf");
            let st = SafeTensorsFile::parse(&st_bytes).expect("parse safetensors");

            let mut recipe_g =
                ModernBertRecipe::from_metadata(arch, Some(gguf.metadata())).expect("recipe gguf");
            recipe_g.num_layers = 1;
            recipe_g.hidden_dim = 8;
            recipe_g.ffn_dim = 16;
            recipe_g.n_heads = 2;
            recipe_g.head_dim = 4;
            recipe_g.num_labels = 3;

            let mut recipe_s =
                ModernBertRecipe::from_metadata(arch, Some(st.metadata())).expect("recipe st");
            assert_eq!(recipe_s.num_layers, 1);
            assert_eq!(recipe_s.hidden_dim, 8);
            assert_eq!(recipe_s.ffn_dim, 16);
            assert_eq!(recipe_s.n_heads, 2);
            recipe_s.head_dim = 4;
            assert_eq!(recipe_s.num_labels, 3);

            let prompt = [1_u32, 3, 7, 11];
            let mut emb_g = [0.0_f32; 8];
            let mut emb_s = [0.0_f32; 8];
            let mut log_g = [0.0_f32; 3];
            let mut log_s = [0.0_f32; 3];

            let mut ctx_g = ExecutionContext::new(&prompt, 64, &mut log_g, &mut [])
                .expect("ctx g")
                .with_embedding_output(&mut emb_g, recipe_g.effective_default_pooling(), None)
                .expect("emb g");
            let mut ctx_s = ExecutionContext::new(&prompt, 64, &mut log_s, &mut [])
                .expect("ctx s")
                .with_embedding_output(&mut emb_s, recipe_s.effective_default_pooling(), None)
                .expect("emb s");

            recipe_g
                .forward_classify(&mut ctx_g, &gguf)
                .expect("run gguf");
            recipe_s.forward_classify(&mut ctx_s, &st).expect("run st");

            for i in 0..8 {
                assert!(
                    (emb_g[i] - emb_s[i]).abs() < 1e-5,
                    "{arch} GGUF vs SafeTensors embedding mismatch at {i}: {} vs {}",
                    emb_g[i],
                    emb_s[i]
                );
            }
            for i in 0..3 {
                assert!(
                    (log_g[i] - log_s[i]).abs() < 1e-5,
                    "{arch} GGUF vs SafeTensors logit mismatch at {i}: {} vs {}",
                    log_g[i],
                    log_s[i]
                );
            }
        }
    }

    fn build_hf_modernbert_safetensors(include_layer1_attn_norm: bool) -> Vec<u8> {
        use super::test_fixtures::{deterministic_f32_bytes, norm_ones_f32_bytes};
        use jero_core::safetensors::test_helpers::create_synthetic_safetensors;

        let (num_layers, vocab_size, hidden_dim, ffn_dim) = (2_usize, 16_usize, 8_usize, 16_usize);
        let mut payloads: Vec<(String, Vec<usize>, Vec<u8>)> = vec![
            (
                "model.embeddings.tok_embeddings.weight".to_owned(),
                vec![vocab_size, hidden_dim],
                deterministic_f32_bytes(vocab_size * hidden_dim, 1),
            ),
            (
                "model.embeddings.norm.weight".to_owned(),
                vec![hidden_dim],
                norm_ones_f32_bytes(hidden_dim),
            ),
            (
                "model.final_norm.weight".to_owned(),
                vec![hidden_dim],
                norm_ones_f32_bytes(hidden_dim),
            ),
        ];
        let mut seed = 2_usize;
        for l in 0..num_layers {
            // HF ModernBERT-base omits layer 0 attn_norm (identity after embeddings.norm).
            if l > 0 && include_layer1_attn_norm {
                payloads.push((
                    format!("model.layers.{l}.attn_norm.weight"),
                    vec![hidden_dim],
                    norm_ones_f32_bytes(hidden_dim),
                ));
                seed += 1;
            }
            payloads.push((
                format!("model.layers.{l}.attn.Wqkv.weight"),
                vec![hidden_dim * 3, hidden_dim],
                deterministic_f32_bytes(hidden_dim * 3 * hidden_dim, seed),
            ));
            seed += 1;
            payloads.push((
                format!("model.layers.{l}.attn.Wo.weight"),
                vec![hidden_dim, hidden_dim],
                deterministic_f32_bytes(hidden_dim * hidden_dim, seed),
            ));
            seed += 1;
            payloads.push((
                format!("model.layers.{l}.mlp_norm.weight"),
                vec![hidden_dim],
                norm_ones_f32_bytes(hidden_dim),
            ));
            seed += 1;
            payloads.push((
                format!("model.layers.{l}.mlp.Wi.weight"),
                vec![ffn_dim * 2, hidden_dim],
                deterministic_f32_bytes(ffn_dim * 2 * hidden_dim, seed),
            ));
            seed += 1;
            payloads.push((
                format!("model.layers.{l}.mlp.Wo.weight"),
                vec![hidden_dim, ffn_dim],
                deterministic_f32_bytes(hidden_dim * ffn_dim, seed),
            ));
            seed += 1;
        }
        let meta = [
            ("general.architecture", "modernbert"),
            ("num_hidden_layers", "2"),
            ("hidden_size", "8"),
            ("intermediate_size", "16"),
            ("num_attention_heads", "2"),
        ];
        let borrowed: Vec<(&str, QuantType, &[usize], &[u8])> = payloads
            .iter()
            .map(|(n, s, b)| (n.as_str(), QuantType::F32, s.as_slice(), b.as_slice()))
            .collect();
        create_synthetic_safetensors(&meta, &borrowed)
    }

    #[test]
    fn test_hf_modernbert_safetensors_layout_and_layer1_fail_closed() {
        let bytes_ok = build_hf_modernbert_safetensors(true);
        let st_ok = SafeTensorsFile::parse(&bytes_ok).expect("parse hf modernbert st");
        let recipe =
            ModernBertRecipe::from_metadata("modernbert", Some(st_ok.metadata())).expect("recipe");
        assert_eq!(recipe.num_layers, 2);
        assert_eq!(recipe.hidden_dim, 8);
        assert_eq!(recipe.ffn_dim, 16);
        assert_eq!(recipe.n_heads, 2);

        let mut embed_out = [0.0_f32; 8];
        let mut ctx = ExecutionContext::new(&[1_u32, 3, 5], 64, &mut [], &mut [])
            .expect("ctx")
            .with_embedding_output(&mut embed_out, EmbeddingPooling::Mean, None)
            .expect("attach embed");
        recipe
            .forward_classify(&mut ctx, &st_ok)
            .expect("HF ModernBERT SafeTensors layout with omitted layer-0 attn_norm succeeds");
        let norm_sq: f32 = embed_out.iter().map(|v| v * v).sum();
        assert!((norm_sq - 1.0).abs() < 1e-4);

        // Layer > 0 missing attn_norm must still fail closed.
        let bytes_missing_l1 = build_hf_modernbert_safetensors(false);
        let st_missing_l1 = SafeTensorsFile::parse(&bytes_missing_l1).expect("parse incomplete st");
        let mut embed_err = [0.0_f32; 8];
        let mut ctx_err = ExecutionContext::new(&[1_u32, 3, 5], 64, &mut [], &mut [])
            .expect("ctx err")
            .with_embedding_output(&mut embed_err, EmbeddingPooling::Mean, None)
            .expect("attach embed err");
        let err = recipe
            .forward_classify(&mut ctx_err, &st_missing_l1)
            .expect_err("layer 1 missing attn_norm must fail closed");
        assert!(matches!(err, K3Error::TensorBoundsError { .. }));
    }
}
