//! Strict ADR-045 weight hydration and tensor lookup helpers across `GGUF` and `SafeTensors`.
//!
//! Every required weight tensor (token embeddings, layer norms, bidirectional Q/K/V/O projections,
//! FFN projections, and classification/scoring heads) is resolved fail-closed via
//! [`ModelWeights::tensor`]; missing, truncated, shape-mismatched, or non-finite tensors return
//! [`K3Error`] rather than synthesizing fallback outputs.

use jero_core::{
    EmbeddingPooling, ExecutionContext, K3Error, ModelMetadata, ModelWeights, TensorDescriptor,
    TensorView,
};

use crate::kernels_compat::{
    decode_norm_weights, dequantize_embedding_row, layernorm_compat, matvec_any,
    pool_and_l2_normalize_compat,
};

/// Finds a [`TensorDescriptor`] by exact tensor name.
#[must_use]
pub(crate) fn find_tensor_desc<'a>(
    tensors: &'a [TensorDescriptor],
    name: &str,
) -> Option<&'a TensorDescriptor> {
    tensors.iter().find(|t| t.name() == name)
}

/// Resolves the output dimension of a 2D linear weight tensor against `hidden_dim`, handling both
/// `GGUF` `[in_features, out_features]` and `SafeTensors` `[out_features, in_features]` layouts.
#[must_use]
pub(crate) fn infer_linear_out_dim(desc: &TensorDescriptor, hidden_dim: usize) -> Option<usize> {
    let shape = desc.shape();
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return None;
    }
    if hidden_dim > 0 {
        if shape[0] == hidden_dim {
            return Some(shape[1]);
        }
        if shape[1] == hidden_dim {
            return Some(shape[0]);
        }
    }
    Some(shape[1])
}

/// Resolves the first matching required tensor from `candidates`, failing closed with
/// [`K3Error::TensorBoundsError`] if none exist.
pub(crate) fn resolve_required_tensor<'w>(
    weights: &'w dyn ModelWeights,
    candidates: &[String],
    canonical_name: &str,
) -> Result<TensorView<'w>, K3Error> {
    for name in candidates {
        if weights.contains_tensor(name) {
            return weights.tensor(name);
        }
    }
    Err(K3Error::TensorBoundsError {
        tensor_name: canonical_name.to_owned(),
        offset: 0,
        byte_len: 0,
        buffer_len: weights.total_weight_bytes(),
        reason: format!(
            "missing required weight tensor '{canonical_name}' (checked: {})",
            candidates.join(", ")
        ),
    })
}

/// Resolves the first matching optional tensor from `candidates`, propagating any corruption error.
pub(crate) fn resolve_optional_tensor<'w>(
    weights: &'w dyn ModelWeights,
    candidates: &[String],
) -> Result<Option<TensorView<'w>>, K3Error> {
    for name in candidates {
        if weights.contains_tensor(name) {
            return weights.tensor(name).map(Some);
        }
    }
    Ok(None)
}

/// Resolves an optional matching `.bias` tensor for `weight_name` (or explicit `extra_candidates`)
/// and strictly decodes `expected_len` elements.
pub(crate) fn resolve_optional_bias(
    weights: &dyn ModelWeights,
    weight_name: &str,
    extra_candidates: &[String],
    expected_len: usize,
) -> Result<Option<Vec<f32>>, K3Error> {
    let mut candidates = Vec::with_capacity(1 + extra_candidates.len());
    if let Some(stem) = weight_name.strip_suffix(".weight") {
        candidates.push(format!("{stem}.bias"));
    }
    candidates.extend_from_slice(extra_candidates);

    let Some(view) = resolve_optional_tensor(weights, &candidates)? else {
        return Ok(None);
    };
    let mut bias = vec![0.0_f32; expected_len];
    decode_norm_weights(&view, &mut bias)?;
    Ok(Some(bias))
}

pub(crate) type NormPair = (Vec<f32>, Option<Vec<f32>>);

fn decode_norm_pair(
    weights: &dyn ModelWeights,
    view: &TensorView<'_>,
    len: usize,
) -> Result<NormPair, K3Error> {
    let mut weight = vec![0.0_f32; len];
    decode_norm_weights(view, &mut weight)?;
    let bias = resolve_optional_bias(weights, view.name(), &[], len)?;
    Ok((weight, bias))
}

/// Resolves a required 1D normalization weight tensor (and optional matching bias).
pub(crate) fn resolve_required_norm(
    weights: &dyn ModelWeights,
    candidates: &[String],
    canonical_name: &str,
    len: usize,
) -> Result<NormPair, K3Error> {
    let view = resolve_required_tensor(weights, candidates, canonical_name)?;
    decode_norm_pair(weights, &view, len)
}

/// Resolves an optional 1D normalization weight tensor (and optional matching bias).
pub(crate) fn resolve_optional_norm(
    weights: &dyn ModelWeights,
    candidates: &[String],
    len: usize,
) -> Result<Option<NormPair>, K3Error> {
    let Some(view) = resolve_optional_tensor(weights, candidates)? else {
        return Ok(None);
    };
    decode_norm_pair(weights, &view, len).map(Some)
}

/// Adds an optional bias vector element-wise to `dst`, checking exact length.
pub(crate) fn add_optional_bias(
    dst: &mut [f32],
    bias: Option<&[f32]>,
    tensor_name: &str,
) -> Result<(), K3Error> {
    let Some(b) = bias else {
        return Ok(());
    };
    if dst.len() != b.len() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor_name.to_owned(),
            offset: 0,
            byte_len: size_of_val(b),
            buffer_len: size_of_val(dst),
            reason: format!("bias length {} != output length {}", b.len(), dst.len()),
        });
    }
    for (slot, &val) in dst.iter_mut().zip(b.iter()) {
        *slot += val;
    }
    Ok(())
}

/// Candidate names for token embedding matrices across `GGUF` and `SafeTensors` checkpoints.
pub(crate) fn token_embedding_candidates() -> Vec<String> {
    [
        "token_embd.weight",
        "embeddings.word_embeddings.weight",
        "bert.embeddings.word_embeddings.weight",
        "deberta.embeddings.word_embeddings.weight",
        "roberta.embeddings.word_embeddings.weight",
        "model.embeddings.tok_embeddings.weight",
        "model.embed_tokens.weight",
        "embeddings.tok_embeddings.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

/// Candidate names for optional embedding layer norms.
fn embedding_norm_candidates() -> Vec<String> {
    [
        "token_embd_norm.weight",
        "embeddings.LayerNorm.weight",
        "bert.embeddings.LayerNorm.weight",
        "deberta.embeddings.LayerNorm.weight",
        "roberta.embeddings.LayerNorm.weight",
        "embeddings.norm.weight",
        "model.embeddings.norm.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

/// Candidate names for optional learned absolute position embedding tables.
fn position_embedding_candidates() -> Vec<String> {
    [
        "position_embd.weight",
        "embeddings.position_embeddings.weight",
        "bert.embeddings.position_embeddings.weight",
        "roberta.embeddings.position_embeddings.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

/// Candidate names for optional token-type / segment embedding tables.
fn token_type_embedding_candidates() -> Vec<String> {
    [
        "token_types.weight",
        "embeddings.token_type_embeddings.weight",
        "bert.embeddings.token_type_embeddings.weight",
        "deberta.embeddings.token_type_embeddings.weight",
        "roberta.embeddings.token_type_embeddings.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

/// Candidate names for per-layer attention norm weights.
pub(crate) fn attn_norm_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.attn_norm.weight"),
        format!("blk.{layer}.attn_output_norm.weight"),
        format!("encoder.layer.{layer}.attention.output.LayerNorm.weight"),
        format!("bert.encoder.layer.{layer}.attention.output.LayerNorm.weight"),
        format!("deberta.encoder.layer.{layer}.attention.output.LayerNorm.weight"),
        format!("roberta.encoder.layer.{layer}.attention.output.LayerNorm.weight"),
        format!("model.layers.{layer}.attn_norm.weight"),
        format!("model.layers.{layer}.input_layernorm.weight"),
        format!("encoder.layers.{layer}.norm1.weight"),
        format!("layers.{layer}.attn_norm.weight"),
        format!("layers.{layer}.norm1.weight"),
    ]
}

/// Candidate names for per-layer FFN norm weights.
pub(crate) fn ffn_norm_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.ffn_norm.weight"),
        format!("blk.{layer}.layer_output_norm.weight"),
        format!("encoder.layer.{layer}.output.LayerNorm.weight"),
        format!("bert.encoder.layer.{layer}.output.LayerNorm.weight"),
        format!("deberta.encoder.layer.{layer}.output.LayerNorm.weight"),
        format!("roberta.encoder.layer.{layer}.output.LayerNorm.weight"),
        format!("model.layers.{layer}.ffn_norm.weight"),
        format!("model.layers.{layer}.mlp_norm.weight"),
        format!("model.layers.{layer}.post_attention_layernorm.weight"),
        format!("encoder.layers.{layer}.norm2.weight"),
        format!("layers.{layer}.ffn_norm.weight"),
        format!("layers.{layer}.norm2.weight"),
    ]
}

/// Candidate names for per-layer query projection weights.
pub(crate) fn attn_q_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.attn_q.weight"),
        format!("encoder.layer.{layer}.attention.self.query.weight"),
        format!("bert.encoder.layer.{layer}.attention.self.query.weight"),
        format!("deberta.encoder.layer.{layer}.attention.self.query_proj.weight"),
        format!("roberta.encoder.layer.{layer}.attention.self.query.weight"),
        format!("model.layers.{layer}.self_attn.q_proj.weight"),
        format!("layers.{layer}.attn.q_proj.weight"),
    ]
}

/// Candidate names for per-layer key projection weights.
pub(crate) fn attn_k_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.attn_k.weight"),
        format!("encoder.layer.{layer}.attention.self.key.weight"),
        format!("bert.encoder.layer.{layer}.attention.self.key.weight"),
        format!("deberta.encoder.layer.{layer}.attention.self.key_proj.weight"),
        format!("roberta.encoder.layer.{layer}.attention.self.key.weight"),
        format!("model.layers.{layer}.self_attn.k_proj.weight"),
        format!("layers.{layer}.attn.k_proj.weight"),
    ]
}

/// Candidate names for per-layer value projection weights.
pub(crate) fn attn_v_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.attn_v.weight"),
        format!("encoder.layer.{layer}.attention.self.value.weight"),
        format!("bert.encoder.layer.{layer}.attention.self.value.weight"),
        format!("deberta.encoder.layer.{layer}.attention.self.value_proj.weight"),
        format!("roberta.encoder.layer.{layer}.attention.self.value.weight"),
        format!("model.layers.{layer}.self_attn.v_proj.weight"),
        format!("layers.{layer}.attn.v_proj.weight"),
    ]
}

/// Candidate names for fused QKV projection weights.
pub(crate) fn attn_qkv_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.attn_qkv.weight"),
        format!("encoder.layers.{layer}.attn.Wqkv.weight"),
        format!("model.layers.{layer}.attn.Wqkv.weight"),
        format!("layers.{layer}.attn.Wqkv.weight"),
    ]
}

/// Candidate names for attention output projection weights.
pub(crate) fn attn_output_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.attn_output.weight"),
        format!("encoder.layer.{layer}.attention.output.dense.weight"),
        format!("bert.encoder.layer.{layer}.attention.output.dense.weight"),
        format!("deberta.encoder.layer.{layer}.attention.output.dense.weight"),
        format!("roberta.encoder.layer.{layer}.attention.output.dense.weight"),
        format!("model.layers.{layer}.attn.Wo.weight"),
        format!("model.layers.{layer}.self_attn.o_proj.weight"),
        format!("encoder.layers.{layer}.attn.out_proj.weight"),
        format!("layers.{layer}.attn.out_proj.weight"),
    ]
}

/// Candidate names for FFN up (or fused `Wi`) projection weights.
pub(crate) fn ffn_up_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.ffn_up.weight"),
        format!("encoder.layer.{layer}.intermediate.dense.weight"),
        format!("bert.encoder.layer.{layer}.intermediate.dense.weight"),
        format!("deberta.encoder.layer.{layer}.intermediate.dense.weight"),
        format!("roberta.encoder.layer.{layer}.intermediate.dense.weight"),
        format!("model.layers.{layer}.mlp.up_proj.weight"),
        format!("model.layers.{layer}.mlp.Wi.weight"),
        format!("encoder.layers.{layer}.mlp.fc11.weight"),
        format!("layers.{layer}.mlp.fc1.weight"),
    ]
}

/// Candidate names for gated FFN gate projection weights.
pub(crate) fn ffn_gate_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.ffn_gate.weight"),
        format!("model.layers.{layer}.mlp.gate_proj.weight"),
        format!("encoder.layers.{layer}.mlp.fc12.weight"),
        format!("layers.{layer}.mlp.gate_proj.weight"),
    ]
}

/// Candidate names for FFN down projection weights.
pub(crate) fn ffn_down_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.ffn_down.weight"),
        format!("encoder.layer.{layer}.output.dense.weight"),
        format!("bert.encoder.layer.{layer}.output.dense.weight"),
        format!("deberta.encoder.layer.{layer}.output.dense.weight"),
        format!("roberta.encoder.layer.{layer}.output.dense.weight"),
        format!("model.layers.{layer}.mlp.down_proj.weight"),
        format!("model.layers.{layer}.mlp.Wo.weight"),
        format!("encoder.layers.{layer}.mlp.fc2.weight"),
        format!("layers.{layer}.mlp.fc2.weight"),
    ]
}

/// Candidate names for optional final encoder layer norms.
pub(crate) fn final_norm_candidates() -> Vec<String> {
    [
        "final_norm.weight",
        "output_norm.weight",
        "model.norm.weight",
        "model.final_norm.weight",
        "norm.weight",
        "encoder.layer_norm.weight",
        "final_layer_norm.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

/// Candidate names for optional intermediate `[CLS]` pooler dense layers.
fn pooler_dense_candidates() -> Vec<String> {
    [
        "pooler.dense.weight",
        "bert.pooler.dense.weight",
        "deberta.pooler.dense.weight",
        "classifier.dense.weight",
        "cls.pooler.weight",
        "head.dense.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

/// Candidate names for classification / reranking head weights.
pub(crate) fn classifier_head_candidates() -> Vec<String> {
    [
        "classifier.weight",
        "classifier.out_proj.weight",
        "cls.weight",
        "score.weight",
        "output.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

/// Candidate names for classification / reranking head bias vectors.
pub(crate) fn classifier_bias_candidates() -> Vec<String> {
    [
        "classifier.bias",
        "classifier.out_proj.bias",
        "cls.bias",
        "score.bias",
        "output.bias",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

fn validate_embedding_table_rows(
    view: &TensorView<'_>,
    hidden_dim: usize,
) -> Result<usize, K3Error> {
    if hidden_dim == 0 || view.num_elements() == 0 || view.num_elements() % hidden_dim != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: view.name().to_owned(),
            offset: view.offset(),
            byte_len: view.byte_len(),
            buffer_len: view.as_bytes().len(),
            reason: format!(
                "embedding table elements ({}) incompatible with hidden_dim ({hidden_dim})",
                view.num_elements()
            ),
        });
    }
    let rows = view.num_elements() / hidden_dim;
    let shape = view.shape();
    if shape.len() == 2 {
        let valid = (shape[0] == hidden_dim && shape[1] == rows)
            || (shape[1] == hidden_dim && shape[0] == rows);
        if !valid {
            return Err(K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: view.offset(),
                byte_len: view.byte_len(),
                buffer_len: view.as_bytes().len(),
                reason: format!(
                    "embedding table shape {shape:?} incompatible with [{rows}, {hidden_dim}]"
                ),
            });
        }
    }
    Ok(rows)
}

/// Hydrates sequence embeddings `[seq_len, hidden_dim]` strictly from `token_embd.weight` (or
/// `SafeTensors` aliases), optionally applying token-type embeddings, learned position embeddings
/// (or deterministic sinusoidal encoding when no rotary/`ALiBi`/learned position mechanism applies),
/// and optional embedding layer normalization.
pub(crate) fn embed_encoder_sequence(
    weights: &dyn ModelWeights,
    tokens: &[u32],
    segment_ids: Option<&[u32]>,
    hidden_dim: usize,
    add_sinusoidal_pos: bool,
) -> Result<Vec<f32>, K3Error> {
    let embd_view =
        resolve_required_tensor(weights, &token_embedding_candidates(), "token_embd.weight")?;
    let vocab_rows = validate_embedding_table_rows(&embd_view, hidden_dim)?;

    let type_embd = resolve_optional_tensor(weights, &token_type_embedding_candidates())?;
    let type_rows = match &type_embd {
        Some(v) => Some(validate_embedding_table_rows(v, hidden_dim)?),
        None => None,
    };

    let pos_embd = resolve_optional_tensor(weights, &position_embedding_candidates())?;
    let pos_rows = match &pos_embd {
        Some(v) => Some(validate_embedding_table_rows(v, hidden_dim)?),
        None => None,
    };

    let mut seq_hidden = vec![0.0_f32; tokens.len() * hidden_dim];
    let mut aux_row = vec![0.0_f32; hidden_dim];

    for (pos, &token_id) in tokens.iter().enumerate() {
        let tok_idx = usize::try_from(token_id).map_err(|_| K3Error::TensorBoundsError {
            tensor_name: embd_view.name().to_owned(),
            offset: embd_view.offset(),
            byte_len: embd_view.byte_len(),
            buffer_len: embd_view.as_bytes().len(),
            reason: format!("token_id {token_id} overflows usize"),
        })?;
        if tok_idx >= vocab_rows {
            return Err(K3Error::TensorBoundsError {
                tensor_name: embd_view.name().to_owned(),
                offset: embd_view.offset(),
                byte_len: embd_view.byte_len(),
                buffer_len: embd_view.as_bytes().len(),
                reason: format!(
                    "token_id {token_id} out of bounds for embedding vocab size {vocab_rows}"
                ),
            });
        }

        let row = &mut seq_hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
        dequantize_embedding_row(&embd_view, tok_idx, row)?;

        if let (Some(tv), Some(max_types)) = (&type_embd, type_rows) {
            let seg = usize::try_from(segment_ids.and_then(|s| s.get(pos).copied()).unwrap_or(0))
                .map_err(|_| K3Error::TensorBoundsError {
                tensor_name: tv.name().to_owned(),
                offset: tv.offset(),
                byte_len: tv.byte_len(),
                buffer_len: tv.as_bytes().len(),
                reason: "segment_id overflows usize".to_owned(),
            })?;
            if seg >= max_types {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: tv.name().to_owned(),
                    offset: tv.offset(),
                    byte_len: tv.byte_len(),
                    buffer_len: tv.as_bytes().len(),
                    reason: format!("segment_id {seg} >= token_type vocab rows {max_types}"),
                });
            }
            dequantize_embedding_row(tv, seg, &mut aux_row)?;
            for (dst, &val) in row.iter_mut().zip(aux_row.iter()) {
                *dst += val;
            }
        }

        if let (Some(pv), Some(max_pos)) = (&pos_embd, pos_rows) {
            if pos >= max_pos {
                return Err(K3Error::ContextLengthExceeded {
                    requested_tokens: pos + 1,
                    max_context_tokens: max_pos,
                });
            }
            dequantize_embedding_row(pv, pos, &mut aux_row)?;
            for (dst, &val) in row.iter_mut().zip(aux_row.iter()) {
                *dst += val;
            }
        } else if add_sinusoidal_pos {
            add_sinusoidal_position_signal(row, pos, hidden_dim);
        }
    }

    if let Some(norm_view) = resolve_optional_tensor(weights, &embedding_norm_candidates())? {
        let mut norm_w = vec![0.0_f32; hidden_dim];
        decode_norm_weights(&norm_view, &mut norm_w)?;
        let norm_b = resolve_optional_bias(weights, norm_view.name(), &[], hidden_dim)?;
        let mut normed = vec![0.0_f32; hidden_dim];
        for pos in 0..tokens.len() {
            let row = &mut seq_hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
            layernorm_compat(&mut normed, row, &norm_w, norm_b.as_deref(), 1e-12)?;
            row.copy_from_slice(&normed);
        }
    }

    Ok(seq_hidden)
}

fn add_sinusoidal_position_signal(row: &mut [f32], pos: usize, hidden_dim: usize) {
    #[expect(
        clippy::cast_precision_loss,
        reason = "position and dim fit cleanly in f32"
    )]
    let (pos_f, dim_f) = (pos as f32, hidden_dim as f32);
    for (i, slot) in row.iter_mut().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "hidden index fits cleanly in f32"
        )]
        let exp = (2 * (i / 2)) as f32 / dim_f;
        let angle = pos_f / 10_000.0_f32.powf(exp);
        let signal = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        *slot += 0.05_f32 * signal;
    }
}

/// Applies optional final encoder `LayerNorm` (`final_norm` / `model.norm` / etc.) if present.
pub(crate) fn apply_optional_final_norm(
    weights: &dyn ModelWeights,
    seq_hidden: &mut [f32],
    seq_len: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<(), K3Error> {
    let Some(norm_view) = resolve_optional_tensor(weights, &final_norm_candidates())? else {
        return Ok(());
    };
    let mut norm_w = vec![0.0_f32; hidden_dim];
    decode_norm_weights(&norm_view, &mut norm_w)?;
    let norm_b = resolve_optional_bias(weights, norm_view.name(), &[], hidden_dim)?;
    let mut tmp = vec![0.0_f32; hidden_dim];
    for pos in 0..seq_len {
        let row = &mut seq_hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
        layernorm_compat(&mut tmp, row, &norm_w, norm_b.as_deref(), eps)?;
        row.copy_from_slice(&tmp);
    }
    Ok(())
}

/// Resolves and applies optional intermediate pooler projection + `tanh` activation (`[hidden -> hidden]`).
pub(crate) fn apply_optional_pooler_dense(
    weights: &dyn ModelWeights,
    pooled: &mut [f32],
) -> Result<(), K3Error> {
    let hidden_dim = pooled.len();
    let Some(pooler_view) = resolve_optional_tensor(weights, &pooler_dense_candidates())? else {
        return Ok(());
    };
    if pooler_view.num_elements() != hidden_dim.saturating_mul(hidden_dim) {
        return Err(K3Error::TensorBoundsError {
            tensor_name: pooler_view.name().to_owned(),
            offset: pooler_view.offset(),
            byte_len: pooler_view.byte_len(),
            buffer_len: pooler_view.as_bytes().len(),
            reason: format!(
                "pooler tensor elements ({}) != {hidden_dim} * {hidden_dim}",
                pooler_view.num_elements()
            ),
        });
    }
    let mut out = vec![0.0_f32; hidden_dim];
    matvec_any(&mut out, &pooler_view, pooled)?;
    let bias = resolve_optional_bias(weights, pooler_view.name(), &[], hidden_dim)?;
    add_optional_bias(&mut out, bias.as_deref(), pooler_view.name())?;
    for val in &mut out {
        *val = val.tanh();
    }
    pooled.copy_from_slice(&out);
    Ok(())
}

/// Strictly projects pooled hidden states through the classifier head (`classifier.weight` /
/// `classifier.out_proj.weight` / `cls.weight` / `score.weight` / `output.weight` + optional bias),
/// failing closed if no classifier head tensor exists or shapes mismatch.
pub(crate) fn project_classifier_logits_strict(
    weights: &dyn ModelWeights,
    pooled: &[f32],
    expected_labels: Option<usize>,
) -> Result<Vec<f32>, K3Error> {
    let hidden_dim = pooled.len();
    let head_view =
        resolve_required_tensor(weights, &classifier_head_candidates(), "classifier.weight")?;
    let out_labels = match expected_labels {
        Some(n) => n,
        None => resolve_head_label_count(&head_view, hidden_dim)?,
    };
    if out_labels == 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: head_view.name().to_owned(),
            offset: head_view.offset(),
            byte_len: head_view.byte_len(),
            buffer_len: head_view.as_bytes().len(),
            reason: "classifier head output dimension must be > 0".to_owned(),
        });
    }

    let mut pooled_buf = pooled.to_vec();
    apply_optional_pooler_dense(weights, &mut pooled_buf)?;

    let mut logits = vec![0.0_f32; out_labels];
    matvec_any(&mut logits, &head_view, &pooled_buf)?;
    let bias = resolve_optional_bias(
        weights,
        head_view.name(),
        &classifier_bias_candidates(),
        out_labels,
    )?;
    add_optional_bias(&mut logits, bias.as_deref(), "classifier.bias")?;
    Ok(logits)
}

fn resolve_head_label_count(head: &TensorView<'_>, hidden_dim: usize) -> Result<usize, K3Error> {
    if hidden_dim == 0 || head.num_elements() == 0 || head.num_elements() % hidden_dim != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: head.name().to_owned(),
            offset: head.offset(),
            byte_len: head.byte_len(),
            buffer_len: head.as_bytes().len(),
            reason: format!(
                "classifier head elements ({}) not divisible by hidden_dim ({hidden_dim})",
                head.num_elements()
            ),
        });
    }
    let inferred = head.num_elements() / hidden_dim;
    let shape = head.shape();
    if shape.len() == 2 {
        let valid = (shape[0] == hidden_dim && shape[1] == inferred)
            || (shape[1] == hidden_dim && shape[0] == inferred);
        if !valid {
            return Err(K3Error::TensorBoundsError {
                tensor_name: head.name().to_owned(),
                offset: head.offset(),
                byte_len: head.byte_len(),
                buffer_len: head.as_bytes().len(),
                reason: format!(
                    "classifier shape {shape:?} incompatible with hidden_dim ({hidden_dim})"
                ),
            });
        }
    }
    Ok(inferred)
}

/// Checks whether any classifier head tensor is present in `weights`.
#[must_use]
pub(crate) fn has_classifier_head(weights: &dyn ModelWeights) -> bool {
    classifier_head_candidates()
        .iter()
        .any(|name| weights.contains_tensor(name))
}

/// Infers `num_labels` from `ModelMetadata` tensor descriptors if a classifier weight or bias exists.
#[must_use]
pub(crate) fn infer_num_labels_from_metadata(
    meta: &ModelMetadata,
    hidden_dim: usize,
) -> Option<usize> {
    for bias_name in classifier_bias_candidates() {
        if let Some(bias) = find_tensor_desc(meta.tensors(), &bias_name)
            && !bias.shape().is_empty()
            && bias.shape()[0] > 0
        {
            return Some(bias.shape()[0]);
        }
    }
    for head_name in classifier_head_candidates() {
        if let Some(head) = find_tensor_desc(meta.tensors(), &head_name)
            && let Some(labels) = infer_linear_out_dim(head, hidden_dim)
        {
            return Some(labels);
        }
    }
    None
}

/// Writes pooled L2-normalized embeddings into `ctx.embedding_output_mut()` (with optional
/// Matryoshka truncation) when embedding output or pooling is requested, returning `Ok(true)` if
/// embedding mode handled the output or `Ok(false)` if classification mode should run instead.
pub(crate) fn maybe_finalize_embedding_output(
    ctx: &mut ExecutionContext<'_>,
    seq_hidden: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    default_pooling: EmbeddingPooling,
    weights: &dyn ModelWeights,
) -> Result<bool, K3Error> {
    let requested_pooling = ctx.embedding_pooling();
    let matryoshka_dim = ctx.matryoshka_dim();

    let Some(embed_out) = ctx.embedding_output_mut() else {
        if requested_pooling != EmbeddingPooling::None {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "embedding_output".to_owned(),
                offset: 0,
                byte_len: hidden_dim.saturating_mul(size_of::<f32>()),
                buffer_len: 0,
                reason: format!(
                    "embedding pooling '{}' requested without an attached embedding_output buffer",
                    requested_pooling.as_str()
                ),
            });
        }
        return Ok(false);
    };

    let active_pooling = if requested_pooling == EmbeddingPooling::None {
        default_pooling
    } else {
        requested_pooling
    };
    let mode_code = active_pooling
        .mode_code()
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "encoder".to_owned(),
            reason: "embedding output requires Mean, Cls, or LastToken pooling".to_owned(),
        })?;

    let effective_dim = match matryoshka_dim {
        Some(d) if d == 0 || d > hidden_dim => {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "embed_matryoshka_dim".to_owned(),
                offset: 0,
                byte_len: d,
                buffer_len: hidden_dim,
                reason: format!("matryoshka_dim ({d}) must lie within 1..={hidden_dim}"),
            });
        }
        Some(d) => d,
        None => hidden_dim,
    };

    if embed_out.len() < effective_dim {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "embedding_output".to_owned(),
            offset: 0,
            byte_len: effective_dim.saturating_mul(size_of::<f32>()),
            buffer_len: size_of_val(embed_out),
            reason: format!(
                "embedding_output length {} < required effective_dim ({effective_dim})",
                embed_out.len()
            ),
        });
    }

    pool_and_l2_normalize_compat(
        &mut embed_out[..effective_dim],
        seq_hidden,
        seq_len,
        hidden_dim,
        mode_code,
        matryoshka_dim,
    )?;
    if embed_out.len() > effective_dim {
        embed_out[effective_dim..].fill(0.0);
    }

    // In embedding mode, optionally populate logits only if a classifier head is present and a
    // logits buffer was attached; pure embedding models omit classifier weights.
    if ctx.compute_logits() && !ctx.logits().is_empty() {
        if has_classifier_head(weights) {
            let mut pooled = vec![0.0_f32; hidden_dim];
            let pool_cls = active_pooling == EmbeddingPooling::Mean;
            crate::kernels_compat::pool_sequence_compat(
                &mut pooled,
                seq_hidden,
                seq_len,
                hidden_dim,
                pool_cls,
            );
            let logits = project_classifier_logits_strict(weights, &pooled, None)?;
            let out_logits = ctx.logits_mut();
            let copy_len = out_logits.len().min(logits.len());
            out_logits.fill(0.0);
            out_logits[..copy_len].copy_from_slice(&logits[..copy_len]);
        } else {
            ctx.logits_mut().fill(0.0);
        }
    }

    Ok(true)
}
