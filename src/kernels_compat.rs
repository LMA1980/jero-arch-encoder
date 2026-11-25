//! Self-contained compute and dequantization dispatchers across `with-kernels` and portable builds.
//!
//! When the `with-kernels` feature is enabled (default), operations dispatch directly to
//! `jero_kernels_rvv` (`RVV 1.0` on `riscv64` and scalar reference kernels elsewhere). When compiled
//! with `--no-default-features`, equivalent portable scalar fallbacks keep all recipes functional.

use jero_core::{K3Error, QuantType, TensorView};

#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    BaselineKernel, RopeType, VectorKernel, apply_rope_inplace_with_layout,
    bidirectional_attention_f32, compute_deberta_rel_pos_bucket,
    deberta_disentangled_attention_f32, dequantize_token_row, dot_product_f32_scalar, geglu_f32,
    gelu_f32, layernorm_f32, pool_and_l2_normalize_f32, pool_sequence_f32,
    rerank_sigmoid_scores_f32, swiglu_f32, vec_dot_bf16, vec_dot_f16, vec_dot_i2_s, vec_dot_iq4_nl,
    vec_dot_q2_k, vec_dot_q3_k, vec_dot_q4_0, vec_dot_q4_1, vec_dot_q5_0, vec_dot_q5_1,
    vec_dot_q5_k, vec_dot_q6_k, vec_dot_q8_0,
};

#[cfg(not(feature = "with-kernels"))]
const Q4_0_BLOCK_ELEMENTS: usize = 32;
#[cfg(not(feature = "with-kernels"))]
const Q4_0_BLOCK_BYTES: usize = 18;
#[cfg(not(feature = "with-kernels"))]
const Q8_0_BLOCK_ELEMENTS: usize = 32;
#[cfg(not(feature = "with-kernels"))]
const Q8_0_BLOCK_BYTES: usize = 34;

/// Decodes IEEE-754 binary16 bits into `f32`, failing closed on `NaN`/`Inf` when `reject_non_finite` is set.
#[inline]
fn decode_f16_bits(bits: u16) -> Option<f32> {
    let sign = u32::from((bits >> 15) & 0x0001);
    let exp = u32::from((bits >> 10) & 0x001F);
    let mant = u32::from(bits & 0x03FF);
    if exp == 31 {
        return None;
    }
    if exp == 0 {
        if mant == 0 {
            return Some(f32::from_bits(sign << 31));
        }
        let mut m = mant;
        let mut e = 127_u32 - 14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        return Some(f32::from_bits((sign << 31) | (e << 23) | (m << 13)));
    }
    let e = exp + 127 - 15;
    Some(f32::from_bits((sign << 31) | (e << 23) | (mant << 13)))
}

/// Decodes `BF16` bits into `f32`, returning `None` on `NaN`/`Inf`.
#[inline]
fn decode_bf16_bits(bits: u16) -> Option<f32> {
    if (bits & 0x7FFF) >= 0x7F80 {
        return None;
    }
    Some(f32::from_bits(u32::from(bits) << 16))
}

#[cfg(not(feature = "with-kernels"))]
#[inline]
fn f16_to_f32_finite(bits: u16) -> f32 {
    decode_f16_bits(bits).unwrap_or(0.0)
}

#[cfg(not(feature = "with-kernels"))]
#[inline]
fn bf16_to_f32_finite(bits: u16) -> f32 {
    decode_bf16_bits(bits).unwrap_or(0.0)
}

/// Computes scalar dot product over two equal-length slices.
#[must_use]
pub(crate) fn dot_product_f32_compat(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "with-kernels")]
    {
        dot_product_f32_scalar(a, b)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let mut sum = 0.0_f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            sum += x * y;
        }
        sum
    }
}

fn validate_matvec_geometry(
    out: &[f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(usize, usize), K3Error> {
    let rows = out.len();
    let cols = input.len();
    if rows == 0 || cols == 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "matvec output and input slices cannot be empty".to_owned(),
        });
    }
    for (idx, &val) in input.iter().enumerate() {
        if !val.is_finite() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!("non-finite matvec input value {val} at index {idx}"),
            });
        }
    }

    let expected_elems = rows
        .checked_mul(cols)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "matvec dimensions overflowed usize".to_owned(),
        })?;
    if weights.num_elements() != expected_elems {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "tensor elements {} != out({rows}) * in({cols})",
                weights.num_elements()
            ),
        });
    }

    let shape = weights.shape();
    if shape.len() == 2 {
        let valid =
            (shape[0] == rows && shape[1] == cols) || (shape[0] == cols && shape[1] == rows);
        if !valid {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!(
                    "2D tensor shape {shape:?} incompatible with out({rows}) x in({cols})"
                ),
            });
        }
    }
    Ok((rows, cols))
}

fn matvec_f32_checked(
    out: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(), K3Error> {
    let raw = weights.as_bytes();
    let required_bytes = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: raw.len(),
            reason: "F32 matvec byte size overflowed usize".to_owned(),
        })?;
    if raw.len() < required_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: raw.len(),
            reason: format!(
                "F32 buffer length {} < required {required_bytes}",
                raw.len()
            ),
        });
    }

    let mut row_buf = vec![0.0_f32; cols];
    for (r, out_slot) in out.iter_mut().enumerate() {
        let base = r * cols * 4;
        for (c, slot) in row_buf.iter_mut().enumerate() {
            let off = base + c * 4;
            let val = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            if !val.is_finite() {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: weights.offset() + off,
                    byte_len: 4,
                    buffer_len: raw.len(),
                    reason: format!("non-finite F32 weight value {val} at row {r}, col {c}"),
                });
            }
            *slot = val;
        }
        *out_slot = dot_product_f32_compat(&row_buf, input);
    }
    Ok(())
}

/// Dispatches matrix-vector multiplication `out = weights * input` across all encoder-supported
/// quantization formats, accepting both GGUF `[cols, rows]` and `SafeTensors` `[rows, cols]` layouts.
pub(crate) fn matvec_any(
    out: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) = validate_matvec_geometry(out, weights, input)?;
    if weights.quant() == QuantType::F32 {
        return matvec_f32_checked(out, weights, input, rows, cols);
    }

    #[cfg(feature = "with-kernels")]
    {
        match weights.quant() {
            QuantType::F16 => vec_dot_f16(out, weights, input),
            QuantType::BF16 => vec_dot_bf16(out, weights, input),
            QuantType::Q4_0 => vec_dot_q4_0(out, weights, input),
            QuantType::Q4_1 => vec_dot_q4_1(out, weights, input),
            QuantType::Q5_0 => vec_dot_q5_0(out, weights, input),
            QuantType::Q5_1 => vec_dot_q5_1(out, weights, input),
            QuantType::Q8_0 => vec_dot_q8_0(out, weights, input),
            QuantType::Q2_K => vec_dot_q2_k(out, weights, input),
            QuantType::Q3_K => vec_dot_q3_k(out, weights, input),
            QuantType::Q4_K_M => BaselineKernel::new().vec_dot_q4_k_m(out, weights, input),
            QuantType::Q5_K => vec_dot_q5_k(out, weights, input),
            QuantType::Q6_K => vec_dot_q6_k(out, weights, input),
            QuantType::IQ4_NL => vec_dot_iq4_nl(out, weights, input),
            QuantType::I2_S => vec_dot_i2_s(out, weights, input),
            other => Err(K3Error::unsupported_tensor_quant(
                "encoder",
                weights.name(),
                other,
            )),
        }
    }

    #[cfg(not(feature = "with-kernels"))]
    {
        matvec_portable(out, weights, input, rows, cols)
    }
}

#[cfg(not(feature = "with-kernels"))]
fn matvec_portable(
    out: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(), K3Error> {
    let mut row_buf = vec![0.0_f32; cols];
    for (r, slot) in out.iter_mut().enumerate().take(rows) {
        dequantize_row_portable(weights, r, &mut row_buf)?;
        *slot = dot_product_f32_compat(&row_buf, input);
    }
    Ok(())
}

/// Decodes a 1D norm/bias tensor (`F32`, `F16`, or `BF16`) into `out`, failing closed on length
/// mismatch, truncated payload, unsupported quant type, or non-finite values.
pub(crate) fn decode_norm_weights(tensor: &TensorView<'_>, out: &mut [f32]) -> Result<(), K3Error> {
    let expected = out.len();
    if expected == 0 || tensor.num_elements() != expected {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!(
                "1D tensor element count {} != expected {expected}",
                tensor.num_elements()
            ),
        });
    }

    let raw = tensor.as_bytes();
    match tensor.quant() {
        QuantType::F32 => {
            let req = expected * 4;
            if raw.len() < req {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: tensor.name().to_owned(),
                    offset: tensor.offset(),
                    byte_len: req,
                    buffer_len: raw.len(),
                    reason: format!("truncated F32 tensor payload ({} < {req})", raw.len()),
                });
            }
            for (i, slot) in out.iter_mut().enumerate() {
                let off = i * 4;
                let val = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                if !val.is_finite() {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: tensor.name().to_owned(),
                        offset: tensor.offset() + off,
                        byte_len: 4,
                        buffer_len: raw.len(),
                        reason: format!("non-finite F32 element {val} at index {i}"),
                    });
                }
                *slot = val;
            }
        }
        QuantType::F16 | QuantType::BF16 => {
            let req = expected * 2;
            if raw.len() < req {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: tensor.name().to_owned(),
                    offset: tensor.offset(),
                    byte_len: req,
                    buffer_len: raw.len(),
                    reason: format!("truncated half tensor payload ({} < {req})", raw.len()),
                });
            }
            let is_f16 = tensor.quant() == QuantType::F16;
            for (i, slot) in out.iter_mut().enumerate() {
                let off = i * 2;
                let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                let decoded = if is_f16 {
                    decode_f16_bits(bits)
                } else {
                    decode_bf16_bits(bits)
                };
                *slot = decoded.ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: tensor.name().to_owned(),
                    offset: tensor.offset() + off,
                    byte_len: 2,
                    buffer_len: raw.len(),
                    reason: format!("non-finite half-precision element at index {i}"),
                })?;
            }
        }
        other => {
            return Err(K3Error::unsupported_tensor_quant(
                "encoder",
                tensor.name(),
                other,
            ));
        }
    }
    Ok(())
}

/// Dequantizes a single row from an embedding or relative-position table, rejecting non-finite outputs.
pub(crate) fn dequantize_embedding_row(
    table: &TensorView<'_>,
    row_idx: usize,
    out: &mut [f32],
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        dequantize_token_row(table, row_idx, out)?;
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        dequantize_row_portable(table, row_idx, out)?;
    }

    for (idx, &val) in out.iter().enumerate() {
        if !val.is_finite() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: table.name().to_owned(),
                offset: table.offset(),
                byte_len: table.byte_len(),
                buffer_len: table.as_bytes().len(),
                reason: format!("non-finite dequantized value {val} at row {row_idx}, col {idx}"),
            });
        }
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
fn dequantize_row_portable(
    table: &TensorView<'_>,
    row_idx: usize,
    out: &mut [f32],
) -> Result<(), K3Error> {
    let cols = out.len();
    if cols == 0 || table.num_elements() == 0 || table.num_elements() % cols != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: table.name().to_owned(),
            offset: table.offset(),
            byte_len: table.byte_len(),
            buffer_len: table.as_bytes().len(),
            reason: format!(
                "tensor elements {} incompatible with row width {cols}",
                table.num_elements()
            ),
        });
    }
    let num_rows = table.num_elements() / cols;
    if row_idx >= num_rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: table.name().to_owned(),
            offset: table.offset(),
            byte_len: table.byte_len(),
            buffer_len: table.as_bytes().len(),
            reason: format!("row_idx {row_idx} out of bounds for {num_rows} rows"),
        });
    }
    let quant = table.quant();
    let blk_elems = quant.block_elements();
    if cols % blk_elems != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: table.name().to_owned(),
            offset: table.offset(),
            byte_len: table.byte_len(),
            buffer_len: table.as_bytes().len(),
            reason: format!("row width {cols} not aligned to {quant} block size {blk_elems}"),
        });
    }
    let row_bytes = (cols / blk_elems) * quant.block_bytes();
    let raw = table.as_bytes();
    let start = row_idx * row_bytes;
    let end = start + row_bytes;
    if end > raw.len() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: table.name().to_owned(),
            offset: table.offset(),
            byte_len: end,
            buffer_len: raw.len(),
            reason: "truncated row payload".to_owned(),
        });
    }
    let slice = &raw[start..end];
    match quant {
        QuantType::F32 => {
            for (dst, ch) in out.iter_mut().zip(slice.chunks_exact(4)) {
                *dst = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
            }
        }
        QuantType::F16 => {
            for (dst, ch) in out.iter_mut().zip(slice.chunks_exact(2)) {
                *dst = f16_to_f32_finite(u16::from_le_bytes([ch[0], ch[1]]));
            }
        }
        QuantType::BF16 => {
            for (dst, ch) in out.iter_mut().zip(slice.chunks_exact(2)) {
                *dst = bf16_to_f32_finite(u16::from_le_bytes([ch[0], ch[1]]));
            }
        }
        QuantType::Q8_0 => {
            for (blk, dst_chunk) in slice
                .chunks_exact(Q8_0_BLOCK_BYTES)
                .zip(out.chunks_exact_mut(Q8_0_BLOCK_ELEMENTS))
            {
                let d = f16_to_f32_finite(u16::from_le_bytes([blk[0], blk[1]]));
                for (dst, &q) in dst_chunk.iter_mut().zip(blk[2..34].iter()) {
                    *dst = d * f32::from(i8::from_ne_bytes([q]));
                }
            }
        }
        QuantType::Q4_0 => {
            for (blk, dst_chunk) in slice
                .chunks_exact(Q4_0_BLOCK_BYTES)
                .zip(out.chunks_exact_mut(Q4_0_BLOCK_ELEMENTS))
            {
                let d = f16_to_f32_finite(u16::from_le_bytes([blk[0], blk[1]]));
                for (j, &packed) in blk[2..18].iter().enumerate() {
                    let q0 = i8::from_ne_bytes([packed & 0x0F]) - 8;
                    let q1 = i8::from_ne_bytes([(packed >> 4) & 0x0F]) - 8;
                    dst_chunk[j] = d * f32::from(q0);
                    dst_chunk[j + 16] = d * f32::from(q1);
                }
            }
        }
        other => {
            return Err(K3Error::unsupported_tensor_quant(
                "encoder",
                table.name(),
                other,
            ));
        }
    }
    Ok(())
}

/// Applies Layer Normalization with affine weight and optional bias.
pub(crate) fn layernorm_compat(
    out: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        layernorm_f32(out, input, Some(weight), bias, eps)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let len = input.len();
        if len == 0 || out.len() != len || weight.len() != len || !eps.is_finite() || eps <= 0.0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "layernorm".to_owned(),
                offset: 0,
                byte_len: size_of_val(out),
                buffer_len: size_of_val(input),
                reason: "invalid layernorm dimensions or epsilon".to_owned(),
            });
        }
        if let Some(b) = bias
            && b.len() != len
        {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "layernorm_bias".to_owned(),
                offset: 0,
                byte_len: size_of_val(b),
                buffer_len: size_of_val(input),
                reason: "layernorm bias length mismatch".to_owned(),
            });
        }
        #[expect(clippy::cast_precision_loss, reason = "dim fits cleanly in f64")]
        let inv_n = 1.0_f64 / (len as f64);
        let mean = input.iter().map(|&x| f64::from(x)).sum::<f64>() * inv_n;
        let var = input
            .iter()
            .map(|&x| {
                let d = f64::from(x) - mean;
                d * d
            })
            .sum::<f64>()
            * inv_n;
        let inv_std = 1.0_f64 / (var + f64::from(eps)).sqrt();
        for i in 0..len {
            let norm = (f64::from(input[i]) - mean) * inv_std;
            let scaled = norm * f64::from(weight[i]);
            let shifted = match bias {
                Some(b) => scaled + f64::from(b[i]),
                None => scaled,
            };
            #[expect(
                clippy::cast_possible_truncation,
                reason = "normalized f64 intermediate converts cleanly back to f32"
            )]
            {
                out[i] = shifted as f32;
            }
        }
        Ok(())
    }
}

/// Evaluates element-wise `GELU_tanh` activation.
pub(crate) fn gelu_compat(out: &mut [f32], input: &[f32]) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        gelu_f32(out, input)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if input.is_empty() || out.len() != input.len() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "gelu".to_owned(),
                offset: 0,
                byte_len: size_of_val(out),
                buffer_len: size_of_val(input),
                reason: "gelu dimension mismatch".to_owned(),
            });
        }
        let k0 = (2.0_f32 / core::f32::consts::PI).sqrt();
        for (dst, &x) in out.iter_mut().zip(input.iter()) {
            let inner = k0 * (x + 0.044_715_f32 * x * x * x);
            *dst = 0.5_f32 * x * (1.0_f32 + inner.tanh());
        }
        Ok(())
    }
}

/// Evaluates `SwiGLU` gated activation (`silu(gate) * up`).
pub(crate) fn swiglu_compat(out: &mut [f32], gate: &[f32], up: &[f32]) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        swiglu_f32(out, gate, up)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if gate.is_empty() || out.len() != gate.len() || up.len() != gate.len() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "swiglu".to_owned(),
                offset: 0,
                byte_len: size_of_val(out),
                buffer_len: size_of_val(gate),
                reason: "swiglu dimension mismatch".to_owned(),
            });
        }
        for i in 0..gate.len() {
            let g = gate[i];
            let silu = if g < -20.0 {
                0.0
            } else if g > 20.0 {
                g
            } else {
                g / (1.0 + (-g).exp())
            };
            out[i] = silu * up[i];
        }
        Ok(())
    }
}

/// Evaluates `GeGLU` gated activation (`gelu(gate) * up`).
pub(crate) fn geglu_compat(out: &mut [f32], gate: &[f32], up: &[f32]) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        geglu_f32(out, gate, up)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        gelu_compat(out, gate)?;
        for (dst, &u) in out.iter_mut().zip(up.iter()) {
            *dst *= u;
        }
        Ok(())
    }
}

/// Applies `NeoX` rotary position embeddings to all heads in `qk` at sequence position `pos`.
pub(crate) fn apply_rope_compat(
    qk: &mut [f32],
    head_dim: usize,
    pos: usize,
    seq_len: usize,
    theta_base: f32,
) -> Result<(), K3Error> {
    let rotary_dim = head_dim - (head_dim % 2);
    if rotary_dim < 2 {
        return Ok(());
    }
    #[cfg(feature = "with-kernels")]
    {
        apply_rope_inplace_with_layout(
            qk,
            head_dim,
            rotary_dim,
            RopeType::NeoX,
            pos,
            seq_len,
            theta_base,
            None,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let _ = seq_len;
        if qk.len() % head_dim != 0 || !theta_base.is_finite() || theta_base <= 0.0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "rope_qk".to_owned(),
                offset: 0,
                byte_len: size_of_val(qk),
                buffer_len: size_of_val(qk),
                reason: "invalid RoPE geometry or theta".to_owned(),
            });
        }
        let half = rotary_dim / 2;
        #[expect(
            clippy::cast_precision_loss,
            reason = "position/dim fit cleanly in f32"
        )]
        let (pos_f, rot_f) = (pos as f32, rotary_dim as f32);
        for head in qk.chunks_exact_mut(head_dim) {
            for i in 0..half {
                #[expect(clippy::cast_precision_loss, reason = "pair index fits cleanly in f32")]
                let freq = pos_f / theta_base.powf((2 * i) as f32 / rot_f);
                let (sin_a, cos_a) = freq.sin_cos();
                let x0 = head[i];
                let x1 = head[i + half];
                head[i] = x0 * cos_a - x1 * sin_a;
                head[i + half] = x0 * sin_a + x1 * cos_a;
            }
        }
        Ok(())
    }
}

/// Computes standard Press et al. `ALiBi` per-head linear distance slopes.
#[must_use]
pub(crate) fn alibi_slopes(n_heads: usize) -> Vec<f32> {
    if n_heads == 0 {
        return Vec::new();
    }
    let closest_pow2 = 1_usize << (usize::BITS - 1 - n_heads.leading_zeros());
    let mut slopes = Vec::with_capacity(n_heads);
    #[expect(
        clippy::cast_precision_loss,
        reason = "head count fits comfortably in f32"
    )]
    let base_exp = -8.0_f32 / (closest_pow2 as f32);
    for h in 0..closest_pow2.min(n_heads) {
        #[expect(
            clippy::cast_precision_loss,
            reason = "head index fits comfortably in f32"
        )]
        let slope = 2.0_f32.powf(base_exp * ((h + 1) as f32));
        slopes.push(slope);
    }
    if closest_pow2 < n_heads {
        #[expect(
            clippy::cast_precision_loss,
            reason = "head count fits comfortably in f32"
        )]
        let extra_exp = -4.0_f32 / (closest_pow2 as f32);
        let remaining = n_heads - closest_pow2;
        for h in 0..remaining {
            #[expect(
                clippy::cast_precision_loss,
                reason = "head index fits comfortably in f32"
            )]
            let slope = 2.0_f32.powf(extra_exp * ((2 * h + 1) as f32));
            slopes.push(slope);
        }
    }
    slopes
}

/// Evaluates unmasked bidirectional attention with optional symmetric window or `ALiBi` slopes.
#[expect(
    clippy::too_many_arguments,
    reason = "attention dispatcher carries Q/K/V, geometry, window, ALiBi flag, and scale"
)]
pub(crate) fn bidirectional_attention_compat(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    window_size: Option<usize>,
    use_alibi: bool,
    scale: f32,
) -> Result<(), K3Error> {
    if !use_alibi {
        #[cfg(feature = "with-kernels")]
        {
            return bidirectional_attention_f32(
                out,
                q,
                k,
                v,
                seq_len,
                n_heads,
                head_dim,
                window_size,
                scale,
            );
        }
    }

    bidirectional_attention_with_optional_alibi(
        out,
        q,
        k,
        v,
        seq_len,
        n_heads,
        head_dim,
        window_size,
        use_alibi,
        scale,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "bidirectional attention implementation requires explicit geometry and Q/K/V buffers"
)]
fn bidirectional_attention_with_optional_alibi(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    window_size: Option<usize>,
    use_alibi: bool,
    scale: f32,
) -> Result<(), K3Error> {
    if seq_len == 0 || n_heads == 0 || head_dim == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "bidirectional_attention".to_owned(),
            reason: "seq_len, n_heads, and head_dim must be non-zero".to_owned(),
        });
    }
    let expected_len = seq_len
        .checked_mul(n_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "bidirectional_attention".to_owned(),
            reason: "attention buffer dimensions overflow usize".to_owned(),
        })?;
    for (name, len) in [
        ("out", out.len()),
        ("q", q.len()),
        ("k", k.len()),
        ("v", v.len()),
    ] {
        if len != expected_len {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: expected_len.saturating_mul(size_of::<f32>()),
                buffer_len: len.saturating_mul(size_of::<f32>()),
                reason: format!("expected {expected_len} elements for {name}, got {len}"),
            });
        }
    }

    let slopes = if use_alibi {
        alibi_slopes(n_heads)
    } else {
        vec![0.0_f32; n_heads]
    };
    let mut scores = vec![0.0_f32; seq_len];

    for i in 0..seq_len {
        let (win_start, win_end) = match window_size {
            Some(w) => (i.saturating_sub(w), i.saturating_add(w).min(seq_len - 1)),
            None => (0, seq_len - 1),
        };
        let active_len = win_end - win_start + 1;

        for (h, &slope) in slopes.iter().enumerate() {
            let q_off = (i * n_heads + h) * head_dim;
            let q_head = &q[q_off..q_off + head_dim];

            let mut max_score = f32::NEG_INFINITY;
            for (rel_idx, j) in (win_start..=win_end).enumerate() {
                let k_off = (j * n_heads + h) * head_dim;
                let k_head = &k[k_off..k_off + head_dim];
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "sequence distance fits comfortably in f32"
                )]
                let alibi_bias = slope * (i.abs_diff(j) as f32);
                let score = dot_product_f32_compat(q_head, k_head) * scale - alibi_bias;
                scores[rel_idx] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            let mut weight_sum = 0.0_f32;
            for score in &mut scores[..active_len] {
                let w = (*score - max_score).exp();
                *score = w;
                weight_sum += w;
            }
            #[expect(
                clippy::cast_precision_loss,
                reason = "active window length fits comfortably in f32"
            )]
            let inv_sum = if weight_sum > 0.0 {
                1.0 / weight_sum
            } else {
                1.0 / (active_len as f32)
            };

            let out_head = &mut out[q_off..q_off + head_dim];
            out_head.fill(0.0);
            for (rel_idx, j) in (win_start..=win_end).enumerate() {
                let weight = scores[rel_idx] * inv_sum;
                if weight == 0.0 {
                    continue;
                }
                let v_off = (j * n_heads + h) * head_dim;
                let v_head = &v[v_off..v_off + head_dim];
                for (dst, &val) in out_head.iter_mut().zip(v_head.iter()) {
                    *dst += val * weight;
                }
            }
        }
    }
    Ok(())
}

/// Computes `DeBERTa-v3` log-bucketed relative position index.
#[must_use]
pub(crate) fn compute_deberta_rel_pos_bucket_compat(
    q_idx: usize,
    k_idx: usize,
    max_rel_pos: usize,
) -> usize {
    #[cfg(feature = "with-kernels")]
    {
        compute_deberta_rel_pos_bucket(q_idx, k_idx, max_rel_pos)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if max_rel_pos == 0 {
            return 0;
        }
        if q_idx == k_idx {
            return max_rel_pos;
        }
        let abs_d = q_idx.abs_diff(k_idx);
        let half = (max_rel_pos / 2).max(1);
        let mag = if abs_d < half || half >= max_rel_pos {
            abs_d.min(max_rel_pos)
        } else {
            #[expect(clippy::cast_precision_loss, reason = "relative distance fits f32")]
            let ratio = ((abs_d as f32) / (half as f32)).ln()
                / (((2 * max_rel_pos) as f32) / (half as f32)).ln();
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss,
                reason = "bucket index bounded inside [half, max_rel_pos]"
            )]
            let scaled =
                half + ((ratio.clamp(0.0, 1.0) * ((max_rel_pos - half) as f32)).floor() as usize);
            scaled.clamp(half, max_rel_pos)
        };
        if q_idx < k_idx {
            max_rel_pos.saturating_sub(mag)
        } else {
            (max_rel_pos + mag).min(2 * max_rel_pos - 1)
        }
    }
}

/// Evaluates `DeBERTa-v3` disentangled attention (`c2c + c2p + p2c`).
#[expect(
    clippy::too_many_arguments,
    reason = "disentangled attention requires content, relative position, bucket table, and geometry"
)]
pub(crate) fn deberta_disentangled_attention_compat(
    out: &mut [f32],
    q_content: &[f32],
    k_content: &[f32],
    v_content: &[f32],
    q_pos: &[f32],
    k_pos: &[f32],
    rel_pos_buckets: &[usize],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        deberta_disentangled_attention_f32(
            out,
            q_content,
            k_content,
            v_content,
            q_pos,
            k_pos,
            rel_pos_buckets,
            seq_len,
            n_heads,
            head_dim,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        deberta_disentangled_attention_portable(
            out,
            q_content,
            k_content,
            v_content,
            q_pos,
            k_pos,
            rel_pos_buckets,
            seq_len,
            n_heads,
            head_dim,
        )
    }
}

#[cfg(not(feature = "with-kernels"))]
#[expect(
    clippy::too_many_arguments,
    reason = "portable DeBERTa disentangled attention matches kernel argument signature"
)]
fn deberta_disentangled_attention_portable(
    out: &mut [f32],
    q_content: &[f32],
    k_content: &[f32],
    v_content: &[f32],
    q_pos: &[f32],
    k_pos: &[f32],
    rel_pos_buckets: &[usize],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<(), K3Error> {
    if seq_len == 0 || n_heads == 0 || head_dim == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "deberta-v3".to_owned(),
            reason: "seq_len, n_heads, and head_dim must be non-zero".to_owned(),
        });
    }
    let hidden = n_heads * head_dim;
    let expected_seq = seq_len * hidden;
    if out.len() != expected_seq
        || q_content.len() != expected_seq
        || k_content.len() != expected_seq
        || v_content.len() != expected_seq
        || rel_pos_buckets.len() != seq_len * seq_len
        || q_pos.len() != k_pos.len()
        || q_pos.is_empty()
        || q_pos.len() % hidden != 0
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "deberta_attention".to_owned(),
            offset: 0,
            byte_len: expected_seq * size_of::<f32>(),
            buffer_len: size_of_val(out),
            reason: "deberta disentangled attention buffer length mismatch".to_owned(),
        });
    }
    let num_buckets = q_pos.len() / hidden;
    for &b in rel_pos_buckets {
        if b >= num_buckets {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "rel_pos_buckets".to_owned(),
                offset: b,
                byte_len: num_buckets,
                buffer_len: q_pos.len(),
                reason: format!("bucket index {b} >= num_buckets {num_buckets}"),
            });
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "head_dim fits cleanly in f32")]
    let scale = 1.0_f32 / (3.0_f32 * (head_dim as f32)).sqrt();
    let mut scores = vec![0.0_f32; seq_len];

    for i in 0..seq_len {
        for h in 0..n_heads {
            let head_base = i * hidden + h * head_dim;
            let qc = &q_content[head_base..head_base + head_dim];
            let mut max_score = f32::NEG_INFINITY;
            for (j, slot) in scores.iter_mut().enumerate() {
                let key_off = j * hidden + h * head_dim;
                let kc = &k_content[key_off..key_off + head_dim];
                let kr = &k_pos[rel_pos_buckets[i * seq_len + j] * hidden + h * head_dim..];
                let qr = &q_pos[rel_pos_buckets[j * seq_len + i] * hidden + h * head_dim..];
                let score = (dot_product_f32_compat(qc, kc)
                    + dot_product_f32_compat(qc, &kr[..head_dim])
                    + dot_product_f32_compat(kc, &qr[..head_dim]))
                    * scale;
                *slot = score;
                max_score = max_score.max(score);
            }
            let mut sum_exp = 0.0_f32;
            for s in &mut scores {
                let e = (*s - max_score).exp();
                *s = e;
                sum_exp += e;
            }
            let inv = if sum_exp > 0.0 {
                1.0 / sum_exp
            } else {
                #[expect(clippy::cast_precision_loss, reason = "seq_len fits cleanly in f32")]
                {
                    1.0 / (seq_len as f32)
                }
            };
            let out_head = &mut out[head_base..head_base + head_dim];
            out_head.fill(0.0);
            for (j, &w) in scores.iter().enumerate() {
                let v_base = j * hidden + h * head_dim;
                for (dst, &val) in out_head
                    .iter_mut()
                    .zip(v_content[v_base..v_base + head_dim].iter())
                {
                    *dst += val * (w * inv);
                }
            }
        }
    }
    Ok(())
}

/// Pools sequence hidden states (`Mean` or `[CLS]`).
pub(crate) fn pool_sequence_compat(
    out: &mut [f32],
    hidden: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    mean_pool: bool,
) {
    #[cfg(feature = "with-kernels")]
    {
        pool_sequence_f32(out, hidden, seq_len, hidden_dim, mean_pool);
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        out.fill(0.0);
        if seq_len == 0 || hidden_dim == 0 || out.is_empty() || hidden.is_empty() {
            return;
        }
        let dim = hidden_dim.min(out.len()).min(hidden.len());
        let active_seq = seq_len.min(hidden.len() / hidden_dim);
        if active_seq == 0 {
            return;
        }
        let pooled = &mut out[..dim];
        if !mean_pool {
            pooled.copy_from_slice(&hidden[..dim]);
            return;
        }
        for s in 0..active_seq {
            let start = s * hidden_dim;
            for (dst, &val) in pooled.iter_mut().zip(hidden[start..start + dim].iter()) {
                *dst += val;
            }
        }
        #[expect(clippy::cast_precision_loss, reason = "seq_len fits cleanly in f32")]
        let inv = 1.0_f32 / (active_seq as f32);
        for v in pooled {
            *v *= inv;
        }
    }
}

/// Pools `[seq_len, hidden_dim]` (`1 = Mean`, `2 = Cls`, `3 = LastToken`), truncates to optional
/// `matryoshka_dim`, and applies safe L2 normalization.
pub(crate) fn pool_and_l2_normalize_compat(
    out: &mut [f32],
    hidden_seq: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    mode_code: u8,
    matryoshka_dim: Option<usize>,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        pool_and_l2_normalize_f32(
            out,
            hidden_seq,
            seq_len,
            hidden_dim,
            mode_code,
            matryoshka_dim,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if !matches!(mode_code, 1..=3) {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "embed_pooling_mode".to_owned(),
                offset: 0,
                byte_len: usize::from(mode_code),
                buffer_len: 3,
                reason: format!("invalid pooling mode_code {mode_code}"),
            });
        }
        let effective_dim = match matryoshka_dim {
            None => hidden_dim,
            Some(d) if d == 0 || d > hidden_dim => {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: "embed_matryoshka_dim".to_owned(),
                    offset: 0,
                    byte_len: d,
                    buffer_len: hidden_dim,
                    reason: format!("matryoshka_dim ({d}) must be in 1..={hidden_dim}"),
                });
            }
            Some(d) => d,
        };
        if seq_len == 0
            || hidden_dim == 0
            || hidden_seq.len() != seq_len.saturating_mul(hidden_dim)
            || out.len() != effective_dim
            || hidden_seq.iter().any(|v| !v.is_finite())
        {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "embed_hidden_seq".to_owned(),
                offset: 0,
                byte_len: size_of_val(out),
                buffer_len: size_of_val(hidden_seq),
                reason: "invalid embedding pooling buffers or non-finite input".to_owned(),
            });
        }
        match mode_code {
            1 => {
                out.copy_from_slice(&hidden_seq[..effective_dim]);
                for s in 1..seq_len {
                    let row = &hidden_seq[s * hidden_dim..s * hidden_dim + effective_dim];
                    for (dst, &val) in out.iter_mut().zip(row.iter()) {
                        *dst += val;
                    }
                }
                #[expect(clippy::cast_precision_loss, reason = "seq_len fits cleanly in f32")]
                let inv = 1.0_f32 / (seq_len as f32);
                for dst in out.iter_mut() {
                    *dst *= inv;
                }
            }
            2 => out.copy_from_slice(&hidden_seq[..effective_dim]),
            3 => {
                let start = (seq_len - 1) * hidden_dim;
                out.copy_from_slice(&hidden_seq[start..start + effective_dim]);
            }
            _ => unreachable!("validated upfront"),
        }
        let sum_sq: f64 = out.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        if sum_sq > 0.0 && sum_sq.is_finite() {
            let inv_norm = 1.0_f64 / sum_sq.sqrt();
            for dst in out.iter_mut() {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "L2-normalized scalar lies inside [-1.0, 1.0]"
                )]
                {
                    *dst = (f64::from(*dst) * inv_norm) as f32;
                }
            }
        } else {
            out.fill(0.0);
        }
        Ok(())
    }
}

/// Applies numerically stable scaled sigmoid `1.0 / (1.0 + exp(-(logit * scale)))`.
pub(crate) fn rerank_sigmoid_scores_compat(
    out: &mut [f32],
    logits: &[f32],
    scale: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        rerank_sigmoid_scores_f32(out, logits, scale)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if logits.is_empty()
            || out.len() != logits.len()
            || !scale.is_finite()
            || scale <= 0.0
            || logits.iter().any(|v| !v.is_finite())
        {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "rerank_logits".to_owned(),
                offset: 0,
                byte_len: size_of_val(out),
                buffer_len: size_of_val(logits),
                reason: "invalid rerank sigmoid buffers, scale, or non-finite logits".to_owned(),
            });
        }
        for (dst, &logit) in out.iter_mut().zip(logits.iter()) {
            let z = logit * scale;
            *dst = if z >= 88.0 {
                1.0
            } else if z <= -88.0 {
                0.0
            } else if z >= 0.0 {
                (1.0 / (1.0 + (-z).exp())).clamp(0.0, 1.0)
            } else {
                let e = z.exp();
                (e / (1.0 + e)).clamp(0.0, 1.0)
            };
        }
        Ok(())
    }
}
