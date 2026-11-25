---
name: jero-arch-encoder
description: Zero-KV bidirectional encoder, dense embedding, and cross-encoder reranker recipes (`BERT`, `ModernBERT`, `Nomic`, `BGE`, `Jina-BERT`, `DeBERTa-v3`) for the SpacemiT K3 platform.
---

# `jero-arch-encoder` Architecture Crate

Provides Zero-KV bidirectional encoders, dense embedding extractors, and cross-encoder rerankers for single-pass inference without persistent KV cache allocation.

## Exported Recipes & Capabilities

- **`ModernBertRecipe`**:
  - `"modernbert"`: Alternating local (`128`-token) and every-3rd-layer global bidirectional attention with `NeoX` `RoPE`, `GeGLU`/`GELU`, and `Mean` pooling.
  - `"bert"` / `"bge"`: Global bidirectional attention, `GELU`, and `[CLS]` pooling.
  - `"nomic-bert"`: Global bidirectional attention with `NeoX` `RoPE`, `SwiGLU`, and `Mean` pooling.
  - `"jina-bert"`: Global bidirectional attention with `ALiBi` linear distance biases, `GeGLU`, and `Mean` pooling.
- **`DebertaV3Recipe` & `DebertaRelPosBuffers`**:
  - `"deberta-v3"`: Disentangled content-to-content / content-to-position / position-to-content attention (`deberta_disentangled_attention_f32`), strict `Q8_0` / `F16` / `BF16` / `F32` precision floor, and `[CLS]` softmax / embedding output.
- **`CrossEncoderRerankRecipe` & `RerankScoringMode`**:
  - Joint `(query, document)` bidirectional encoding (`rerank_pair`, `rerank_documents`, `rank_documents`, and `ArchRecipe::forward_classify`) supporting `SingleLogitSigmoid`, `TwoLogitPositive`, and `Auto` head detection calibrated via `rerank_sigmoid_scores_f32`.
- **Dense Embeddings (`ExecutionContext`)**:
  - Writes `Mean`, `Cls`, and `LastToken` embeddings with optional `matryoshka_dim` truncation via `pool_and_l2_normalize_f32` into `ctx.embedding_output_mut()`.

## ADR-045 Fail-Closed Weight Hydration

- All required weights (`token_embd`, per-layer `attn_norm`, `Q/K/V` or fused `Wqkv`, `attn_output`, `ffn_norm`, `ffn_up`/`ffn_gate`/`ffn_down`, `rel_embeddings` for `DeBERTa-v3`, and classifier/rerank heads) resolve strictly via `ModelWeights::tensor` across both `GgufFile` and `SafeTensorsFile` naming conventions.
- Missing, out-of-bounds, shape-mismatched, or non-finite weights fail closed with `K3Error`.

## Crate Contract & Dependencies

- **License**: `LGPL-3.0-only WITH LGPL-3.0-linking-exception`
- **Upstream Dependencies**: `jero-core` (`ModelWeights`, `ExecutionContext`, `ArchRecipe`, `K3Error`) and optional `jero-kernels-rvv` (enabled by default via `with-kernels`).
- **Safety & Lints**: `#![deny(unsafe_op_in_unsafe_fn)]`, `clippy::pedantic`, and `clippy::undocumented_unsafe_blocks`.

## Local Verification Gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo clippy --no-default-features --all-targets -- -D warnings
cargo check --target riscv64gc-unknown-linux-gnu --all-targets
cargo test --lib --tests
cargo test --no-default-features --lib --tests
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --lib
cargo agents plugin validate Symposium.toml --no-check-crates
```
