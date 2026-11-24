---
name: jero-arch-encoder
description: Zero-KV bidirectional encoder, dense embedding, and cross-encoder reranker recipes (`BERT`, `ModernBERT`, `Nomic`, `BGE`, `Jina-BERT`, `DeBERTa-v3`) for the SpacemiT K3 platform.
---

# `jero-arch-encoder` Architecture Crate

Provides Zero-KV bidirectional encoders (`ModernBertRecipe`, `DebertaV3Recipe`), dense embedding pooling (`nomic-bert`, `bge`, `jina-bert`), and `CrossEncoderRerankRecipe` for K3 inference.

## Crate Contract & Dependencies

- **License**: `LGPL-3.0-only WITH LGPL-3.0-linking-exception`
- **Upstream Dependencies**: `jero-core` (zero-copy `ModelWeights`, `ExecutionContext`, `ArchRecipe`, `K3Error`) and optional `jero-kernels-rvv` (enabled by default via `with-kernels`).
- **Safety & Lints**: `#![deny(unsafe_op_in_unsafe_fn)]`, `clippy::pedantic`, and `clippy::undocumented_unsafe_blocks`.

## Local Verification Gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --lib --tests
cargo test --no-default-features --lib --tests
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --lib
```
