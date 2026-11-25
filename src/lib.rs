//! `jero-arch-encoder`: Zero-KV bidirectional encoder, dense embedding, and cross-encoder
//! reranker recipes (`BERT`, `ModernBERT`, `Nomic`, `BGE`, `Jina-BERT`, `DeBERTa-v3`) for the
//! `SpacemiT` K3 platform.
//!
//! Enforces strict ADR-045 fail-closed weight hydration across both `GGUF` and `SafeTensors`
//! checkpoints without persistent KV cache allocation.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod deberta;
mod kernels_compat;
pub mod modernbert;
pub mod rerank;
mod weights;

use jero_core::{ArchRecipe, ModelMetadata};

pub use deberta::{DebertaRelPosBuffers, DebertaV3Recipe};
pub use modernbert::{EncoderFfnActivation, ModernBertRecipe};
pub use rerank::{CrossEncoderRerankRecipe, RerankScoringMode};

/// Canonical crate identifier for this architecture recipe package.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// Canonical SPDX license identifier for this architecture recipe package.
pub const CRATE_LICENSE: &str = "LGPL-3.0-only WITH LGPL-3.0-linking-exception";

/// Returns the crate description configured in `Cargo.toml`.
#[must_use]
pub const fn crate_description() -> &'static str {
    env!("CARGO_PKG_DESCRIPTION")
}

/// Resolves a boxed [`ArchRecipe`] for any supported zero-KV encoder, dense embedding, or
/// cross-encoder reranking architecture identifier.
#[must_use]
pub fn resolve_encoder_recipe(
    arch: &str,
    metadata: Option<&ModelMetadata>,
) -> Option<Box<dyn ArchRecipe>> {
    match arch {
        "deberta-v3" => Some(Box::new(DebertaV3Recipe::from_metadata(metadata))),
        "cross-encoder" | "reranker" | "bge-reranker" | "jina-reranker" | "modernbert-reranker" => {
            CrossEncoderRerankRecipe::from_metadata(arch, metadata)
                .map(|r| Box::new(r) as Box<dyn ArchRecipe>)
        }
        _ => ModernBertRecipe::from_metadata(arch, metadata)
            .map(|r| Box::new(r) as Box<dyn ArchRecipe>),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_metadata_matches_policy() {
        assert_eq!(CRATE_NAME, "jero-arch-encoder");
        assert_eq!(
            CRATE_LICENSE,
            "LGPL-3.0-only WITH LGPL-3.0-linking-exception"
        );
        assert_ne!(crate_description(), "");
        assert_eq!(jero_core::DEFAULT_ALIGNMENT, 32);
    }

    #[test]
    fn resolve_encoder_recipe_covers_all_architectures() {
        for arch in [
            "modernbert",
            "bert",
            "nomic-bert",
            "bge",
            "jina-bert",
            "deberta-v3",
            "cross-encoder",
            "bge-reranker",
            "jina-reranker",
            "modernbert-reranker",
        ] {
            let recipe = resolve_encoder_recipe(arch, None).expect("supported encoder arch");
            assert_eq!(recipe.architecture_id(), arch);
            assert!(!recipe.requires_kv_cache());
        }
        assert!(resolve_encoder_recipe("llama", None).is_none());
    }
}
