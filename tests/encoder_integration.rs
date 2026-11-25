//! Integration tests for `jero-arch-encoder` covering ADR-045 fail-closed hydration,
//! `F16`/`BF16`/`Q8_0` multi-quant support, `SafeTensors` fused `Wqkv`/`Wi` layouts,
//! dense embeddings with Matryoshka truncation, and cross-encoder reranking.

use jero_arch_encoder::{
    CrossEncoderRerankRecipe, DebertaV3Recipe, ModernBertRecipe, RerankScoringMode,
    resolve_encoder_recipe,
};
use jero_core::safetensors::test_helpers::create_synthetic_safetensors;
use jero_core::test_helpers::create_synthetic_gguf_v3;
use jero_core::{
    ArchRecipe, EmbeddingPooling, ExecutionContext, GgufFile, K3Error, QuantType, SafeTensorsFile,
};

fn encode_f16_from_f32(val: f32) -> [u8; 2] {
    let bits = val.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).unwrap_or(0);
    let exp = i32::try_from((bits >> 23) & 0xFF).unwrap_or(0);
    let mant = bits & 0x007F_FFFF;
    if exp == 0 {
        return sign.to_le_bytes();
    }
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 {
        return sign.to_le_bytes();
    }
    if new_exp >= 31 {
        return (sign | 0x7C00).to_le_bytes();
    }
    let exp_bits = u16::try_from(new_exp).unwrap_or(0) << 10;
    let mant_bits = u16::try_from(mant >> 13).unwrap_or(0);
    (sign | exp_bits | mant_bits).to_le_bytes()
}

fn deterministic_f16_bytes(len: usize, seed: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 2);
    for idx in 0..len {
        #[expect(clippy::cast_precision_loss, reason = "bounded test index fits f32")]
        let val = ((((idx * 5 + seed * 3 + 1) % 17) as f32) - 8.0) * 0.0625;
        out.extend_from_slice(&encode_f16_from_f32(val));
    }
    out
}

fn norm_ones_f16_bytes(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 2);
    for _ in 0..len {
        out.extend_from_slice(&encode_f16_from_f32(1.0));
    }
    out
}

fn deterministic_f32_bytes_vec(len: usize, seed: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 4);
    for idx in 0..len {
        #[expect(clippy::cast_precision_loss, reason = "bounded test index fits f32")]
        let val = ((((idx * 7 + seed * 3 + 5) % 23) as f32) - 11.0) * 0.05;
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

fn norm_ones_f32_bytes_vec(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 4);
    for _ in 0..len {
        out.extend_from_slice(&1.0_f32.to_le_bytes());
    }
    out
}

#[test]
fn integration_safetensors_fused_wqkv_and_wi_modernbert_embedding() {
    let hidden_dim = 8_usize;
    let ffn_dim = 16_usize;
    let vocab_size = 16_usize;

    let emb_bytes = deterministic_f16_bytes(vocab_size * hidden_dim, 1);
    let norm1_bytes = norm_ones_f16_bytes(hidden_dim);
    let wqkv_bytes = deterministic_f16_bytes(3 * hidden_dim * hidden_dim, 2);
    let attn_out_bytes = deterministic_f16_bytes(hidden_dim * hidden_dim, 3);
    let norm2_bytes = norm_ones_f16_bytes(hidden_dim);
    let fused_mlp_bytes = deterministic_f16_bytes(2 * ffn_dim * hidden_dim, 4);
    let down_mlp_bytes = deterministic_f16_bytes(hidden_dim * ffn_dim, 5);

    let st_bytes = create_synthetic_safetensors(
        &[
            ("general.architecture", "modernbert"),
            ("num_hidden_layers", "1"),
            ("hidden_size", "8"),
            ("num_attention_heads", "2"),
            ("rope_theta", "10000.0"),
        ],
        &[
            (
                "model.embeddings.tok_embeddings.weight",
                QuantType::F16,
                &[vocab_size, hidden_dim],
                &emb_bytes,
            ),
            (
                "model.layers.0.attn_norm.weight",
                QuantType::F16,
                &[hidden_dim],
                &norm1_bytes,
            ),
            (
                "model.layers.0.attn.Wqkv.weight",
                QuantType::F16,
                &[3 * hidden_dim, hidden_dim],
                &wqkv_bytes,
            ),
            (
                "model.layers.0.attn.Wo.weight",
                QuantType::F16,
                &[hidden_dim, hidden_dim],
                &attn_out_bytes,
            ),
            (
                "model.layers.0.ffn_norm.weight",
                QuantType::F16,
                &[hidden_dim],
                &norm2_bytes,
            ),
            (
                "model.layers.0.mlp.Wi.weight",
                QuantType::F16,
                &[2 * ffn_dim, hidden_dim],
                &fused_mlp_bytes,
            ),
            (
                "model.layers.0.mlp.Wo.weight",
                QuantType::F16,
                &[hidden_dim, ffn_dim],
                &down_mlp_bytes,
            ),
        ],
    );

    let st = SafeTensorsFile::parse(&st_bytes).expect("parse fused SafeTensors ModernBERT");
    let recipe = ModernBertRecipe::from_metadata("modernbert", Some(st.metadata()))
        .expect("hydrate modernbert");
    assert_eq!(recipe.num_layers, 1);
    assert_eq!(recipe.hidden_dim, 8);
    assert_eq!(recipe.ffn_dim, 16);
    assert_eq!(recipe.n_heads, 2);
    assert_eq!(recipe.head_dim, 4);

    let mut embed = [0.0_f32; 4];
    let mut ctx = ExecutionContext::new(&[1_u32, 4, 7], 32, &mut [], &mut [])
        .expect("ctx")
        .with_embedding_output(&mut embed, EmbeddingPooling::Mean, Some(4))
        .expect("attach matryoshka=4");
    recipe
        .forward_classify(&mut ctx, &st)
        .expect("F16 fused Wqkv/Wi SafeTensors ModernBERT embedding succeeds");
    let norm_sq: f32 = embed.iter().map(|v| v * v).sum();
    assert!((norm_sq - 1.0).abs() < 1e-4);
}

#[test]
fn integration_adr045_corrupted_nan_norm_fails_closed() {
    let mut nan_norm = [0.0_f32; 8];
    nan_norm[3] = f32::NAN;
    let nan_bytes: Vec<u8> = nan_norm.iter().flat_map(|v| v.to_le_bytes()).collect();
    let emb_bytes = [0_u8; 16 * 8 * 4];
    let w_bytes = [0_u8; 8 * 8 * 4];
    let ffn_bytes = [0_u8; 16 * 8 * 4];
    let zero_norm = [0_u8; 32];
    let zero_cls = [0_u8; 64];

    let st_nan = create_synthetic_safetensors(
        &[
            ("general.architecture", "bert"),
            ("num_hidden_layers", "1"),
            ("hidden_size", "8"),
            ("num_attention_heads", "2"),
        ],
        &[
            (
                "embeddings.word_embeddings.weight",
                QuantType::F32,
                &[16, 8],
                &emb_bytes,
            ),
            (
                "encoder.layer.0.attention.output.LayerNorm.weight",
                QuantType::F32,
                &[8],
                &nan_bytes,
            ),
            (
                "encoder.layer.0.attention.self.query.weight",
                QuantType::F32,
                &[8, 8],
                &w_bytes,
            ),
            (
                "encoder.layer.0.attention.self.key.weight",
                QuantType::F32,
                &[8, 8],
                &w_bytes,
            ),
            (
                "encoder.layer.0.attention.self.value.weight",
                QuantType::F32,
                &[8, 8],
                &w_bytes,
            ),
            (
                "encoder.layer.0.attention.output.dense.weight",
                QuantType::F32,
                &[8, 8],
                &w_bytes,
            ),
            (
                "encoder.layer.0.output.LayerNorm.weight",
                QuantType::F32,
                &[8],
                &zero_norm,
            ),
            (
                "encoder.layer.0.intermediate.dense.weight",
                QuantType::F32,
                &[16, 8],
                &ffn_bytes,
            ),
            (
                "encoder.layer.0.output.dense.weight",
                QuantType::F32,
                &[8, 16],
                &ffn_bytes,
            ),
            ("classifier.weight", QuantType::F32, &[2, 8], &zero_cls),
        ],
    );
    let st = SafeTensorsFile::parse(&st_nan).expect("parse nan norm safetensors");
    let recipe = ModernBertRecipe::from_metadata("bert", Some(st.metadata())).expect("bert");
    let mut logits = [0.0_f32; 2];
    let mut ctx = ExecutionContext::new(&[1_u32, 2], 32, &mut logits, &mut []).expect("ctx");
    let err = recipe
        .forward_classify(&mut ctx, &st)
        .expect_err("ADR-045: NaN norm weight must fail closed");
    assert!(matches!(err, K3Error::TensorBoundsError { .. }));
}

#[test]
fn integration_adr045_nomic_missing_gate_fails_closed() {
    let emb_bytes = [0_u8; 16 * 8 * 4];
    let w_bytes = [0_u8; 8 * 8 * 4];
    let ffn_bytes = [0_u8; 16 * 8 * 4];
    let zero_norm = [0_u8; 32];

    let st_no_gate = create_synthetic_safetensors(
        &[
            ("general.architecture", "nomic-bert"),
            ("num_hidden_layers", "1"),
            ("hidden_size", "8"),
            ("num_attention_heads", "2"),
        ],
        &[
            (
                "embeddings.word_embeddings.weight",
                QuantType::F32,
                &[16, 8],
                &emb_bytes,
            ),
            (
                "encoder.layer.0.attention.output.LayerNorm.weight",
                QuantType::F32,
                &[8],
                &zero_norm,
            ),
            (
                "encoder.layer.0.attention.self.query.weight",
                QuantType::F32,
                &[8, 8],
                &w_bytes,
            ),
            (
                "encoder.layer.0.attention.self.key.weight",
                QuantType::F32,
                &[8, 8],
                &w_bytes,
            ),
            (
                "encoder.layer.0.attention.self.value.weight",
                QuantType::F32,
                &[8, 8],
                &w_bytes,
            ),
            (
                "encoder.layer.0.attention.output.dense.weight",
                QuantType::F32,
                &[8, 8],
                &w_bytes,
            ),
            (
                "encoder.layer.0.output.LayerNorm.weight",
                QuantType::F32,
                &[8],
                &zero_norm,
            ),
            (
                "encoder.layer.0.intermediate.dense.weight",
                QuantType::F32,
                &[16, 8],
                &ffn_bytes,
            ),
            (
                "encoder.layer.0.output.dense.weight",
                QuantType::F32,
                &[8, 16],
                &ffn_bytes,
            ),
        ],
    );
    let st_ng = SafeTensorsFile::parse(&st_no_gate).expect("parse nomic without gate");
    let nomic =
        ModernBertRecipe::from_metadata("nomic-bert", Some(st_ng.metadata())).expect("nomic");
    let mut embed = [0.0_f32; 8];
    let mut ctx_emb = ExecutionContext::new(&[1_u32, 2], 32, &mut [], &mut [])
        .expect("ctx")
        .with_embedding_output(&mut embed, EmbeddingPooling::Mean, None)
        .expect("emb");
    let gate_err = nomic
        .forward_classify(&mut ctx_emb, &st_ng)
        .expect_err("ADR-045: nomic-bert missing SwiGLU gate projection must fail closed");
    assert!(matches!(gate_err, K3Error::TensorBoundsError { .. }));
}

#[test]
fn integration_deberta_quant_admission() {
    let q8_bytes = create_synthetic_gguf_v3(
        "deberta-v3",
        64,
        &[
            ("token_embd.weight", QuantType::Q8_0, &[32, 4][..]),
            ("classifier.weight", QuantType::Q8_0, &[32, 3][..]),
        ],
    );
    let q8_gguf = GgufFile::parse(&q8_bytes).expect("parse q8 deberta");
    let deberta = DebertaV3Recipe::from_metadata(Some(q8_gguf.metadata()));
    deberta
        .validate_model(q8_gguf.metadata(), &q8_gguf)
        .expect("Q8_0 DeBERTa-v3 passes quant admission");

    let q4_bytes = create_synthetic_gguf_v3(
        "deberta-v3",
        64,
        &[("blk.0.attn_q.weight", QuantType::Q4_K_M, &[32, 8][..])],
    );
    let q4_gguf = GgufFile::parse(&q4_bytes).expect("parse q4 deberta");
    assert!(matches!(
        deberta.validate_model(q4_gguf.metadata(), &q4_gguf),
        Err(K3Error::CapabilityMismatch { .. })
    ));
}

fn build_two_logit_bge_safetensors() -> Vec<u8> {
    let hidden_dim = 8_usize;
    let ffn_dim = 16_usize;
    let emb_bytes = deterministic_f32_bytes_vec(16 * hidden_dim, 1);
    let norm_bytes = norm_ones_f32_bytes_vec(hidden_dim);
    let q_bytes = deterministic_f32_bytes_vec(hidden_dim * hidden_dim, 2);
    let k_bytes = deterministic_f32_bytes_vec(hidden_dim * hidden_dim, 3);
    let v_bytes = deterministic_f32_bytes_vec(hidden_dim * hidden_dim, 4);
    let o_bytes = deterministic_f32_bytes_vec(hidden_dim * hidden_dim, 5);
    let up_bytes = deterministic_f32_bytes_vec(ffn_dim * hidden_dim, 6);
    let down_bytes = deterministic_f32_bytes_vec(hidden_dim * ffn_dim, 7);
    let head_weights = deterministic_f32_bytes_vec(2 * hidden_dim, 8);
    let head_bias = deterministic_f32_bytes_vec(2, 9);

    create_synthetic_safetensors(
        &[
            ("general.architecture", "bge"),
            ("num_hidden_layers", "1"),
            ("hidden_size", "8"),
            ("num_attention_heads", "2"),
        ],
        &[
            (
                "embeddings.word_embeddings.weight",
                QuantType::F32,
                &[16, hidden_dim],
                &emb_bytes,
            ),
            (
                "encoder.layer.0.attention.output.LayerNorm.weight",
                QuantType::F32,
                &[hidden_dim],
                &norm_bytes,
            ),
            (
                "encoder.layer.0.attention.self.query.weight",
                QuantType::F32,
                &[hidden_dim, hidden_dim],
                &q_bytes,
            ),
            (
                "encoder.layer.0.attention.self.key.weight",
                QuantType::F32,
                &[hidden_dim, hidden_dim],
                &k_bytes,
            ),
            (
                "encoder.layer.0.attention.self.value.weight",
                QuantType::F32,
                &[hidden_dim, hidden_dim],
                &v_bytes,
            ),
            (
                "encoder.layer.0.attention.output.dense.weight",
                QuantType::F32,
                &[hidden_dim, hidden_dim],
                &o_bytes,
            ),
            (
                "encoder.layer.0.output.LayerNorm.weight",
                QuantType::F32,
                &[hidden_dim],
                &norm_bytes,
            ),
            (
                "encoder.layer.0.intermediate.dense.weight",
                QuantType::F32,
                &[ffn_dim, hidden_dim],
                &up_bytes,
            ),
            (
                "encoder.layer.0.output.dense.weight",
                QuantType::F32,
                &[hidden_dim, ffn_dim],
                &down_bytes,
            ),
            (
                "classifier.weight",
                QuantType::F32,
                &[2, hidden_dim],
                &head_weights,
            ),
            ("classifier.bias", QuantType::F32, &[2], &head_bias),
        ],
    )
}

#[test]
fn integration_rerank_two_logit_softmax_parity() {
    let st_two_logit = build_two_logit_bge_safetensors();
    let st_rerank = SafeTensorsFile::parse(&st_two_logit).expect("parse 2-logit safetensors");
    let mut bge_cls =
        ModernBertRecipe::from_metadata("bge", Some(st_rerank.metadata())).expect("bge");
    bge_cls.head_dim = 4;
    let pair_tokens = [1_u32, 3, 5, 7];
    let mut raw_logits = [0.0_f32; 2];
    let mut cls_ctx =
        ExecutionContext::new(&pair_tokens, 32, &mut raw_logits, &mut []).expect("cls ctx");
    bge_cls
        .forward_classify(&mut cls_ctx, &st_rerank)
        .expect("bge raw 2-logit forward");
    let expected_softmax_pos = 1.0_f32 / (1.0_f32 + (-(raw_logits[1] - raw_logits[0])).exp());

    let mut reranker =
        CrossEncoderRerankRecipe::new().with_scoring_mode(RerankScoringMode::TwoLogitPositive);
    reranker.encoder.num_layers = 1;
    reranker.encoder.hidden_dim = 8;
    reranker.encoder.ffn_dim = 16;
    reranker.encoder.n_heads = 2;
    reranker.encoder.head_dim = 4;
    let score = reranker
        .rerank_pair(&[1_u32, 3], &[5_u32, 7], 32, &st_rerank)
        .expect("rerank_pair 2-logit");
    assert!(
        (score - expected_softmax_pos).abs() < 1e-5,
        "TwoLogitPositive score ({score}) must match softmax positive probability ({expected_softmax_pos})"
    );

    let cls_recipe = resolve_encoder_recipe("bge", None).expect("bge");
    assert_eq!(cls_recipe.architecture_id(), "bge");
    assert_eq!(reranker.architecture_id(), "cross-encoder");
}
