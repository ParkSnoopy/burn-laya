//! Explicit ONNX conversion for the supported inference architecture.

use anyhow::Context;
use burn_onnx::{
    LoadStrategy,
    ModelGen,
    ext::{
        ArgType,
        CodegenContext,
        DType,
        Node,
        NodeType,
        OpOverride,
        ProcessError,
        arg_to_ident,
        proc_macro2::TokenStream,
        quote::quote,
    },
};

#[derive(Debug, Clone, Copy)]
struct TiledAttention;

impl OpOverride for TiledAttention {
    fn target(&self) -> NodeType {
        NodeType::Attention
    }

    fn forward(
        &self,
        node: &Node,
        context: &mut CodegenContext<'_, '_>,
    ) -> Result<TokenStream, ProcessError> {
        let Node::Attention(node) = node else {
            return Err(ProcessError::Custom(
                "expected an Attention node".to_owned(),
            ));
        };
        if !(3..=4).contains(&node.inputs.len())
            || node.outputs.len() != 1
            || node.config.is_causal
            || node.config.softcap != 0.0
            || node.config.softmax_precision.is_some()
            || node.inputs.iter().take(3).any(|argument| !matches!(&argument.ty, ArgType::Tensor(tensor) if tensor.rank == 4 && tensor.dtype == DType::F32))
        {
            return Err(ProcessError::Custom("this graph requires rank-four FP32 self-attention without caches or causal/softcap modes".to_owned()));
        }
        let query = context.arg(&node.inputs[0]);
        let key = context.arg(&node.inputs[1]);
        let value = context.arg(&node.inputs[2]);
        let output = arg_to_ident(&node.outputs[0]);
        let bias = if let Some(argument) = node.inputs.get(3) {
            let ArgType::Tensor(tensor) = &argument.ty else {
                return Err(ProcessError::Custom(
                    "attention mask must be a tensor".to_owned(),
                ));
            };
            if tensor.rank != 4 {
                return Err(ProcessError::Custom(
                    "attention mask must have rank four".to_owned(),
                ));
            }
            let bias = context.arg(argument);
            match tensor.dtype {
                DType::F32 => quote!(Some(#bias)),
                DType::Bool(_) => {
                    quote!({
                        let allowed = #bias;
                        Some(allowed.clone().float().mul_scalar(0.0).mask_fill(allowed.bool_not(), f32::NEG_INFINITY))
                    })
                }
                _ => {
                    return Err(ProcessError::Custom(
                        "unsupported attention mask dtype".to_owned(),
                    ));
                }
            }
        } else {
            quote!(None)
        };
        let scale = node
            .config
            .scale
            .map_or_else(|| quote!(None), |scale| quote!(Some(#scale)));
        Ok(quote! {
            let #output = crate::inference::attention::forward(#query, #key, #value, #bias, #scale);
        })
    }
}

/// Convert a supported local ONNX checkpoint to complete Burnpack weights.
///
/// # Errors
/// Returns an error for missing assets or a graph incompatible with the bundled model.
pub fn generate(directory: &std::path::Path) -> anyhow::Result<()> {
    let graph = directory.join("laya.onnx");
    let encoder = directory.join("encoder/config.json");
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(encoder)?)?;
    let capacity = config["max_position_embeddings"]
        .as_u64()
        .filter(|value| *value > 0)
        .ok_or_else(|| anyhow::anyhow!("encoder needs a positive max_position_embeddings"))?;
    let vocabulary = config["vocab_size"]
        .as_u64()
        .filter(|value| *value > 0)
        .ok_or_else(|| anyhow::anyhow!("encoder needs a positive vocab_size"))?;
    anyhow::ensure!(
        capacity == crate::model::MAX_SEQUENCE_LENGTH as u64
            && vocabulary == crate::model::VOCABULARY_SIZE as u64,
        "encoder bounds differ from the bundled inference architecture"
    );
    anyhow::ensure!(
        graph.is_file(),
        "Missing laya.onnx. Prepare the verified checkpoint with tools/reference.py --export."
    );
    ModelGen::new()
        .register_op_override(TiledAttention)
        .input(
            graph
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Model path must be UTF-8"))?,
        )
        .out_dir(
            directory
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Model path must be UTF-8"))?,
        )
        .load_strategy(LoadStrategy::File)
        .run_from_cli();
    let source_path = directory.join("laya.rs");
    let source = std::fs::read_to_string(&source_path)?;
    let weights = directory.join("laya.bpk");
    let weight_literal = burn_onnx::ext::proc_macro2::Literal::string(
        weights.to_str().context("Model path must be UTF-8")?,
    );
    let source = source
        .replacen(
            source.lines().next().context("empty generated source")?,
            "// Generated from the verified English Laya ONNX graph by burn-onnx 0.22.",
            1,
        )
        .replace(&weight_literal.to_string(), "\"laya.bpk\"");
    anyhow::ensure!(
        source == include_str!("generated/laya.rs"),
        "exported graph differs from the bundled inference architecture; regenerate and review the bundled source before deployment"
    );
    std::fs::write(source_path, source)?;
    Ok(())
}
