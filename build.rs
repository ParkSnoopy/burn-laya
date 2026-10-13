//! Convert the selected local ONNX checkpoint to Burn source and weights.

use std::io::Write;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = std::io::stdout().lock();
    writeln!(output, "cargo:rerun-if-env-changed=LAYA_MODEL_DIR")?;
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(std::env::var_os("LAYA_MODEL_DIR").unwrap_or_else(|| "models/laya".into()))
        .canonicalize()?;
    let graph = directory.join("laya.onnx");
    writeln!(
        output,
        "cargo:rustc-env=LAYA_MODEL_DIR={}",
        directory.display()
    )?;
    writeln!(output, "cargo:rerun-if-changed={}", graph.display())?;
    writeln!(
        output,
        "cargo:rerun-if-changed={}",
        directory.join("laya.onnx.data").display()
    )?;
    writeln!(output, "cargo:rerun-if-changed=build.rs")?;
    let encoder = directory.join("encoder/config.json");
    writeln!(output, "cargo:rerun-if-changed={}", encoder.display())?;
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(encoder)?)?;
    let capacity = config["max_position_embeddings"]
        .as_u64()
        .filter(|value| *value > 0)
        .ok_or("encoder needs a positive max_position_embeddings")?;
    let vocabulary = config["vocab_size"]
        .as_u64()
        .filter(|value| *value > 0)
        .ok_or("encoder needs a positive vocab_size")?;
    writeln!(
        std::fs::File::create(
            std::path::Path::new(&std::env::var("OUT_DIR")?).join("model_limits.rs")
        )?,
        "/// Native sequence capacity of the exported encoder.\npub const MAX_SEQUENCE_LENGTH: usize = {capacity};\n/// Vocabulary size of the exported encoder.\npub const VOCABULARY_SIZE: i64 = {vocabulary};"
    )?;
    if !graph.is_file() {
        return Err(
            "Missing laya.onnx. Prepare the verified checkpoint with tools/reference.py --export."
                .into(),
        );
    }
    ModelGen::new()
        .register_op_override(TiledAttention)
        .input(graph.to_str().ok_or("Model path must be UTF-8")?)
        .out_dir("model")
        .load_strategy(LoadStrategy::File)
        .run_from_script();
    Ok(())
}
