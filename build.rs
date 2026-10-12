use burn_onnx::{
    LoadStrategy,
    ModelGen,
};

fn main() {
    println!("cargo:rerun-if-changed=models/laya/laya.onnx");
    println!("cargo:rerun-if-changed=models/laya/laya.onnx.data");
    println!("cargo:rerun-if-changed=build.rs");
    assert!(
        std::path::Path::new("models/laya/laya.onnx").is_file(),
        "Missing models/laya/laya.onnx. Prepare the verified checkpoint with tests/reference.py --export."
    );
    ModelGen::new()
        .input("models/laya/laya.onnx")
        .out_dir("model")
        .load_strategy(LoadStrategy::File)
        .run_from_script();
}
