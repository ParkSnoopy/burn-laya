use burn::tensor::Device;
use laya_rs::{
    input::Batch,
    model::{
        Model,
        infer,
    },
};
use serde::Deserialize;
mod common;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    inputs: Batch,
    logits: Vec<Vec<f32>>,
    act_logits: Vec<Vec<f32>>,
}


#[test]
fn full_checkpoint_matches_pytorch() {
    let fixture: Fixture = serde_json::from_str(include_str!("fixtures/reference.json")).unwrap();
    let device = Device::flex();
    let model = Model::from_file(concat!(env!("OUT_DIR"), "/model/laya.bpk"), &device);
    for case in fixture.cases {
        let (logits, act_logits) = infer(&model, &case.inputs, &device).unwrap();
        for (name, actual, expected) in [
            ("logits", logits, case.logits),
            ("act_logits", act_logits, case.act_logits),
        ] {
            let maximum = common::assert_logits(&actual, &expected);
            println!(
                "{}/{name}: maximum absolute difference {maximum}",
                case.name
            );
        }
    }
}
