//! Integrated Vulkan inference parity and synchronized timing.
#![cfg(feature = "vulkan")]
#![expect(
    clippy::unwrap_used,
    reason = "Invalid required artifacts or fixtures must fail the test."
)]
#![expect(
    clippy::print_stdout,
    reason = "Captured output records measured parity and timings."
)]

use std::time::Instant;

use burn::tensor::{Device, DeviceKind};
use burn_laya::{
    Request,
    input::InputBuilder,
    model::{Model, infer},
    output::{Config, decode},
};
use serde::Deserialize;
use serde_json::Value;

mod common;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    request: Request,
    answers: Value,
    logits: Vec<Vec<f32>>,
    act_logits: Vec<Vec<f32>>,
}

#[test]
fn integrated_vulkan_matches_python() {
    let device = Device::vulkan(DeviceKind::IntegratedGpu(0));
    let directory = common::model_directory();
    let start = Instant::now();
    let model = Model::from_file(directory.join("laya.bpk"), &device);
    let config: Config =
        serde_json::from_slice(&std::fs::read(directory.join("rl_agent_config.json")).unwrap())
            .unwrap();
    let inputs = InputBuilder::load(
        directory,
        burn_laya::model::MAX_SEQUENCE_LENGTH,
        config.head_max_len,
    )
    .unwrap();
    println!("Vulkan load: {:?}", start.elapsed());
    let fixture: Fixture = serde_json::from_str(&common::reference()).unwrap();
    for case in fixture.cases {
        let start = Instant::now();
        let (_, batch) = inputs.encode(&case.request).unwrap();
        let encode_time = start.elapsed();
        let start = Instant::now();
        let (logits, actions) = infer(&model, &batch, &device).unwrap();
        let forward_time = start.elapsed();
        let logit_error = common::assert_logits(&logits, &case.logits);
        let action_error = common::assert_logits(&actions, &case.act_logits);
        let start = Instant::now();
        let answers = decode(&config, &case.request, &logits, &actions).unwrap();
        let decode_time = start.elapsed();
        common::assert_answers(&serde_json::to_value(answers).unwrap(), &case.answers);
        println!(
            "Vulkan {}: encode={encode_time:?}, forward={forward_time:?}, decode={decode_time:?}, max logits={logit_error}, max actions={action_error}",
            case.name,
        );
        if case.name == "mixed" {
            let start = Instant::now();
            let (logits, actions) = infer(&model, &batch, &device).unwrap();
            let forward_time = start.elapsed();
            let start = Instant::now();
            let warm = decode(&config, &case.request, &logits, &actions).unwrap();
            let decode_time = start.elapsed();
            common::assert_answers(&serde_json::to_value(warm).unwrap(), &case.answers);
            println!("Vulkan mixed warm: forward={forward_time:?}, decode={decode_time:?}");
        }
    }
}
