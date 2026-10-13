//! Native-capacity input, raw-output, and answer parity against published Laya.
#![expect(
    clippy::unwrap_used,
    reason = "Missing required model artifacts or invalid fixtures must fail."
)]
#![expect(
    clippy::print_stdout,
    reason = "Captured output records native-capacity numerical parity."
)]

use burn::tensor::Device;
use burn_laya::{
    Request,
    input::{Batch, InputBuilder, Row},
    model::{MAX_SEQUENCE_LENGTH, Model, infer},
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
    items: Vec<Row>,
    inputs: Batch,
    logits: Vec<Vec<f32>>,
    act_logits: Vec<Vec<f32>>,
    answers: Value,
}

fn check_capacity(device: &Device) {
    let fixture: Fixture = serde_json::from_str(&common::fixture("capacity.json")).unwrap();
    let config = common::config();
    let builder = InputBuilder::load(
        common::model_directory(),
        MAX_SEQUENCE_LENGTH,
        config.head_max_len,
    )
    .unwrap();
    let model = Model::from_file(concat!(env!("OUT_DIR"), "/model/laya.bpk"), device);
    for case in fixture.cases {
        let (rows, batch) = builder.encode(&case.request).unwrap();
        assert_eq!(rows, case.items, "{} rows", case.name);
        assert_eq!(batch, case.inputs, "{} tensors", case.name);
        match case.name.as_str() {
            "capacity" => assert_eq!(batch.input_ids[0].len(), MAX_SEQUENCE_LENGTH),
            "many-questions" => assert!(batch.input_ids.len() > 64),
            "many-options" => assert!(batch.marker_pos[0].len() > 256),
            name => unreachable!("unknown capacity fixture {name}"),
        }
        let (logits, actions) = infer(&model, &batch, device).unwrap();
        let logit_error = common::assert_logits(&logits, &case.logits);
        let action_error = common::assert_logits(&actions, &case.act_logits);
        let answers = burn_laya::output::decode(&config, &case.request, &logits, &actions).unwrap();
        common::assert_answers(&serde_json::to_value(answers).unwrap(), &case.answers);
        println!(
            "{}: native capacity matches Python; logits={logit_error}, actions={action_error}",
            case.name
        );
    }
}

#[test]
fn native_capacity_matches_python() {
    check_capacity(&Device::flex());
}

#[cfg(feature = "vulkan")]
#[test]
fn integrated_vulkan_native_capacity_matches_python() {
    check_capacity(&Device::vulkan(burn::tensor::DeviceKind::IntegratedGpu(0)));
}
