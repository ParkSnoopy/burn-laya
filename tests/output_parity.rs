//! Typed-answer parity and invalid-output boundaries.
#![expect(
    clippy::unwrap_used,
    reason = "Invalid fixed fixtures must fail the test immediately."
)]

use burn_laya::{
    input::Request,
    output::{Config, decode},
};
use serde::Deserialize;
use serde_json::{Value, json};
mod common;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    request: Request,
    logits: Vec<Vec<f32>>,
    act_logits: Vec<Vec<f32>>,
    answers: Value,
}

fn config() -> Config {
    common::config()
}

#[test]
fn typed_answers_match_python() {
    let fixture: Fixture = serde_json::from_str(&common::reference()).unwrap();
    for case in fixture.cases {
        let answers = decode(&config(), &case.request, &case.logits, &case.act_logits).unwrap();
        assert_eq!(
            serde_json::to_value(answers).unwrap(),
            case.answers,
            "{}",
            case.name
        );
    }
}

#[test]
fn ties_temperature_clamps_and_invalid_outputs() {
    let request: Request = serde_json::from_value(json!({"state":"", "questions":{"x":{"type":"choice","instructions":"Pick", "criteria":{"first":null,"second":null}}}})).unwrap();
    let mut config = config();
    let logits = [vec![1.0, 1.0]];
    let actions = [vec![0.0, 0.0]];
    let answer =
        serde_json::to_value(decode(&config, &request, &logits, &actions).unwrap()).unwrap();
    assert_eq!(answer["x"]["choice"], "first");
    assert_eq!(answer["x"]["confidence"], 0.0);
    assert_eq!(answer["x"]["answer_confidence"], 0.5);
    for (raw, applied) in [
        (json!(0.1), 0.5),
        (json!(8.0), 5.0),
        (json!(true), 1.0),
        (Value::Null, 1.0),
    ] {
        config.temperature_by_options.insert("choice:2".into(), raw);
        let actual =
            serde_json::to_value(decode(&config, &request, &[vec![0.0, 1.0]], &actions).unwrap())
                .unwrap();
        config
            .temperature_by_options
            .insert("choice:2".into(), json!(applied));
        let expected =
            serde_json::to_value(decode(&config, &request, &[vec![0.0, 1.0]], &actions).unwrap())
                .unwrap();
        assert_eq!(actual, expected);
    }
    assert!(decode(&config, &request, &[vec![f32::NAN, 0.0]], &actions).is_err());
    assert!(decode(&config, &request, &logits, &[vec![f32::INFINITY, 0.0]]).is_err());
    assert!(decode(&config, &request, &[], &[]).is_err());
}
