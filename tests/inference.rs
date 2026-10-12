//! Complete checkpoint inference parity.
#![expect(
    clippy::unwrap_used,
    reason = "Invalid required artifacts or fixtures must fail the test."
)]
#![expect(
    clippy::print_stdout,
    reason = "Captured parity output identifies each exercised case."
)]

use std::path::Path;

use burn::tensor::Device;
use laya::{Agent, Request};
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
    items: Vec<laya::input::Row>,
}

#[test]
fn real_requests_match_python() {
    let directory = common::model_directory();
    let agent = Agent::load(directory, &Device::flex()).unwrap();
    let fixture: Fixture = serde_json::from_str(&common::reference()).unwrap();
    for case in fixture.cases {
        let prediction = agent.predict(&case.request).unwrap();
        common::assert_answers(
            &serde_json::to_value(prediction.answers).unwrap(),
            &case.answers,
        );
        let dropped = case
            .items
            .iter()
            .map(|row| row.state_stats.state_tokens_dropped)
            .max()
            .unwrap();
        assert_eq!(
            prediction.usage.input_tokens,
            case.items.iter().map(|row| row.ids.len()).sum::<usize>()
        );
        assert_eq!(prediction.usage.state_tokens_dropped, dropped);
        assert_eq!(prediction.usage.truncated, dropped > 0);
        println!("{}: complete inference matches Python", case.name);
    }
    let empty: Request = serde_json::from_str(r#"{"state":"","questions":{}}"#).unwrap();
    assert!(agent.predict(&empty).is_err());
    assert!(Agent::load(Path::new("models/missing"), &Device::flex()).is_err());
}
