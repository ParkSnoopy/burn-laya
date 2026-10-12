use std::path::Path;

use burn::tensor::Device;
use laya_rs::{
    Agent,
    Request,
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
    items: Vec<laya_rs::input::Row>,
}

#[test]
fn real_requests_match_python() {
    let directory = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/models/laya"));
    let agent = Agent::load(directory, &Device::flex()).unwrap();
    let fixture: Fixture = serde_json::from_str(include_str!("fixtures/reference.json")).unwrap();
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
