//! Tokenization parity and invalid-request boundaries.
#![expect(
    clippy::unwrap_used,
    reason = "Invalid fixed fixtures must fail the test immediately."
)]

use std::path::Path;

use burn_laya::input::{Batch, InputBuilder, Request, Row};
use serde::Deserialize;
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
}

#[test]
fn input_tensors_match_python() {
    let fixture: Fixture = serde_json::from_str(&common::reference()).unwrap();
    let config = common::config();
    let builder = InputBuilder::load(
        common::model_directory(),
        burn_laya::model::MAX_SEQUENCE_LENGTH,
        config.head_max_len,
    )
    .unwrap();
    for case in fixture.cases {
        let (rows, batch) = builder.encode(&case.request).unwrap();
        assert_eq!(rows, case.items, "{} rows", case.name);
        assert_eq!(batch, case.inputs, "{} tensors", case.name);
    }
}

#[test]
fn invalid_requests_fail_before_inference() {
    let config = common::config();
    let builder = InputBuilder::load(
        common::model_directory(),
        burn_laya::model::MAX_SEQUENCE_LENGTH,
        config.head_max_len,
    )
    .unwrap();
    for value in [
        r#"{"state":"text","questions":{}}"#,
        r#"{"state":"text","questions":{"x":{"type":"choice","instructions":"Pick","criteria":{}}}}"#,
        r#"{"state":"text","questions":{"x":{"type":"score","instructions":" ","criteria":["a"]}}}"#,
        r#"{"state":"text","questions":{"x":{"type":"noul","instructions":"True?","criteria":{"other":"bad"}}}}"#,
    ] {
        let request: Request = serde_json::from_str(value).unwrap();
        assert!(builder.encode(&request).is_err(), "{value}");
    }
    for value in [
        r#"{"state":null,"questions":{}}"#,
        r#"{"state":{},"questions":{}}"#,
        r#"{"state":"text","questions":{"x":{"type":"unknown","instructions":"Pick"}}}"#,
        r#"{"state":"text","questions":{"x":{"type":"noul","instructions":"True?","option_order":[1,0]}}}"#,
    ] {
        assert!(serde_json::from_str::<Request>(value).is_err(), "{value}");
    }
    assert!(
        InputBuilder::load(
            common::model_directory(),
            burn_laya::model::MAX_SEQUENCE_LENGTH,
            192
        )
        .is_ok()
    );
    assert!(
        InputBuilder::load(
            Path::new("models/laya"),
            burn_laya::model::MAX_SEQUENCE_LENGTH + 1,
            192
        )
        .is_err()
    );
    assert!(InputBuilder::load(Path::new("models/laya"), 512, 513).is_err());
}
