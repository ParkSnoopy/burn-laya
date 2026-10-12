use std::path::Path;

use laya_rs::input::{
    Batch,
    InputBuilder,
    Request,
    Row,
};
use serde::Deserialize;

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
    let fixture: Fixture = serde_json::from_str(include_str!("fixtures/reference.json")).unwrap();
    let builder = InputBuilder::load(
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/models/laya")),
        512,
        192,
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
    let builder = InputBuilder::load(
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/models/laya")),
        512,
        192,
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
        r#"{"state":{},"questions":{}}"#,
        r#"{"state":"text","questions":{"x":{"type":"unknown","instructions":"Pick"}}}"#,
        r#"{"state":"text","questions":{"x":{"type":"noul","instructions":"True?","option_order":[1,0]}}}"#,
    ] {
        assert!(serde_json::from_str::<Request>(value).is_err(), "{value}");
    }
    assert!(InputBuilder::load(Path::new("models/laya"), 8192, 192).is_err());
    assert!(InputBuilder::load(Path::new("models/laya"), 512, 513).is_err());
}
