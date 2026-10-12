// Each integration test uses a different subset of these checks.
#![expect(
    dead_code,
    reason = "Each integration target uses a different subset of shared checks."
)]

use serde_json::Value;

pub fn model_directory() -> &'static std::path::Path {
    std::path::Path::new(env!("LAYA_MODEL_DIR"))
}

pub fn reference() -> String {
    let original = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/models/laya"))
        .canonicalize()
        .unwrap();
    let path = if model_directory() == original {
        std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/reference.json"
        ))
    } else {
        model_directory().join("reference.json")
    };
    std::fs::read_to_string(path).unwrap()
}

pub fn config() -> laya_rs::output::Config {
    serde_json::from_slice(&std::fs::read(model_directory().join("rl_agent_config.json")).unwrap())
        .unwrap()
}

pub fn assert_logits(actual: &[Vec<f32>], expected: &[Vec<f32>]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f32;
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            let difference = (actual - expected).abs();
            maximum = maximum.max(difference);
            assert!(
                actual.is_finite() && difference <= 1e-4_f32.mul_add(expected.abs(), 1e-4),
                "expected {expected}, got {actual}, difference {difference}"
            );
        }
    }
    maximum
}

pub fn assert_answers(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            let actual = actual.as_f64().unwrap();
            let expected = expected.as_f64().unwrap();
            // FP32 logit drift can cross a four-decimal rounding boundary.
            assert!(
                (actual - expected).abs() <= 1e-4 + 1e-12,
                "expected {expected}, got {actual}"
            );
        }
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(actual.len(), expected.len());
            for (key, expected) in expected {
                assert_answers(actual.get(key).unwrap(), expected);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}
