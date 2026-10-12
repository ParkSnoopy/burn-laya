// Each integration test uses a different subset of these checks.
#![allow(dead_code)]

use serde_json::Value;

pub fn assert_logits(actual: &[Vec<f32>], expected: &[Vec<f32>]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f32;
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            let difference = (actual - expected).abs();
            maximum = maximum.max(difference);
            assert!(
                actual.is_finite() && difference <= 1e-4 + 1e-4 * expected.abs(),
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
