#![allow(dead_code)]

pub mod gpu;

use instmodel_inference::{InstructionModel, InstructionModelInfo};
use serde_json::{Value, json};

/// Parses a model from JSON; `weights`/`bias` default to empty and `feature_size` to the first buffer.
pub fn info(mut model: Value) -> InstructionModelInfo {
    let object = model.as_object_mut().expect("model JSON must be an object");
    object.entry("weights").or_insert_with(|| json!([]));
    object.entry("bias").or_insert_with(|| json!([]));
    if !object.contains_key("features") && !object.contains_key("feature_size") {
        let first = object["buffer_sizes"][0].clone();
        object.insert("feature_size".to_string(), first);
    }
    serde_json::from_value(model).expect("invalid model JSON")
}

pub fn model(json: Value) -> InstructionModel {
    InstructionModel::new(info(json)).expect("model creation failed")
}

pub fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < tolerance,
            "{name}: index {i}: got {a}, expected {e}"
        );
    }
}
