//! Tests for instruction sequence computation loading from JSON.

use instmodel_rust_inference::{InstructionModel, InstructionModelInfo};
use serde_json;
use std::collections::HashMap;

const JSON_MODEL_CONFIG: &str = r#"
{
  "features": ["feature1", "feature2"],
  "buffer_sizes": [2, 2, 2, 1, 3],
  "instructions": [
    {
      "input": 0,
      "output": 1,
      "type": "DOT",
      "weights": 0,
      "activation": "RELU"
    },
    {
      "input": 1,
      "output": 3,
      "type": "DOT",
      "weights": 1
    },
    {
      "input": 3,
      "type": "ACTIVATION",
      "activation": "SIGMOID"
    },
    {
      "input": 1,
      "type": "COPY",
      "internal_index": 0,
      "output": 2
    },
    {
      "input": 3,
      "type": "ADD_ELEMENTWISE",
      "parameters": 0
    },
    {
      "input": 0,
      "type": "MUL_ELEMENTWISE",
      "parameters": 1
    },
    {
      "input": 0,
      "type": "COPY_MASKED",
      "output": 4,
      "indexes": [1, 0]
    },
    {
      "input": 0,
      "type": "MAP_TRANSFORM",
      "output": 4,
      "size": 1,
      "internal_input_index": 0,
      "internal_output_index": 1,
      "map": 0,
      "default": [-5.0]
    }
  ],
  "weights": [
    [
      [2, 0.5],
      [-2, -0.5]
    ],
    [
      [0.5, -1]
    ]
  ],
  "bias": [
    [0.25, -0.25],
    [2]
  ],
  "parameters": [[0.5], [-0.5, 0.6]],
  "maps": [{"232": [0.5], "233": [0.6]}]
}
"#;

#[test]
fn create_instruction_model_from_json() {
    let instruction_model_info: InstructionModelInfo =
        serde_json::from_str(JSON_MODEL_CONFIG).expect("Failed to parse JSON");

    // Verify basic structure
    assert_eq!(instruction_model_info.features.as_ref().unwrap().len(), 2);
    assert_eq!(instruction_model_info.computation_buffer_sizes.len(), 5);
    assert_eq!(instruction_model_info.instructions.len(), 8);
    assert!(instruction_model_info.weights.len() > 0);
    assert!(instruction_model_info.bias.len() > 0);
    assert_eq!(instruction_model_info.weights.len(), 2);
    assert_eq!(instruction_model_info.bias.len(), 2);
    assert!(instruction_model_info.parameters.is_some());
    assert_eq!(instruction_model_info.parameters.as_ref().unwrap().len(), 2);
    assert!(instruction_model_info.maps.is_some());
    assert_eq!(instruction_model_info.maps.as_ref().unwrap().len(), 1);

    // Verify weights structure
    assert_eq!(instruction_model_info.weights[0].len(), 2);
    assert_eq!(instruction_model_info.weights[1].len(), 1);
    assert_eq!(instruction_model_info.bias[0].len(), 2);
    assert_eq!(instruction_model_info.bias[1].len(), 1);
    assert_eq!(
        instruction_model_info.parameters.as_ref().unwrap()[0].len(),
        1
    );
    assert_eq!(
        instruction_model_info.parameters.as_ref().unwrap()[1].len(),
        2
    );
    assert_eq!(instruction_model_info.maps.as_ref().unwrap()[0].len(), 2);
}

#[test]
fn test_basic_model_creation() {
    let instruction_model_info: InstructionModelInfo =
        serde_json::from_str(JSON_MODEL_CONFIG).expect("Failed to parse JSON");

    // This should not panic and should create a valid model
    let model = InstructionModel::new(instruction_model_info);
    assert!(model.is_ok(), "Model creation should succeed");
}

#[test]
fn test_logistic_regression_model() {
    let mut decision_function = HashMap::new();
    decision_function.insert("feature1".to_string(), 0.5);
    decision_function.insert("feature2".to_string(), -0.3);
    decision_function.insert("constant".to_string(), 1.2);

    let model_info = InstructionModelInfo::from_logistic_regression_model(decision_function, None);

    assert!(model_info.is_ok());
    let model_info = model_info.unwrap();

    assert_eq!(model_info.features.as_ref().unwrap().len(), 2);
    assert_eq!(model_info.computation_buffer_sizes, vec![2, 1]);
    assert_eq!(model_info.weights.len(), 1);
    assert_eq!(model_info.bias.len(), 1);
    assert_eq!(model_info.bias[0][0], 1.2);

    // Create the actual model
    let model = InstructionModel::new(model_info);
    assert!(model.is_ok());
}
