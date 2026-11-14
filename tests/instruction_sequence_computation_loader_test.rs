use instmodel_inference::{InstructionModel, InstructionModelInfo};
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
fn json_model_predicts_every_instruction_type() {
    let info: InstructionModelInfo = serde_json::from_str(JSON_MODEL_CONFIG).unwrap();
    let model = InstructionModel::new(info).unwrap();
    // MUL scales the input to [-0.5 * x0, 0.6 * x1]; COPY_MASKED writes it reversed into the
    // output, then MAP_TRANSFORM overwrites output[1] with map[round(input[0])] or the default.
    assert_eq!(model.predict(&[1.0, -1.0]).unwrap(), vec![-0.6, -5.0, 0.0]);
    assert_eq!(model.predict(&[-464.0, 1.0]).unwrap(), vec![0.6, 0.5, 0.0]);
    // Upstream branch: sigmoid(DOT(relu(DOT(x)))) + 0.5 lands in buffer 3.
    let mut buffer = vec![0.0; model.required_memory()];
    buffer[..2].copy_from_slice(&[1.0, -1.0]);
    model.predict_with_buffer(&mut buffer).unwrap();
    assert!((buffer[6] - (1.0 / (1.0 + (-2.875f32).exp()) + 0.5)).abs() < 1e-6);
}

#[test]
fn logistic_regression_model() {
    let decision_function = HashMap::from([
        ("feature1".to_string(), 0.5),
        ("feature2".to_string(), -0.3),
        ("constant".to_string(), 1.2),
    ]);
    let info =
        InstructionModelInfo::from_logistic_regression_model(decision_function, None).unwrap();
    assert_eq!(
        info.features,
        Some(vec!["feature1".to_string(), "feature2".to_string()])
    );
    let model = InstructionModel::new(info).unwrap();
    let expected = 1.0 / (1.0 + (-(0.5f32 * 2.0 - 0.3 * 1.0 + 1.2)).exp());
    assert!((model.predict_single(&[2.0, 1.0]).unwrap() - expected).abs() < 1e-6);

    let reordered = HashMap::from([("a".to_string(), 1.0), ("b".to_string(), 2.0)]);
    let info = InstructionModelInfo::from_logistic_regression_model(
        reordered.clone(),
        Some(vec!["b".to_string(), "a".to_string()]),
    )
    .unwrap();
    assert_eq!(info.weights, vec![vec![vec![2.0, 1.0]]]);
    assert!(
        InstructionModelInfo::from_logistic_regression_model(
            reordered,
            Some(vec!["a".to_string()])
        )
        .is_err()
    );
}
