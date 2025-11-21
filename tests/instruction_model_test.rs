//! Comprehensive tests for the InstructionModel that match the Java implementation.

use instmodel_rust_inference::instruction_model_info::*;
use instmodel_rust_inference::{Activation, InstructionModel, InstructionModelInfo, ValidationData};

const DELTA: f32 = 0.00005;

/// Creates weights for a neural network structure with 2 input features, 2 hidden layer nodes, and 1 output node.
fn create_weights_for_complex_neural_network(
    input_size: usize,
    weights_layer0: &mut [Vec<f32>],
    weights_layer1: &mut [Vec<f32>],
    bias_layer0: &mut [f32],
    bias_layer1: &mut [f32],
) {
    // Initialize weights_layer0
    for weights_row in weights_layer0.iter_mut() {
        weights_row.resize(input_size, 0.0);
    }

    weights_layer0[0][0] = 2.0;
    weights_layer0[0][1] = 0.5;
    bias_layer0[0] = 0.25;

    // Inverse values to make calculations easier to understand
    weights_layer0[1][0] = -2.0;
    weights_layer0[1][1] = -0.5;
    bias_layer0[1] = -0.25;

    // Initialize weights_layer1
    for weights_row in weights_layer1.iter_mut() {
        weights_row.resize(weights_layer0.len(), 0.0);
    }

    weights_layer1[0][0] = 0.5;
    weights_layer1[0][1] = -1.0;
    bias_layer1[0] = 2.0;
}

#[test]
fn external_memory_management() {
    let layer_sizes = vec![2, 1];
    let weights = vec![vec![0.0; layer_sizes[0]]; layer_sizes[1]];
    let bias = vec![0.0; layer_sizes[1]];

    let model_info = InstructionModelInfo {
        features: Some(vec!["feature1".to_string(), "feature2".to_string()]),
        feature_size: None,
        computation_buffer_sizes: layer_sizes.clone(),
        instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
            input: 0,
            output: 1,
            weights: 0,
            activation: None,
        })],
        weights: vec![weights],
        bias: vec![bias],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let model = InstructionModel::new(model_info).expect("Model creation should succeed");

    // Test that we need a computation buffer
    let result = model.predict(&[1.0, -1.0]);
    assert!(
        result.is_ok(),
        "Prediction should work with internal buffer allocation"
    );
}

#[test]
fn one_forward_pass() {
    let layer_sizes = vec![2, 1];
    let mut weights = vec![vec![0.0; layer_sizes[0]]; layer_sizes[1]];
    let mut bias = vec![0.0; layer_sizes[1]];

    weights[0][0] = 2.0;
    weights[0][1] = 0.5;
    bias[0] = 0.25;

    let model_info = InstructionModelInfo {
        features: Some(vec!["feature1".to_string(), "feature2".to_string()]),
        feature_size: None,
        computation_buffer_sizes: layer_sizes.clone(),
        instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
            input: 0,
            output: 1,
            weights: 0,
            activation: None,
        })],
        weights: vec![weights],
        bias: vec![bias],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let model = InstructionModel::new(model_info).expect("Model creation should succeed");

    let inputs = vec![1.0, -1.0];
    let result = model
        .predict_single(&inputs)
        .expect("Prediction should succeed");
    // 1 * 2 - 1 * 0.5 + 0.25 = 1.75
    assert!((result - 1.75).abs() < DELTA);

    let inputs2 = vec![-1.0, 1.0];
    let result2 = model
        .predict_single(&inputs2)
        .expect("Prediction should succeed");
    // -1 * 2 + 1 * 0.5 + 0.25 = -1.25
    assert!((result2 - (-1.25)).abs() < DELTA);
}

#[test]
fn activation_layers() {
    let layer_sizes = vec![3];

    // Test RELU
    let model_info = InstructionModelInfo {
        features: Some(vec!["f1".to_string(), "f2".to_string(), "f3".to_string()]),
        feature_size: None,
        computation_buffer_sizes: layer_sizes.clone(),
        instructions: vec![InstructionInfo::Activation(ActivationInstructionInfo {
            input: 0,
            activation: Activation::Relu,
        })],
        weights: vec![],
        bias: vec![],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let model = InstructionModel::new(model_info).expect("Model creation should succeed");
    let inputs = vec![1.0, -1.0, 0.5];
    let outputs = model.predict(&inputs).expect("Prediction should succeed");
    assert!((outputs[0] - 1.0).abs() < DELTA);
    assert!((outputs[1] - 0.0).abs() < DELTA);
    assert!((outputs[2] - 0.5).abs() < DELTA);

    // Test SIGMOID
    let model_info = InstructionModelInfo {
        features: Some(vec!["f1".to_string(), "f2".to_string(), "f3".to_string()]),
        feature_size: None,
        computation_buffer_sizes: layer_sizes.clone(),
        instructions: vec![InstructionInfo::Activation(ActivationInstructionInfo {
            input: 0,
            activation: Activation::Sigmoid,
        })],
        weights: vec![],
        bias: vec![],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let model = InstructionModel::new(model_info).expect("Model creation should succeed");
    let inputs = vec![1.0, 0.0, -0.5];
    let outputs = model.predict(&inputs).expect("Prediction should succeed");
    assert!((outputs[0] - 0.7311).abs() < DELTA);
    assert!((outputs[1] - 0.5).abs() < DELTA);
    assert!((outputs[2] - 0.3775).abs() < DELTA);

    // Test SOFTMAX
    let model_info = InstructionModelInfo {
        features: Some(vec!["f1".to_string(), "f2".to_string(), "f3".to_string()]),
        feature_size: None,
        computation_buffer_sizes: layer_sizes.clone(),
        instructions: vec![InstructionInfo::Activation(ActivationInstructionInfo {
            input: 0,
            activation: Activation::Softmax,
        })],
        weights: vec![],
        bias: vec![],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let model = InstructionModel::new(model_info).expect("Model creation should succeed");
    let inputs = vec![1.0, 2.0, 3.0];
    let outputs = model.predict(&inputs).expect("Prediction should succeed");
    assert!((outputs[0] - 0.09003057).abs() < DELTA);
    assert!((outputs[1] - 0.24472847).abs() < DELTA);
    assert!((outputs[2] - 0.66524096).abs() < DELTA);
}

#[test]
fn copy_layers() {
    let layer_sizes = vec![2, 2, 6, 6];

    let instructions = vec![
        InstructionInfo::Copy(CopyInstructionInfo {
            input: 0,
            output: 1,
            internal_index: 0,
        }),
        InstructionInfo::Copy(CopyInstructionInfo {
            input: 0,
            output: 2,
            internal_index: 0,
        }),
        InstructionInfo::Activation(ActivationInstructionInfo {
            input: 0,
            activation: Activation::Sigmoid,
        }),
        InstructionInfo::Activation(ActivationInstructionInfo {
            input: 1,
            activation: Activation::Relu,
        }),
        InstructionInfo::Copy(CopyInstructionInfo {
            input: 0,
            output: 2,
            internal_index: 4,
        }),
        InstructionInfo::Copy(CopyInstructionInfo {
            input: 1,
            output: 2,
            internal_index: 2,
        }),
        InstructionInfo::Copy(CopyInstructionInfo {
            input: 2,
            output: 3,
            internal_index: 0,
        }),
    ];

    let model_info = InstructionModelInfo {
        features: Some(vec!["feature1".to_string(), "feature2".to_string()]),
        feature_size: None,
        computation_buffer_sizes: layer_sizes,
        instructions,
        weights: vec![],
        bias: vec![],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let model = InstructionModel::new(model_info).expect("Model creation should succeed");
    let inputs = vec![1.0, -1.0];
    let result = model.predict(&inputs).expect("Prediction should succeed");

    // The final result should be the copied values from layer 2 to layer 3
    // Based on the debug output, the final result contains the complete layer 2 content:
    // [original inputs, relu results, sigmoid results]
    assert!((result[0] - 1.0).abs() < DELTA); // Copy of original input[0] 
    assert!((result[1] - (-1.0)).abs() < DELTA); // Copy of original input[1]
    assert!((result[2] - 1.0).abs() < DELTA); // Copy of ReLU input[0] = max(0, 1) = 1
    assert!((result[3] - 0.0).abs() < DELTA); // Copy of ReLU input[1] = max(0, -1) = 0
    assert!((result[4] - 0.7311).abs() < DELTA); // Copy of sigmoid input[0]
    assert!((result[5] - 0.2689).abs() < DELTA); // Copy of sigmoid input[1]
}

#[test]
fn copy_masked_layers() {
    let layer_sizes = vec![3, 3, 2, 3];

    let instructions = vec![
        InstructionInfo::CopyMasked(CopyMaskedInstructionInfo {
            input: 0,
            output: 1,
            indexes: vec![1, 2, 0],
        }),
        InstructionInfo::CopyMasked(CopyMaskedInstructionInfo {
            input: 1,
            output: 2,
            indexes: vec![1, 0],
        }),
        InstructionInfo::CopyMasked(CopyMaskedInstructionInfo {
            input: 0,
            output: 3,
            indexes: vec![2, 0, 1],
        }),
    ];

    let model_info = InstructionModelInfo {
        features: Some(vec!["f1".to_string(), "f2".to_string(), "f3".to_string()]),
        feature_size: None,
        computation_buffer_sizes: layer_sizes,
        instructions,
        weights: vec![],
        bias: vec![],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let model = InstructionModel::new(model_info).expect("Model creation should succeed");
    let inputs = vec![1.0, -1.0, 2.0];
    let result = model.predict(&inputs).expect("Prediction should succeed");

    // Verify masked copy results
    assert!((result[0] - 2.0).abs() < DELTA); // Copy of original [2]
    assert!((result[1] - 1.0).abs() < DELTA); // Copy of original [0]
}

#[test]
fn element_wise() {
    let layer_sizes = vec![3, 3, 3];

    let parameters = vec![
        vec![-1.0, 0.0, 1.0],
        vec![5.0, 2.0, 2.0],
        vec![-1.0, 3.0, 1.5],
    ];

    let instructions = vec![
        InstructionInfo::ElemWiseAdd(ElemWiseAddInstructionInfo {
            input: 0,
            parameters: 0,
        }),
        InstructionInfo::Copy(CopyInstructionInfo {
            input: 0,
            output: 1,
            internal_index: 0,
        }),
        InstructionInfo::ElemWiseMul(ElemWiseMulInstructionInfo {
            input: 1,
            parameters: 1,
        }),
        InstructionInfo::Copy(CopyInstructionInfo {
            input: 1,
            output: 2,
            internal_index: 0,
        }),
        InstructionInfo::ElemWiseAdd(ElemWiseAddInstructionInfo {
            input: 2,
            parameters: 2,
        }),
        InstructionInfo::Activation(ActivationInstructionInfo {
            input: 2,
            activation: Activation::Relu,
        }),
    ];

    let model_info = InstructionModelInfo {
        features: Some(vec!["f1".to_string(), "f2".to_string(), "f3".to_string()]),
        feature_size: None,
        computation_buffer_sizes: layer_sizes,
        instructions,
        weights: vec![],
        bias: vec![],
        parameters: Some(parameters),
        maps: None,
        validation_data: None,
    };

    let model = InstructionModel::new(model_info).expect("Model creation should succeed");
    let inputs = vec![1.0, -1.0, 0.0];
    let result = model.predict(&inputs).expect("Prediction should succeed");

    // After element-wise operations and relu
    assert!((result[0] - 0.0).abs() < DELTA);
    assert!((result[1] - 1.0).abs() < DELTA);
    assert!((result[2] - 3.5).abs() < DELTA);
}

#[test]
fn element_wise_buffers() {
    let layer_sizes = vec![3, 3, 3, 3];

    let instructions = vec![
        InstructionInfo::ElemWiseBuffersAdd(ElemWiseBuffersAddInstructionInfo {
            input: vec![0, 1],
            output: 2,
        }),
        InstructionInfo::ElemWiseBuffersMul(ElemWiseBuffersMulInstructionInfo {
            input: vec![0, 1],
            output: 3,
        }),
    ];

    let model_info = InstructionModelInfo {
        features: Some(vec![
            "f1".to_string(),
            "f2".to_string(),
            "f3".to_string(),
            "f4".to_string(),
            "f5".to_string(),
            "f6".to_string(),
        ]),
        feature_size: Some(6),
        computation_buffer_sizes: layer_sizes,
        instructions,
        weights: vec![],
        bias: vec![],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let model = InstructionModel::new(model_info).expect("Model creation should succeed");
    let inputs = vec![1.0, 0.0, 0.5, -4.0, 3.5, -5.0];
    let result = model.predict(&inputs).expect("Prediction should succeed");

    // The final result contains the multiplication results from layer 3 (the last layer)
    // Layer 0: [1.0, 0.0, 0.5], Layer 1: [-4.0, 3.5, -5.0]
    // Layer 2 (addition): [-3.0, 3.5, -4.5]
    // Layer 3 (multiplication): [1.0*(-4.0), 0.0*3.5, 0.5*(-5.0)] = [-4.0, 0.0, -2.5]
    assert!((result[0] - (-4.0)).abs() < DELTA); // 1.0 * (-4.0) = -4.0
    assert!((result[1] - 0.0).abs() < DELTA); // 0.0 * 3.5 = 0.0  
    assert!((result[2] - (-2.5)).abs() < DELTA); // 0.5 * (-5.0) = -2.5
}

#[test]
fn complex_neural_network() {
    let layer_sizes = vec![2, 2, 1];
    let mut weights_layer0 = vec![vec![0.0; 2]; 2];
    let mut bias_layer0 = vec![0.0; 2];
    let mut weights_layer1 = vec![vec![0.0; 2]; 1];
    let mut bias_layer1 = vec![0.0; 1];

    create_weights_for_complex_neural_network(
        layer_sizes[0],
        &mut weights_layer0,
        &mut weights_layer1,
        &mut bias_layer0,
        &mut bias_layer1,
    );

    let instructions = vec![
        InstructionInfo::Dot(DotInstructionInfo {
            input: 0,
            output: 1,
            weights: 0,
            activation: Some(Activation::Relu),
        }),
        InstructionInfo::Dot(DotInstructionInfo {
            input: 1,
            output: 2,
            weights: 1,
            activation: Some(Activation::Sigmoid),
        }),
    ];

    let model_info = InstructionModelInfo {
        features: Some(vec!["feature1".to_string(), "feature2".to_string()]),
        feature_size: None,
        computation_buffer_sizes: layer_sizes,
        instructions,
        weights: vec![weights_layer0, weights_layer1],
        bias: vec![bias_layer0, bias_layer1],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let model = InstructionModel::new(model_info).expect("Model creation should succeed");
    let inputs = vec![1.0, -1.0];
    let result = model
        .predict_single(&inputs)
        .expect("Prediction should succeed");

    // result = sigmoid(relu(1 * 2 - 1 * 0.5 + 0.25) * 0.5 + relu(-1 * 2 + 1 * 0.5 - 0.25) * (-1) + 2)
    // = sigmoid(relu(1.75) * 0.5 + relu(-1.75) * (-1) + 2)
    // = sigmoid(1.75 * 0.5 + 0 * (-1) + 2) = sigmoid(2.875)
    assert!((result - 0.9466).abs() < DELTA);
}

#[test]
fn test_feature_size_calculation() {
    // Test with explicit features
    let model_info = InstructionModelInfo {
        features: Some(vec![
            "feature1".to_string(),
            "feature2[3]".to_string(),
            "feature3".to_string(),
        ]),
        feature_size: None,
        computation_buffer_sizes: vec![5],
        instructions: vec![],
        weights: vec![],
        bias: vec![],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let result = InstructionModel::calculate_feature_size(&model_info).unwrap();
    assert_eq!(result, 5); // 1 + 3 + 1

    // Test with feature size only
    let model_info2 = InstructionModelInfo {
        features: None,
        feature_size: Some(10),
        computation_buffer_sizes: vec![10],
        instructions: vec![],
        weights: vec![],
        bias: vec![],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let result2 = InstructionModel::calculate_feature_size(&model_info2).unwrap();
    assert_eq!(result2, 10);
}

#[test]
fn test_validation() {
    let layer_sizes = vec![2, 1];
    let mut weights = vec![vec![0.0; 2]; 1];
    let mut bias = vec![0.0; 1];

    weights[0][0] = 2.0;
    weights[0][1] = 0.5;
    bias[0] = 0.25;

    let validation_data = ValidationData {
        inputs: vec![vec![1.0, -1.0], vec![-1.0, 1.0]],
        expected_outputs: vec![vec![1.75], vec![-1.25]],
    };

    let model_info = InstructionModelInfo {
        features: Some(vec!["feature1".to_string(), "feature2".to_string()]),
        feature_size: None,
        computation_buffer_sizes: layer_sizes,
        instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
            input: 0,
            output: 1,
            weights: 0,
            activation: None,
        })],
        weights: vec![weights],
        bias: vec![bias],
        parameters: None,
        maps: None,
        validation_data: Some(validation_data),
    };

    // This should not panic since validation should pass
    let model = InstructionModel::new(model_info);
    assert!(
        model.is_ok(),
        "Model with valid validation data should be created successfully"
    );
}
