//! Integration tests for flat-θ mapping and in-place model mutation:
//! `ParamLayout::from_info`, `apply_theta`/`read_theta`, `try_clone`, and
//! parity between mutated models and freshly built ones.

use instmodel_inference::activation::Activation;
use instmodel_inference::errors::InstructionModelError;
use instmodel_inference::instruction_model_info::{
    AttentionInstructionInfo, DotInstructionInfo, InstructionInfo,
};
use instmodel_inference::{InstructionModel, InstructionModelInfo, ParamLayout};

/// 3 → 4 (Relu) → 2 MLP with two weight slots.
fn mlp_info(weights: Vec<Vec<Vec<f32>>>, bias: Vec<Vec<f32>>) -> InstructionModelInfo {
    InstructionModelInfo {
        features: None,
        feature_size: Some(3),
        computation_buffer_sizes: vec![3, 4, 2],
        instructions: vec![
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
                activation: None,
            }),
        ],
        weights,
        bias,
        parameters: None,
        maps: None,
        validation_data: None,
    }
}

fn mlp_shapes() -> Vec<[usize; 2]> {
    vec![[4, 3], [2, 4]]
}

fn zeroed_mlp_info() -> InstructionModelInfo {
    let layout = ParamLayout::from_shapes(&mlp_shapes());
    let jagged = layout.scatter_to_jagged(&vec![0.0; layout.total]).unwrap();
    mlp_info(jagged.weights, jagged.bias)
}

/// Distinct, exactly-f32-representable θ.
fn theta_for(len: usize) -> Vec<f32> {
    (0..len).map(|k| k as f32 * 0.25 - 3.0).collect()
}

#[test]
fn from_info_matches_shapes_for_mlp() {
    let layout = ParamLayout::from_info(&zeroed_mlp_info()).unwrap();
    assert_eq!(layout, ParamLayout::from_shapes(&mlp_shapes()));
    assert_eq!(layout.total, 12 + 4 + 8 + 2);
}

#[test]
fn apply_theta_matches_freshly_built_model_bitwise() {
    let layout = ParamLayout::from_shapes(&mlp_shapes());
    let theta = theta_for(layout.total);

    let mut mutated = InstructionModel::new(zeroed_mlp_info()).unwrap();
    assert_eq!(mutated.theta_len(), layout.total);
    mutated.apply_theta(&theta).unwrap();

    let jagged = layout.scatter_to_jagged(&theta).unwrap();
    let rebuilt = InstructionModel::new(mlp_info(jagged.weights, jagged.bias)).unwrap();

    for input in [
        [0.0f32, 0.0, 0.0],
        [1.0, -2.0, 3.0],
        [0.5, 0.25, -0.125],
        [-10.0, 7.0, 0.3],
    ] {
        let mutated_output = mutated.predict(&input).unwrap();
        let rebuilt_output = rebuilt.predict(&input).unwrap();
        // Same instruction kernels, same weights: bit-for-bit equality.
        assert_eq!(mutated_output, rebuilt_output);
    }
}

#[test]
fn shared_slot_patches_every_owner() {
    // Two DOTs referencing the SAME weight slot: apply_theta must patch both
    // instructions' private copies.
    let shared_info = |weights: Vec<Vec<Vec<f32>>>, bias: Vec<Vec<f32>>| InstructionModelInfo {
        features: None,
        feature_size: Some(2),
        computation_buffer_sizes: vec![2, 2, 2],
        instructions: vec![
            InstructionInfo::Dot(DotInstructionInfo {
                input: 0,
                output: 1,
                weights: 0,
                activation: Some(Activation::Tanh),
            }),
            InstructionInfo::Dot(DotInstructionInfo {
                input: 1,
                output: 2,
                weights: 0,
                activation: None,
            }),
        ],
        weights,
        bias,
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let layout = ParamLayout::from_shapes(&[[2, 2]]);
    let theta = theta_for(layout.total);
    let zero = layout.scatter_to_jagged(&vec![0.0; layout.total]).unwrap();

    let mut mutated = InstructionModel::new(shared_info(zero.weights, zero.bias)).unwrap();
    assert_eq!(mutated.theta_len(), 6);
    mutated.apply_theta(&theta).unwrap();

    let jagged = layout.scatter_to_jagged(&theta).unwrap();
    let rebuilt = InstructionModel::new(shared_info(jagged.weights, jagged.bias)).unwrap();

    let input = [0.7f32, -0.4];
    assert_eq!(
        mutated.predict(&input).unwrap(),
        rebuilt.predict(&input).unwrap()
    );
}

#[test]
fn attention_slot_participates_in_theta() {
    let attention_info = |weights: Vec<Vec<Vec<f32>>>, bias: Vec<Vec<f32>>| InstructionModelInfo {
        features: None,
        feature_size: Some(4),
        computation_buffer_sizes: vec![2, 2, 2],
        instructions: vec![InstructionInfo::Attention(AttentionInstructionInfo {
            input: 0,
            key: 1,
            output: 2,
            weights: 0,
        })],
        weights,
        bias,
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let layout = ParamLayout::from_shapes(&[[2, 2]]);
    let theta = theta_for(layout.total);
    let zero = layout.scatter_to_jagged(&vec![0.0; layout.total]).unwrap();

    let mut mutated = InstructionModel::new(attention_info(zero.weights, zero.bias)).unwrap();
    assert_eq!(mutated.theta_len(), 6);
    mutated.apply_theta(&theta).unwrap();

    let mut read_back = vec![0.0f32; 6];
    mutated.read_theta(&mut read_back).unwrap();
    assert_eq!(read_back, theta);

    let jagged = layout.scatter_to_jagged(&theta).unwrap();
    let rebuilt = InstructionModel::new(attention_info(jagged.weights, jagged.bias)).unwrap();

    let input = [1.0f32, 2.0, 0.5, -0.5];
    assert_eq!(
        mutated.predict(&input).unwrap(),
        rebuilt.predict(&input).unwrap()
    );
}

#[test]
fn read_theta_recovers_construction_weights() {
    let layout = ParamLayout::from_shapes(&mlp_shapes());
    let theta = theta_for(layout.total);
    let jagged = layout.scatter_to_jagged(&theta).unwrap();
    let info = mlp_info(jagged.weights, jagged.bias);

    let flattened = ParamLayout::flatten_from_info(&info);
    assert_eq!(flattened, theta);

    let model = InstructionModel::new(info).unwrap();
    let mut read_back = vec![0.0f32; model.theta_len()];
    model.read_theta(&mut read_back).unwrap();
    assert_eq!(read_back, theta);
}

#[test]
fn wrong_theta_length_is_rejected() {
    let mut model = InstructionModel::new(zeroed_mlp_info()).unwrap();
    assert!(matches!(
        model.apply_theta(&[1.0, 2.0]),
        Err(InstructionModelError::ThetaLengthMismatch {
            expected: 26,
            got: 2
        })
    ));

    let mut too_short = vec![0.0f32; 3];
    assert!(matches!(
        model.read_theta(&mut too_short),
        Err(InstructionModelError::ThetaLengthMismatch {
            expected: 26,
            got: 3
        })
    ));
}

#[test]
fn test_built_models_have_no_binding() {
    let mut model = InstructionModel::new_for_test(vec![2, 2], vec![], 2).unwrap();
    assert_eq!(model.theta_len(), 0);
    assert!(model.param_layout().is_none());
    assert!(matches!(
        model.apply_theta(&[1.0]),
        Err(InstructionModelError::ThetaUnsupported { .. })
    ));
}

#[test]
fn try_clone_gives_independent_models() {
    let layout = ParamLayout::from_shapes(&mlp_shapes());
    let theta_a = theta_for(layout.total);
    let theta_b: Vec<f32> = theta_a.iter().map(|v| v * -2.0).collect();

    let mut original = InstructionModel::new(zeroed_mlp_info()).unwrap();
    original.apply_theta(&theta_a).unwrap();

    let mut clone = original.try_clone().unwrap();
    clone.apply_theta(&theta_b).unwrap();

    let input = [0.3f32, -0.6, 0.9];
    let original_output = original.predict(&input).unwrap();
    let clone_output = clone.predict(&input).unwrap();
    assert_ne!(original_output, clone_output);

    // The original still carries theta_a.
    let mut original_theta = vec![0.0f32; layout.total];
    original.read_theta(&mut original_theta).unwrap();
    assert_eq!(original_theta, theta_a);
}

#[test]
fn jagged_weights_rejected_by_from_info() {
    let mut info = zeroed_mlp_info();
    info.weights[0][2].pop();
    assert!(matches!(
        ParamLayout::from_info(&info),
        Err(InstructionModelError::JaggedWeightsMatrix {
            slot: 0,
            row: 2,
            expected: 3,
            got: 2
        })
    ));
}
