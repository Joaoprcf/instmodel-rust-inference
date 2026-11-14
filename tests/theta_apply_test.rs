mod common;

use instmodel_inference::errors::InstructionModelError;
use instmodel_inference::params::JaggedParams;
use instmodel_inference::{InstructionModel, InstructionModelInfo, ParamLayout};
use serde_json::{Value, json};

type Build = fn(JaggedParams) -> InstructionModelInfo;
type Shapes = &'static [[usize; 2]];

fn with_params(mut model: Value, params: JaggedParams) -> InstructionModelInfo {
    model["weights"] = json!(params.weights);
    model["bias"] = json!(params.bias);
    common::info(model)
}

/// 3 → 4 (Relu) → 2 MLP with two weight slots.
fn mlp(params: JaggedParams) -> InstructionModelInfo {
    with_params(
        json!({"buffer_sizes": [3, 4, 2], "instructions": [
            {"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "RELU"},
            {"type": "DOT", "input": 1, "output": 2, "weights": 1}]}),
        params,
    )
}

/// Two DOTs sharing weight slot 0.
fn shared_slot(params: JaggedParams) -> InstructionModelInfo {
    with_params(
        json!({"buffer_sizes": [2, 2, 2], "instructions": [
            {"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "TANH"},
            {"type": "DOT", "input": 1, "output": 2, "weights": 0}]}),
        params,
    )
}

fn attention(params: JaggedParams) -> InstructionModelInfo {
    with_params(
        json!({"feature_size": 4, "buffer_sizes": [2, 2, 2], "instructions": [
            {"type": "ATTENTION", "input": 0, "key": 1, "output": 2, "weights": 0}]}),
        params,
    )
}

const MLP_SHAPES: &[[usize; 2]] = &[[4, 3], [2, 4]];

/// Distinct, exactly-f32-representable θ.
fn theta_for(len: usize) -> Vec<f32> {
    (0..len).map(|k| k as f32 * 0.25 - 3.0).collect()
}

fn build(builder: Build, shapes: &[[usize; 2]], theta: &[f32]) -> InstructionModel {
    let layout = ParamLayout::from_shapes(shapes);
    InstructionModel::new(builder(layout.scatter_to_jagged(theta).unwrap())).unwrap()
}

fn zeroed(builder: Build, shapes: &[[usize; 2]]) -> InstructionModel {
    build(
        builder,
        shapes,
        &vec![0.0; ParamLayout::from_shapes(shapes).total],
    )
}

#[test]
fn apply_theta_matches_freshly_built_model_bitwise() {
    let cases: [(&str, Build, Shapes, &[&[f32]]); 3] = [
        (
            "mlp",
            mlp,
            MLP_SHAPES,
            &[
                &[0.0, 0.0, 0.0],
                &[1.0, -2.0, 3.0],
                &[0.5, 0.25, -0.125],
                &[-10.0, 7.0, 0.3],
            ],
        ),
        ("shared_slot", shared_slot, &[[2, 2]], &[&[0.7, -0.4]]),
        ("attention", attention, &[[2, 2]], &[&[1.0, 2.0, 0.5, -0.5]]),
    ];
    for (name, builder, shapes, inputs) in cases {
        let layout = ParamLayout::from_shapes(shapes);
        let theta = theta_for(layout.total);
        assert_eq!(
            ParamLayout::from_info(&builder(layout.scatter_to_jagged(&theta).unwrap())).unwrap(),
            layout,
            "{name}"
        );

        let mut mutated = zeroed(builder, shapes);
        assert_eq!(mutated.theta_len(), layout.total, "{name}");
        mutated.apply_theta(&theta).unwrap();
        let mut read_back = vec![0.0f32; layout.total];
        mutated.read_theta(&mut read_back).unwrap();
        assert_eq!(read_back, theta, "{name}");

        let rebuilt = build(builder, shapes, &theta);
        for input in inputs {
            assert_eq!(
                mutated.predict(input).unwrap(),
                rebuilt.predict(input).unwrap(),
                "{name}"
            );
        }
    }
}

#[test]
fn read_theta_recovers_construction_weights() {
    let layout = ParamLayout::from_shapes(MLP_SHAPES);
    let theta = theta_for(layout.total);
    let info = mlp(layout.scatter_to_jagged(&theta).unwrap());
    assert_eq!(ParamLayout::flatten_from_info(&info), theta);

    let model = InstructionModel::new(info).unwrap();
    let mut read_back = vec![0.0f32; model.theta_len()];
    model.read_theta(&mut read_back).unwrap();
    assert_eq!(read_back, theta);
}

#[test]
fn wrong_theta_length_is_rejected() {
    let mut model = zeroed(mlp, MLP_SHAPES);
    assert!(matches!(
        model.apply_theta(&[1.0, 2.0]),
        Err(InstructionModelError::ThetaLengthMismatch {
            expected: 26,
            got: 2
        })
    ));
    assert!(matches!(
        model.read_theta(&mut [0.0f32; 3]),
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
    let theta_a = theta_for(ParamLayout::from_shapes(MLP_SHAPES).total);
    let theta_b: Vec<f32> = theta_a.iter().map(|v| v * -2.0).collect();

    let mut original = zeroed(mlp, MLP_SHAPES);
    original.apply_theta(&theta_a).unwrap();
    let mut clone = original.try_clone().unwrap();
    clone.apply_theta(&theta_b).unwrap();

    let input = [0.3f32, -0.6, 0.9];
    assert_ne!(
        original.predict(&input).unwrap(),
        clone.predict(&input).unwrap()
    );
    let mut original_theta = vec![0.0f32; theta_a.len()];
    original.read_theta(&mut original_theta).unwrap();
    assert_eq!(original_theta, theta_a);
}

#[test]
fn jagged_weights_rejected_by_from_info() {
    let layout = ParamLayout::from_shapes(MLP_SHAPES);
    let mut info = mlp(layout.scatter_to_jagged(&vec![0.0; layout.total]).unwrap());
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
