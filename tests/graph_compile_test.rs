//! Integration tests for the graph DSL and compiler: layout offsets, weight
//! sharing, per-op numeric parity against hand-computed references, in-place
//! fusion locked through the real model validator, parallel-predict parity,
//! and plan-time error cases.

use std::collections::HashMap;

use instmodel_inference::activation::Activation;
use instmodel_inference::errors::GraphError;
use instmodel_inference::graph::Graph;
use instmodel_inference::params::{ParamCopy, ParamKind};
use instmodel_inference::{
    Constant, InstructionModel, InstructionModelInfo, ParamLayout, PredictConfig,
};

/// Distinct, exactly-f32-representable θ.
fn theta_for(len: usize) -> Vec<f32> {
    (0..len).map(|k| k as f32 * 0.25 - 3.0).collect()
}

#[test]
fn two_layer_mlp_layout_offsets() {
    let graph = Graph::new();
    let x = graph.input(3, None);
    let hidden = graph.dense(&x, 4, Some(Activation::Relu));
    let y = graph.dense(&hidden, 1, None);
    let model_graph = graph.model(vec![&x], &y);

    let layout = model_graph.weights_map().unwrap();
    assert_eq!(layout.total, 21);
    assert_eq!(layout.weights_len, 16);
    assert_eq!(layout.bias_len, 5);
    assert_eq!(layout.weight_shapes, vec![[4, 3], [1, 4]]);
    assert_eq!(layout.bias_lens, vec![4, 1]);
    assert_eq!(
        layout.copies,
        vec![
            ParamCopy {
                kind: ParamKind::Weights,
                dest_offset: 0,
                src_offset: 0,
                size: 12
            },
            ParamCopy {
                kind: ParamKind::Bias,
                dest_offset: 0,
                src_offset: 12,
                size: 4
            },
            ParamCopy {
                kind: ParamKind::Weights,
                dest_offset: 12,
                src_offset: 16,
                size: 4
            },
            ParamCopy {
                kind: ParamKind::Bias,
                dest_offset: 4,
                src_offset: 20,
                size: 1
            },
        ]
    );
}

#[test]
fn shared_weight_is_one_slot_and_patches_forward_pass() {
    let graph = Graph::new();
    let x = graph.input(2, None);
    let shared = graph.weight();
    let hidden = graph.dense_shared(&x, 2, None, shared);
    let y = graph.dense_shared(&hidden, 2, None, shared);
    let model_graph = graph.model(vec![&x], &y);

    let layout = model_graph.weights_map().unwrap();
    assert_eq!(layout.weight_shapes, vec![[2, 2]]);
    assert_eq!(layout.total, 6);

    // W = [[0,1],[1,0]], b = [1,2]: one application maps [x1,x2] to
    // [x2+1, x1+2]; applying the same tensor twice gives [x1+3, x2+3].
    let theta = [0.0, 1.0, 1.0, 0.0, 1.0, 2.0];
    let model = model_graph.to_model(&theta).unwrap();
    assert_eq!(model.predict(&[5.0, 7.0]).unwrap(), vec![8.0, 10.0]);
}

#[test]
fn concat_forward_parity() {
    let graph = Graph::new();
    let a = graph.input(2, None);
    let b = graph.input(3, None);
    let y = graph.concat(&[&b, &a]);
    let model_graph = graph.model(vec![&a, &b], &y);

    let model = model_graph.to_model(&[]).unwrap();
    assert_eq!(
        model.predict(&[1.0, 2.0, 10.0, 20.0, 30.0]).unwrap(),
        vec![10.0, 20.0, 30.0, 1.0, 2.0]
    );
}

#[test]
fn fanout_diamond_compiles_once_and_predicts() {
    let graph = Graph::new();
    let x = graph.input(2, None);
    let shared = graph.dense(&x, 2, None);
    let left = graph.dense(&shared, 2, None);
    let right = graph.dense(&shared, 2, None);
    let y = graph.add(&[&left, &right]);
    let model_graph = graph.model(vec![&x], &y);

    let layout = model_graph.weights_map().unwrap();
    assert_eq!(layout.weight_shapes, vec![[2, 2], [2, 2], [2, 2]]);

    // shared = I·x, left = I·shared, right = 2I·shared → y = 3x.
    let theta = [
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, // shared: identity
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, // left: identity
        2.0, 0.0, 0.0, 2.0, 0.0, 0.0, // right: 2·identity
    ];
    let model = model_graph.to_model(&theta).unwrap();
    assert_eq!(model.predict(&[3.0, -4.0]).unwrap(), vec![9.0, -12.0]);
}

#[test]
fn deep_multibranch_layout_shapes() {
    let graph = Graph::new();
    let a = graph.input(3, None);
    let b = graph.input(2, None);
    let h1 = graph.dense(&a, 5, Some(Activation::Relu));
    let wide1 = graph.concat(&[&h1, &a, &b]);
    let h2 = graph.dense(&wide1, 6, Some(Activation::Relu));
    let h3 = graph.dense(&b, 4, Some(Activation::Relu));
    let wide2 = graph.concat(&[&h2, &h3]);
    let h4 = graph.dense(&wide2, 3, None);
    let y = graph.dense(&h4, 1, None);
    let model_graph = graph.model(vec![&a, &b], &y);

    let layout = model_graph.weights_map().unwrap();
    assert_eq!(
        layout.weight_shapes,
        vec![[5, 3], [6, 10], [4, 2], [3, 10], [1, 3]]
    );
    assert_eq!(layout.total, 135);

    let theta = theta_for(layout.total);
    let info = model_graph.compile(&theta).unwrap();
    assert_eq!(ParamLayout::from_info(&info).unwrap(), layout);
    let model = InstructionModel::new(info).unwrap();
    assert_eq!(model.predict(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap().len(), 1);
}

#[test]
fn theta_len_mismatch_is_rejected() {
    let graph = Graph::new();
    let x = graph.input(3, None);
    let hidden = graph.dense(&x, 4, Some(Activation::Relu));
    let y = graph.dense(&hidden, 1, None);
    let model_graph = graph.model(vec![&x], &y);

    assert!(matches!(
        model_graph.compile(&[0.0; 3]),
        Err(GraphError::ThetaLenMismatch {
            expected: 21,
            got: 3
        })
    ));
}

#[test]
fn shared_weight_shape_mismatch_is_rejected() {
    let graph = Graph::new();
    let x = graph.input(3, None);
    let shared = graph.weight();
    let h1 = graph.dense_shared(&x, 4, None, shared);
    let h2 = graph.dense_shared(&h1, 4, None, shared);
    let model_graph = graph.model(vec![&x], &h2);

    assert!(matches!(
        model_graph.weights_map(),
        Err(GraphError::SharedWeightShapeMismatch {
            first: [4, 3],
            again: [4, 4],
            ..
        })
    ));
}

#[test]
fn unreached_declared_input_is_rejected() {
    let graph = Graph::new();
    let x = graph.input(2, None);
    let unused = graph.input(2, None);
    let y = graph.dense(&x, 1, None);
    let model_graph = graph.model(vec![&x, &unused], &y);

    assert!(matches!(
        model_graph.weights_map(),
        Err(GraphError::InputNotVisited)
    ));
}

#[test]
fn graph_without_inputs_is_rejected() {
    let graph = Graph::new();
    let x = graph.input(2, None);
    let model_graph = graph.model(vec![], &x);

    assert!(matches!(
        model_graph.weights_map(),
        Err(GraphError::NoInputs)
    ));
}

#[test]
fn feature_names_survive_when_every_scalar_is_named() {
    let graph = Graph::new();
    let x = graph.input(2, Some(vec!["a".to_string(), "b".to_string()]));
    let z = graph.input(1, Some(vec!["c".to_string()]));
    let y = graph.dense(&graph.concat(&[&x, &z]), 1, None);
    let model_graph = graph.model(vec![&x, &z], &y);

    let info = model_graph.compile_zeroed().unwrap();
    assert_eq!(
        info.features,
        Some(vec!["a".to_string(), "b".to_string(), "c".to_string()])
    );
    assert_eq!(info.feature_size, Some(3));
}

#[test]
fn partially_named_inputs_expose_no_feature_names() {
    let graph = Graph::new();
    let x = graph.input(2, Some(vec!["a".to_string(), "b".to_string()]));
    let z = graph.input(1, None);
    let y = graph.dense(&graph.concat(&[&x, &z]), 1, None);
    let model_graph = graph.model(vec![&x, &z], &y);

    let info = model_graph.compile_zeroed().unwrap();
    assert_eq!(info.features, None);
    assert_eq!(info.feature_size, Some(3));
}

#[test]
fn dense_with_activation_parity() {
    let graph = Graph::new();
    let x = graph.input(3, None);
    let y = graph.dense(&x, 2, Some(Activation::Relu));
    let model_graph = graph.model(vec![&x], &y);

    // W = [[1,1,1],[-1,-1,-1]], b = [0.5, 0.5].
    let theta = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.5, 0.5];
    let model = model_graph.to_model(&theta).unwrap();
    assert_eq!(model.predict(&[1.0, 2.0, 3.0]).unwrap(), vec![6.5, 0.0]);
}

#[test]
fn gather_parity() {
    let graph = Graph::new();
    let x = graph.input(4, None);
    let y = graph.gather(&x, vec![3, 0, 3]);
    let model_graph = graph.model(vec![&x], &y);

    let model = model_graph.to_model(&[]).unwrap();
    assert_eq!(
        model.predict(&[10.0, 20.0, 30.0, 40.0]).unwrap(),
        vec![40.0, 10.0, 40.0]
    );
}

#[test]
fn activation_parity() {
    let graph = Graph::new();
    let x = graph.input(3, None);
    let y = graph.activation(&x, Activation::Relu);
    let model_graph = graph.model(vec![&x], &y);

    let model = model_graph.to_model(&[]).unwrap();
    assert_eq!(
        model.predict(&[-1.0, 0.0, 2.0]).unwrap(),
        vec![0.0, 0.0, 2.0]
    );
}

#[test]
fn clip_parity() {
    let graph = Graph::new();
    let x = graph.input(3, None);
    let y = graph.clip(
        &x,
        Some(Constant::Scalar(-1.0)),
        Some(Constant::Scalar(1.0)),
    );
    let model_graph = graph.model(vec![&x], &y);
    let model = model_graph.to_model(&[]).unwrap();
    assert_eq!(
        model.predict(&[-5.0, 0.5, 3.0]).unwrap(),
        vec![-1.0, 0.5, 1.0]
    );

    let lower_only = Graph::new();
    let x = lower_only.input(3, None);
    let y = lower_only.clip(&x, Some(Constant::Scalar(0.0)), None);
    let model = lower_only.model(vec![&x], &y).to_model(&[]).unwrap();
    assert_eq!(
        model.predict(&[-5.0, 0.5, 3.0]).unwrap(),
        vec![0.0, 0.5, 3.0]
    );
}

#[test]
fn add_const_parity() {
    let graph = Graph::new();
    let x = graph.input(2, None);
    let y = graph.add_const(&x, Constant::PerElement(vec![10.0, 20.0]));
    let model_graph = graph.model(vec![&x], &y);

    let model = model_graph.to_model(&[]).unwrap();
    assert_eq!(model.predict(&[1.0, 2.0]).unwrap(), vec![11.0, 22.0]);
}

#[test]
fn mul_const_parity() {
    let graph = Graph::new();
    let x = graph.input(2, None);
    let y = graph.mul_const(&x, Constant::Scalar(2.0));
    let model_graph = graph.model(vec![&x], &y);

    let model = model_graph.to_model(&[]).unwrap();
    assert_eq!(model.predict(&[1.5, -3.0]).unwrap(), vec![3.0, -6.0]);
}

#[test]
fn normalize_parity() {
    let graph = Graph::new();
    let x = graph.input(2, None);
    let y = graph.normalize(&x, vec![1.0, 2.0], vec![2.0, 4.0]);
    let model_graph = graph.model(vec![&x], &y);

    let model = model_graph.to_model(&[]).unwrap();
    assert_eq!(model.predict(&[3.0, 6.0]).unwrap(), vec![1.0, 1.0]);
}

#[test]
fn add_buffers_parity() {
    let graph = Graph::new();
    let a = graph.input(2, None);
    let b = graph.input(2, None);
    let y = graph.add(&[&a, &b, &a]);
    let model_graph = graph.model(vec![&a, &b], &y);

    let model = model_graph.to_model(&[]).unwrap();
    assert_eq!(
        model.predict(&[1.0, 2.0, 10.0, 20.0]).unwrap(),
        vec![12.0, 24.0]
    );
}

#[test]
fn mul_buffers_parity() {
    let graph = Graph::new();
    let a = graph.input(2, None);
    let b = graph.input(2, None);
    let y = graph.mul(&[&a, &b]);
    let model_graph = graph.model(vec![&a, &b], &y);

    let model = model_graph.to_model(&[]).unwrap();
    assert_eq!(
        model.predict(&[1.0, 2.0, 10.0, 20.0]).unwrap(),
        vec![10.0, 40.0]
    );
}

#[test]
fn add_heads_parity() {
    let graph = Graph::new();
    let data = graph.input(6, None);
    let heads = graph.input(2, None);
    let y = graph.add_heads(&data, &heads);
    let model_graph = graph.model(vec![&data, &heads], &y);

    let model = model_graph.to_model(&[]).unwrap();
    assert_eq!(
        model
            .predict(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0])
            .unwrap(),
        vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]
    );
}

#[test]
fn mul_heads_parity() {
    let graph = Graph::new();
    let data = graph.input(6, None);
    let heads = graph.input(2, None);
    let y = graph.mul_heads(&data, &heads);
    let model_graph = graph.model(vec![&data, &heads], &y);

    let model = model_graph.to_model(&[]).unwrap();
    assert_eq!(
        model
            .predict(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0])
            .unwrap(),
        vec![2.0, 4.0, 6.0, 12.0, 15.0, 18.0]
    );
}

#[test]
fn reduce_sum_parity() {
    let graph = Graph::new();
    let x = graph.input(4, None);
    let y = graph.reduce_sum(&x);
    let model_graph = graph.model(vec![&x], &y);

    let model = model_graph.to_model(&[]).unwrap();
    assert_eq!(model.predict(&[1.0, 2.0, 3.0, 4.0]).unwrap(), vec![10.0]);
}

#[test]
fn attention_parity() {
    let graph = Graph::new();
    let query = graph.input(2, None);
    let key = graph.input(2, None);
    let y = graph.attention(&query, &key);
    let model_graph = graph.model(vec![&query, &key], &y);

    let layout = model_graph.weights_map().unwrap();
    assert_eq!(layout.weight_shapes, vec![[2, 2]]);

    // W = I, b = 0, key = [0,0] → scores [0,0] → softmax [0.5,0.5]
    // → output = 0.5 · query (exact in f32).
    let theta = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let model = model_graph.to_model(&theta).unwrap();
    assert_eq!(
        model.predict(&[3.0, 5.0, 0.0, 0.0]).unwrap(),
        vec![1.5, 2.5]
    );
}

#[test]
fn map_transform_parity() {
    let graph = Graph::new();
    let key = graph.input(1, None);
    let mut map = HashMap::new();
    map.insert("1".to_string(), vec![10.0, 20.0]);
    map.insert("3".to_string(), vec![30.0, 40.0]);
    let y = graph.map_transform(&key, map, vec![0.5, 0.25]);
    let model_graph = graph.model(vec![&key], &y);

    let model = model_graph.to_model(&[]).unwrap();
    // The lookup key is round(input): 1.2 → "1", 2.6 → "3", 9 → default.
    assert_eq!(model.predict(&[1.2]).unwrap(), vec![10.0, 20.0]);
    assert_eq!(model.predict(&[2.6]).unwrap(), vec![30.0, 40.0]);
    assert_eq!(model.predict(&[9.0]).unwrap(), vec![0.5, 0.25]);
}

#[test]
fn deep_fused_in_place_chain_builds_and_predicts() {
    let graph = Graph::new();
    let x = graph.input(1, None);
    let mut current = graph.dense(&x, 1, None);
    for _ in 0..5_000 {
        current = graph.activation(&current, Activation::Relu);
    }
    let model_graph = graph.model(vec![&x], &current);

    // Identity dense (W=[[1]], b=[0]); Relu on a positive value is identity,
    // so 5,000 fused in-place activations must pass the dead-write validator
    // and return the input unchanged.
    let model = model_graph.to_model(&[1.0, 0.0]).unwrap();
    assert_eq!(model.predict(&[0.75]).unwrap(), vec![0.75]);
}

#[test]
fn in_place_output_is_materialized_into_last_buffer() {
    let graph = Graph::new();
    let x = graph.input(2, None);
    let hidden = graph.dense(&x, 2, None);
    let y = graph.activation(&hidden, Activation::Sigmoid);
    let model_graph = graph.model(vec![&x], &y);

    let info = model_graph.compile_zeroed().unwrap();
    // Dot, Copy (materialize), Activation — output lands in the last buffer.
    assert_eq!(info.computation_buffer_sizes, vec![2, 2, 2]);
    assert_eq!(info.instructions.len(), 3);

    let model = InstructionModel::new(info).unwrap();
    // Zero weights → dense output 0 → sigmoid 0.5.
    assert_eq!(model.predict(&[1.0, 2.0]).unwrap(), vec![0.5, 0.5]);
}

/// A graph touching most op kinds, used for parallel parity and layout
/// round-trip checks.
fn mixed_graph(graph: &Graph) -> instmodel_inference::ModelGraph {
    let x = graph.input(4, None);
    let heads = graph.input(2, None);
    let gathered = graph.gather(&x, vec![3, 2, 1, 0]);
    let normalized = graph.normalize(&gathered, vec![0.5; 4], vec![2.0; 4]);
    let clipped = graph.clip(
        &normalized,
        Some(Constant::Scalar(-0.75)),
        Some(Constant::Scalar(0.75)),
    );
    let weighted = graph.mul_heads(&clipped, &heads);
    let summed = graph.reduce_sum(&weighted);
    let combined = graph.concat(&[&summed, &x]);
    let hidden = graph.dense(&combined, 3, Some(Activation::Tanh));
    let y = graph.dense(&hidden, 2, None);
    graph.model(vec![&x, &heads], &y)
}

#[test]
fn parallel_predict_matches_sequential_on_mixed_graph() {
    let graph = Graph::new();
    let model_graph = mixed_graph(&graph);
    let layout = model_graph.weights_map().unwrap();
    assert_eq!(layout.weight_shapes, vec![[3, 5], [2, 3]]);

    let theta = theta_for(layout.total);
    let model = model_graph.to_model(&theta).unwrap();

    let samples: Vec<Vec<f32>> = (0..6)
        .map(|s| {
            let base = s as f32;
            vec![
                base * 0.5 - 1.0,
                1.25 - base,
                base * 0.125,
                -0.5 * base,
                1.0 + base * 0.25,
                2.0 - base * 0.5,
            ]
        })
        .collect();
    let flat: Vec<f32> = samples.iter().flatten().copied().collect();

    let parallel = model
        .predict_parallel(&flat, PredictConfig::new().with_threads(4))
        .unwrap();
    assert_eq!(parallel.num_samples(), samples.len());

    for (index, sample) in samples.iter().enumerate() {
        let sequential = model.predict(sample).unwrap();
        assert_eq!(parallel.get_result(index).unwrap(), sequential.as_slice());
    }
}

#[test]
fn compiled_info_layout_round_trips_and_survives_serde() {
    let graph = Graph::new();
    let model_graph = mixed_graph(&graph);
    let layout = model_graph.weights_map().unwrap();
    let theta = theta_for(layout.total);

    let info = model_graph.compile(&theta).unwrap();
    assert_eq!(ParamLayout::from_info(&info).unwrap(), layout);
    assert_eq!(ParamLayout::flatten_from_info(&info), theta);

    let json = serde_json::to_string(&info).unwrap();
    let reloaded: InstructionModelInfo = serde_json::from_str(&json).unwrap();

    let original = InstructionModel::new(info).unwrap();
    let restored = InstructionModel::new(reloaded).unwrap();
    let input = [0.3, -0.6, 0.9, 0.1, 1.5, -2.0];
    assert_eq!(
        original.predict(&input).unwrap(),
        restored.predict(&input).unwrap()
    );
}

#[test]
fn plan_time_errors_are_reported() {
    let graph = Graph::new();
    let x = graph.input(3, None);

    let empty_gather = graph.gather(&x, vec![]);
    assert!(matches!(
        graph.model(vec![&x], &empty_gather).weights_map(),
        Err(GraphError::EmptyGather)
    ));

    let out_of_bounds = graph.gather(&x, vec![3]);
    assert!(matches!(
        graph.model(vec![&x], &out_of_bounds).weights_map(),
        Err(GraphError::GatherIndexOutOfBounds {
            index: 3,
            input_size: 3
        })
    ));

    let wrong_len = graph.add_const(&x, Constant::PerElement(vec![1.0, 2.0]));
    assert!(matches!(
        graph.model(vec![&x], &wrong_len).weights_map(),
        Err(GraphError::ConstantLengthMismatch {
            expected: 3,
            got: 2
        })
    ));

    let unbounded = graph.clip(&x, None, None);
    assert!(matches!(
        graph.model(vec![&x], &unbounded).weights_map(),
        Err(GraphError::ClipWithoutBounds)
    ));

    let heads = graph.input(2, None);
    let not_divisible = graph.mul_heads(&x, &heads);
    assert!(matches!(
        graph.model(vec![&x, &heads], &not_divisible).weights_map(),
        Err(GraphError::HeadsNotDivisible {
            data_size: 3,
            heads_size: 2
        })
    ));

    let mut bad_map = HashMap::new();
    bad_map.insert("1".to_string(), vec![1.0, 2.0, 3.0]);
    let bad_values = graph.map_transform(&x, bad_map, vec![0.0, 0.0]);
    assert!(matches!(
        graph.model(vec![&x], &bad_values).weights_map(),
        Err(GraphError::MapValueLengthMismatch { ref key, expected: 2, got: 3 }) if key == "1"
    ));

    let lonely = graph.add(&[&x]);
    assert!(matches!(
        graph.model(vec![&x], &lonely).weights_map(),
        Err(GraphError::InsufficientOperands {
            op: "add",
            minimum: 2,
            got: 1
        })
    ));

    let short = graph.input(2, None);
    let mismatched = graph.add(&[&x, &short]);
    assert!(matches!(
        graph.model(vec![&x, &short], &mismatched).weights_map(),
        Err(GraphError::OperandSizeMismatch {
            op: "add",
            expected: 3,
            got: 2
        })
    ));
}

#[test]
fn structural_errors_are_reported() {
    let graph = Graph::new();
    let x = graph.input(2, None);

    let undeclared = graph.input(2, None);
    let uses_undeclared = graph.dense(&graph.concat(&[&x, &undeclared]), 1, None);
    assert!(matches!(
        graph.model(vec![&x], &uses_undeclared).weights_map(),
        Err(GraphError::UndeclaredInput)
    ));

    let y = graph.dense(&x, 1, None);
    assert!(matches!(
        graph.model(vec![&x, &x], &y).weights_map(),
        Err(GraphError::DuplicateInput)
    ));

    let zero = graph.input(0, None);
    let uses_zero = graph.dense(&zero, 1, None);
    assert!(matches!(
        graph.model(vec![&zero], &uses_zero).weights_map(),
        Err(GraphError::ZeroSizedInput)
    ));

    let half_named = graph.input(2, Some(vec!["only_one".to_string()]));
    let uses_half = graph.dense(&half_named, 1, None);
    assert!(matches!(
        graph.model(vec![&half_named], &uses_half).weights_map(),
        Err(GraphError::FeatureNamesLengthMismatch {
            expected: 2,
            got: 1
        })
    ));

    let bracketed = graph.input(1, Some(vec!["x[0]".to_string()]));
    let uses_bracketed = graph.dense(&bracketed, 1, None);
    assert!(matches!(
        graph.model(vec![&bracketed], &uses_bracketed).weights_map(),
        Err(GraphError::InvalidFeatureName { .. })
    ));
}

#[test]
fn input_passthrough_output_predicts_identity() {
    let graph = Graph::new();
    let x = graph.input(3, None);
    let model_graph = graph.model(vec![&x], &x);

    let model = model_graph.to_model(&[]).unwrap();
    assert_eq!(
        model.predict(&[4.0, 5.0, 6.0]).unwrap(),
        vec![4.0, 5.0, 6.0]
    );
}
