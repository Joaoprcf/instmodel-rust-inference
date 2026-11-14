mod common;

use common::{assert_close, info, model};
use instmodel_inference::{InstructionModel, InstructionModelInfo};
use serde_json::{Value, json};

const DELTA: f32 = 0.00005;

fn check(name: &str, model_json: Value, cases: &[(&[f32], &[f32])]) {
    let model = model(model_json);
    for (input, expected) in cases {
        assert_close(&model.predict(input).unwrap(), expected, DELTA, name);
    }
}

fn activation(name: &str) -> Value {
    json!({"buffer_sizes": [3], "instructions": [{"type": "ACTIVATION", "input": 0, "activation": name}]})
}

#[test]
fn dot_models() {
    check(
        "dense_with_validation_data",
        json!({"features": ["feature1", "feature2"], "buffer_sizes": [2, 1],
            "instructions": [{"type": "DOT", "input": 0, "output": 1, "weights": 0}],
            "weights": [[[2.0, 0.5]]], "bias": [[0.25]],
            "validation_data": {"inputs": [[1.0, -1.0], [-1.0, 1.0]], "expected_outputs": [[1.75], [-1.25]]}}),
        &[(&[1.0, -1.0], &[1.75]), (&[-1.0, 1.0], &[-1.25])],
    );
    // sigmoid(relu(1.75) * 0.5 - relu(-1.75) + 2) = sigmoid(2.875)
    check(
        "two_layer",
        json!({"buffer_sizes": [2, 2, 1],
            "instructions": [
                {"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "RELU"},
                {"type": "DOT", "input": 1, "output": 2, "weights": 1, "activation": "SIGMOID"}],
            "weights": [[[2.0, 0.5], [-2.0, -0.5]], [[0.5, -1.0]]], "bias": [[0.25, -0.25], [2.0]]}),
        &[(&[1.0, -1.0], &[0.9466])],
    );
}

#[test]
fn activations() {
    let cases: [(&str, &[f32], &[f32]); 5] = [
        ("RELU", &[1.0, -1.0, 0.5], &[1.0, 0.0, 0.5]),
        ("SIGMOID", &[1.0, 0.0, -0.5], &[0.7311, 0.5, 0.3775]),
        (
            "SOFTMAX",
            &[1.0, 2.0, 3.0],
            &[0.09003057, 0.24472847, 0.66524096],
        ),
        ("SIGN", &[-3.0, 0.0, 2.0], &[-1.0, 0.0, 1.0]),
        (
            "EXP",
            &[0.0, 1.0, -1.0],
            &[1.0, std::f32::consts::E, 1.0 / std::f32::consts::E],
        ),
    ];
    for (name, input, expected) in cases {
        check(name, activation(name), &[(input, expected)]);
    }
}

#[test]
fn copy_ops() {
    check(
        "copy_into_offsets",
        json!({"buffer_sizes": [2, 2, 6, 6], "instructions": [
            {"type": "COPY", "input": 0, "output": 1, "internal_index": 0},
            {"type": "COPY", "input": 0, "output": 2, "internal_index": 0},
            {"type": "ACTIVATION", "input": 0, "activation": "SIGMOID"},
            {"type": "ACTIVATION", "input": 1, "activation": "RELU"},
            {"type": "COPY", "input": 0, "output": 2, "internal_index": 4},
            {"type": "COPY", "input": 1, "output": 2, "internal_index": 2},
            {"type": "COPY", "input": 2, "output": 3, "internal_index": 0}]}),
        &[(&[1.0, -1.0], &[1.0, -1.0, 1.0, 0.0, 0.7311, 0.2689])],
    );
    check(
        "copy_masked",
        json!({"buffer_sizes": [3, 3, 2, 3], "instructions": [
            {"type": "COPY_MASKED", "input": 0, "output": 1, "indexes": [1, 2, 0]},
            {"type": "COPY_MASKED", "input": 1, "output": 2, "indexes": [1, 0]},
            {"type": "COPY_MASKED", "input": 0, "output": 3, "indexes": [2, 0, 1]}]}),
        &[(&[1.0, -1.0, 2.0], &[2.0, 1.0, -1.0])],
    );
}

#[test]
fn elementwise_ops() {
    check(
        "params_add_mul",
        json!({"buffer_sizes": [3, 3, 3], "instructions": [
            {"type": "ADD_ELEMENTWISE", "input": 0, "parameters": 0},
            {"type": "COPY", "input": 0, "output": 1, "internal_index": 0},
            {"type": "MUL_ELEMENTWISE", "input": 1, "parameters": 1},
            {"type": "COPY", "input": 1, "output": 2, "internal_index": 0},
            {"type": "ADD_ELEMENTWISE", "input": 2, "parameters": 2},
            {"type": "ACTIVATION", "input": 2, "activation": "RELU"}],
            "parameters": [[-1.0, 0.0, 1.0], [5.0, 2.0, 2.0], [-1.0, 3.0, 1.5]]}),
        &[(&[1.0, -1.0, 0.0], &[0.0, 1.0, 3.5])],
    );
    check(
        "buffers_add_mul",
        json!({"feature_size": 6, "buffer_sizes": [3, 3, 3, 3], "instructions": [
            {"type": "ADD_ELEMENTWISE_BUFFERS", "input": [0, 1], "output": 2},
            {"type": "MULTIPLY_ELEMENTWISE_BUFFERS", "input": [0, 1], "output": 3}]}),
        &[(&[1.0, 0.0, 0.5, -4.0, 3.5, -5.0], &[-4.0, 0.0, -2.5])],
    );
}

#[test]
fn clip_elementwise() {
    let clip = |parameters: Value, min: Value, max: Value| {
        json!({"buffer_sizes": [4], "parameters": parameters, "instructions": [
            {"type": "CLIP_ELEMENTWISE", "input": 0, "parameters_min": min, "parameters_max": max}]})
    };
    check(
        "both_bounds",
        clip(
            json!([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
            json!(0),
            json!(1),
        ),
        &[(&[-5.0, 0.3, 0.7, 5.0], &[0.0, 0.3, 0.7, 1.0])],
    );
    check(
        "min_only",
        clip(json!([[-1.0, 0.0, 0.5, 10.0]]), json!(0), Value::Null),
        &[(&[-5.0, -0.2, 0.7, 5.0], &[-1.0, 0.0, 0.7, 10.0])],
    );
    check(
        "max_only",
        clip(json!([[1.0, 0.5, 0.0, -10.0]]), Value::Null, json!(0)),
        &[(&[-5.0, 0.7, 0.7, 5.0], &[-5.0, 0.5, 0.0, -10.0])],
    );
}

#[test]
fn reduce_sum() {
    check(
        "reduce_sum",
        json!({"buffer_sizes": [4, 1], "instructions": [{"type": "REDUCE_SUM", "input": 0, "output": 1}]}),
        &[
            (&[1.0, 2.0, 3.0, 4.0], &[10.0]),
            (&[1.0, -2.0, 3.0, -4.0], &[-2.0]),
        ],
    );
    check(
        "relu_then_reduce_sum",
        json!({"buffer_sizes": [3, 3, 1], "instructions": [
            {"type": "ACTIVATION", "input": 0, "activation": "RELU"},
            {"type": "COPY", "input": 0, "output": 1, "internal_index": 0},
            {"type": "REDUCE_SUM", "input": 1, "output": 2}]}),
        &[(&[2.0, -1.0, 3.0], &[5.0])],
    );
}

#[test]
fn attention() {
    let attention = |size: usize, weights: Value, bias: Value, reduce: bool| {
        let mut instructions =
            vec![json!({"type": "ATTENTION", "input": 0, "key": 1, "output": 2, "weights": 0})];
        let mut buffer_sizes = vec![size, size, size];
        if reduce {
            instructions.push(json!({"type": "REDUCE_SUM", "input": 2, "output": 3}));
            buffer_sizes.push(1);
        }
        json!({"feature_size": 2 * size, "buffer_sizes": buffer_sizes, "instructions": instructions,
            "weights": [weights], "bias": [bias]})
    };
    // softmax(key @ W.T + b) * query
    check(
        "identity",
        attention(2, json!([[1.0, 0.0], [0.0, 1.0]]), json!([0.0, 0.0]), false),
        &[(&[1.0, 2.0, 0.0, 0.0], &[0.5, 1.0])],
    );
    check(
        "weighted",
        attention(
            2,
            json!([[2.0, 0.0], [0.0, 1.0]]),
            json!([1.0, -1.0]),
            false,
        ),
        &[(&[1.0, 1.0, 1.0, 1.0], &[0.95257413, 0.04742587])],
    );
    check(
        "then_reduce_sum",
        attention(
            3,
            json!([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            json!([0.0, 0.0, 0.0]),
            true,
        ),
        &[(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], &[2.0])],
    );
}

#[test]
fn buffer_heads() {
    let heads = |op: &str, data: usize, heads: usize, reduce: bool| {
        let mut instructions = vec![json!({"type": op, "input": [0, 1], "output": 2})];
        let mut buffer_sizes = vec![data, heads, data];
        if reduce {
            instructions.push(json!({"type": "REDUCE_SUM", "input": 2, "output": 3}));
            buffer_sizes.push(1);
        }
        json!({"feature_size": data + heads, "buffer_sizes": buffer_sizes, "instructions": instructions})
    };
    let blocks = |values: [f32; 4]| -> Vec<f32> { values.iter().flat_map(|&v| [v; 5]).collect() };

    let ones_and_heads: Vec<f32> = [vec![1.0; 20], vec![2.0, 3.0, 4.0, 5.0]].concat();
    let zeros_and_heads: Vec<f32> = [vec![0.0; 20], vec![10.0, 20.0, 30.0, 40.0]].concat();
    check(
        "multiply_4_heads",
        heads("MULTIPLY_BUFFER_HEADS", 20, 4, false),
        &[(&ones_and_heads, &blocks([2.0, 3.0, 4.0, 5.0]))],
    );
    check(
        "multiply_2_heads",
        heads("MULTIPLY_BUFFER_HEADS", 8, 2, false),
        &[(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, -1.0],
            &[2.0, 4.0, 6.0, 8.0, -5.0, -6.0, -7.0, -8.0],
        )],
    );
    check(
        "add_4_heads",
        heads("ADD_BUFFER_HEADS", 20, 4, false),
        &[(&zeros_and_heads, &blocks([10.0, 20.0, 30.0, 40.0]))],
    );
    check(
        "add_2_heads",
        heads("ADD_BUFFER_HEADS", 6, 2, false),
        &[(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, -50.0],
            &[101.0, 102.0, 103.0, -46.0, -45.0, -44.0],
        )],
    );
    check(
        "multiply_then_reduce_sum",
        heads("MULTIPLY_BUFFER_HEADS", 8, 2, true),
        &[(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0], &[20.0])],
    );
}

#[test]
fn feature_size_calculation() {
    let size = |json: Value| -> usize {
        let info: InstructionModelInfo = info(json);
        InstructionModel::calculate_feature_size(&info).unwrap()
    };
    assert_eq!(
        size(
            json!({"features": ["feature1", "feature2[3]", "feature3"], "buffer_sizes": [5], "instructions": []})
        ),
        5
    );
    assert_eq!(
        size(json!({"feature_size": 10, "buffer_sizes": [10], "instructions": []})),
        10
    );
}
