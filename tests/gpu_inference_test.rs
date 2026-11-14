//! CPU/GPU parity through the embeddable WGSL interpreter. Requires a GPU adapter
//! (excluded from CI).

mod common;

use common::gpu::gpu_predict;
use common::{assert_close, info};
use instmodel_inference::activation::Activation;
use instmodel_inference::gpu::GpuModel;
use instmodel_inference::instruction_model::InstructionModel;
use serde_json::{Value, json};

const TOLERANCE: f32 = 1e-6;

/// Asserts GPU == CPU for every input and returns the CPU outputs.
fn parity(name: &str, model_json: Value, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let info = info(model_json);
    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
    assert_eq!(
        cpu_model.get_feature_size(),
        gpu_model.feature_size(),
        "{name}"
    );
    assert_eq!(
        cpu_model.get_output_size(),
        gpu_model.output_size(),
        "{name}"
    );
    inputs
        .iter()
        .enumerate()
        .map(|(i, input)| {
            let cpu = cpu_model.predict(input).expect("CPU prediction failed");
            let gpu = gpu_predict(&gpu_model, input).expect("no GPU adapter available");
            assert_close(&gpu, &cpu, TOLERANCE, &format!("{name}[{i}]"));
            cpu
        })
        .collect()
}

fn assert_sums_to_one(name: &str, values: &[f32]) {
    let sum: f32 = values.iter().sum();
    assert!((sum - 1.0).abs() < TOLERANCE, "{name}: softmax sum {sum}");
}

fn single_activation(activation: Activation, size: usize) -> Value {
    json!({"buffer_sizes": [size], "instructions": [{"type": "ACTIVATION", "input": 0, "activation": activation}]})
}

/// Rows of `(row * cols + col) * scale - offset`.
fn ramp(rows: usize, cols: usize, scale: f32, offset: f32) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| (i * cols + j) as f32 * scale - offset)
                .collect()
        })
        .collect()
}

#[test]
fn single_instruction_parity() {
    let cases: Vec<(&str, Value, Vec<f32>, Vec<f32>)> = vec![
        (
            "dot",
            json!({"features": ["f1", "f2"], "feature_size": 2, "buffer_sizes": [2, 1],
                "instructions": [{"type": "DOT", "input": 0, "output": 1, "weights": 0}],
                "weights": [[[2.0, 0.5]]], "bias": [[0.25]]}),
            vec![1.0, -1.0],
            vec![1.75],
        ),
        (
            "dot_relu",
            json!({"buffer_sizes": [3, 2],
                "instructions": [{"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "RELU"}],
                "weights": [[[1.0, -1.0, 0.5], [-2.0, 1.0, 0.0]]], "bias": [[0.0, 0.0]]}),
            vec![1.0, 2.0, 3.0],
            vec![0.5, 0.0],
        ),
        (
            "dot_sigmoid",
            json!({"buffer_sizes": [2, 1],
                "instructions": [{"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "SIGMOID"}],
                "weights": [[[1.0, 1.0]]], "bias": [[0.0]]}),
            vec![0.0, 0.0],
            vec![0.5],
        ),
        (
            "relu",
            single_activation(Activation::Relu, 4),
            vec![-1.0, 0.0, 1.0, 2.0],
            vec![0.0, 0.0, 1.0, 2.0],
        ),
        (
            "add_elementwise",
            json!({"buffer_sizes": [3], "instructions": [{"type": "ADD_ELEMENTWISE", "input": 0, "parameters": 0}],
                "parameters": [[1.0, 2.0, 3.0]]}),
            vec![10.0, 20.0, 30.0],
            vec![11.0, 22.0, 33.0],
        ),
        (
            "mul_elementwise",
            json!({"buffer_sizes": [3], "instructions": [{"type": "MUL_ELEMENTWISE", "input": 0, "parameters": 0}],
                "parameters": [[2.0, 0.5, 0.0]]}),
            vec![10.0, 20.0, 30.0],
            vec![20.0, 10.0, 0.0],
        ),
        (
            "copy_internal_index",
            json!({"buffer_sizes": [2, 4],
                "instructions": [{"type": "COPY", "input": 0, "output": 1, "internal_index": 1}]}),
            vec![5.0, 7.0],
            vec![0.0, 5.0, 7.0, 0.0],
        ),
    ];
    for (name, model_json, input, expected) in cases {
        let cpu = parity(name, model_json, &[input]);
        assert_close(&cpu[0], &expected, TOLERANCE, name);
    }
}

#[test]
fn activation_parity() {
    use Activation::*;
    let groups: [(&[Activation], Vec<f32>); 4] = [
        (
            &[
                Relu, Sigmoid, Tanh, Sqrt, Log, Log10, Inverse, Gelu, Softplus, Sign,
            ],
            vec![0.5, 1.0, 2.0, 4.0],
        ),
        (
            &[Relu, Sigmoid, Tanh, Inverse, Gelu, Softplus, Sign],
            vec![-3.0, -0.5, 0.0, 0.5, 3.0],
        ),
        // Small inputs keep exp's f32 magnitude within the absolute tolerance.
        (&[Exp], vec![-2.0, -0.5, 0.0, 0.5, 1.0]),
        (&[Softmax], vec![1.0, 2.0, 3.0, 4.0]),
    ];
    for (activations, input) in groups {
        for &activation in activations {
            let name = format!("{activation:?}");
            let cpu = parity(
                &name,
                single_activation(activation, input.len()),
                std::slice::from_ref(&input),
            );
            if activation == Softmax {
                assert_sums_to_one(&name, &cpu[0]);
            }
        }
    }
}

#[test]
fn dot_activation_parity() {
    for activation in [
        None,
        Some(Activation::Relu),
        Some(Activation::Sigmoid),
        Some(Activation::Tanh),
        Some(Activation::Softmax),
    ] {
        let name = format!("dot_{activation:?}");
        let cpu = parity(
            &name,
            json!({"buffer_sizes": [3, 4],
                "instructions": [{"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": activation}],
                "weights": [[[0.5, -0.3, 0.2], [-0.4, 0.6, -0.1], [0.3, -0.2, 0.4], [-0.1, 0.5, -0.3]]],
                "bias": [[0.1, -0.1, 0.05, -0.05]]}),
            &[vec![1.0, -0.5, 0.25]],
        );
        if activation == Some(Activation::Softmax) {
            assert_sums_to_one(&name, &cpu[0]);
        }
    }
}

#[test]
fn network_parity() {
    let cases: Vec<(&str, Value, Vec<Vec<f32>>, bool)> = vec![
        (
            "mlp_4_8_4_2",
            json!({"buffer_sizes": [4, 8, 4, 2],
                "instructions": [
                    {"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "RELU"},
                    {"type": "DOT", "input": 1, "output": 2, "weights": 1, "activation": "RELU"},
                    {"type": "DOT", "input": 2, "output": 3, "weights": 2, "activation": "SIGMOID"}],
                "weights": [
                    [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.1, -0.2, 0.3, -0.4], [-0.5, 0.6, -0.7, 0.8],
                     [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5]],
                    vec![vec![0.1; 8], vec![0.2; 8], vec![0.3; 8], vec![0.4; 8]],
                    [[0.5, 0.5, 0.5, 0.5], [0.6, 0.6, 0.6, 0.6]]],
                "bias": [vec![0.1; 8], vec![0.1; 4], [0.0, 0.0]]}),
            vec![vec![1.0, 2.0, 3.0, 4.0]],
            false,
        ),
        (
            "mlp_multiple_inputs",
            json!({"buffer_sizes": [4, 8, 2],
                "instructions": [
                    {"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "RELU"},
                    {"type": "DOT", "input": 1, "output": 2, "weights": 1, "activation": "SIGMOID"}],
                "weights": [
                    [[0.1, -0.2, 0.3, -0.4], [-0.5, 0.6, -0.7, 0.8], [0.9, -0.1, 0.2, -0.3], [-0.4, 0.5, -0.6, 0.7],
                     [0.8, -0.9, 0.1, -0.2], [-0.3, 0.4, -0.5, 0.6], [0.7, -0.8, 0.9, -0.1], [-0.2, 0.3, -0.4, 0.5]],
                    [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]]],
                "bias": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.0, 0.0]]}),
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0, 1.0],
                vec![-1.0, -1.0, -1.0, -1.0],
                vec![1.0, -1.0, 1.0, -1.0],
                vec![0.5, -0.5, 0.25, -0.25],
                vec![100.0, -100.0, 50.0, -50.0],
                vec![0.001, 0.002, 0.003, 0.004],
            ],
            false,
        ),
        (
            "deep_softmax",
            json!({"buffer_sizes": [8, 16, 16, 8, 4, 2],
                "instructions": [
                    {"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "RELU"},
                    {"type": "DOT", "input": 1, "output": 2, "weights": 1, "activation": "TANH"},
                    {"type": "DOT", "input": 2, "output": 3, "weights": 2, "activation": "RELU"},
                    {"type": "DOT", "input": 3, "output": 4, "weights": 3, "activation": "SIGMOID"},
                    {"type": "DOT", "input": 4, "output": 5, "weights": 4, "activation": "SOFTMAX"}],
                "weights": [ramp(16, 8, 0.01, 0.5), ramp(16, 16, 0.005, 0.4), ramp(8, 16, 0.008, 0.3),
                            ramp(4, 8, 0.02, 0.2), ramp(2, 4, 0.05, 0.1)],
                "bias": [vec![0.1; 16], vec![0.05; 16], vec![0.02; 8], vec![0.01; 4], vec![0.0; 2]]}),
            vec![vec![0.5, -0.3, 0.8, -0.2, 0.1, 0.9, -0.7, 0.4]],
            true,
        ),
        (
            "mixed_instructions",
            json!({"buffer_sizes": [4, 4, 4, 2],
                "instructions": [
                    {"type": "MUL_ELEMENTWISE", "input": 0, "parameters": 0},
                    {"type": "ADD_ELEMENTWISE", "input": 0, "parameters": 1},
                    {"type": "COPY", "input": 0, "output": 1, "internal_index": 0},
                    {"type": "ACTIVATION", "input": 1, "activation": "TANH"},
                    {"type": "COPY", "input": 1, "output": 2, "internal_index": 0},
                    {"type": "DOT", "input": 2, "output": 3, "weights": 0, "activation": "SIGMOID"}],
                "weights": [[[0.5, 0.5, 0.5, 0.5], [-0.5, -0.5, -0.5, -0.5]]], "bias": [[0.1, -0.1]],
                "parameters": [[2.0, 0.5, 1.5, 0.25], [0.1, 0.2, 0.3, 0.4]]}),
            vec![vec![1.0, 2.0, 3.0, 4.0]],
            false,
        ),
        (
            "sigmoid_edge_values",
            json!({"buffer_sizes": [4, 4],
                "instructions": [{"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "SIGMOID"}],
                "weights": [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
                "bias": [[0.0, 0.0, 0.0, 0.0]]}),
            vec![
                vec![0.0; 4],
                vec![10.0; 4],
                vec![-10.0; 4],
                vec![1e-10, -1e-10, 1e-10, -1e-10],
            ],
            false,
        ),
        (
            "bracketed_features",
            json!({"features": ["feature_a", "feature_b[3]", "feature_c"], "feature_size": 5, "buffer_sizes": [5, 3],
                "instructions": [{"type": "DOT", "input": 0, "output": 1, "weights": 0}],
                "weights": [[[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]],
                "bias": [[0.1, 0.2, 0.3]]}),
            vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]],
            false,
        ),
        (
            "fan_out_fan_in",
            json!({"buffer_sizes": [4, 3, 3, 6, 2],
                "instructions": [
                    {"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "RELU"},
                    {"type": "DOT", "input": 0, "output": 2, "weights": 1, "activation": "TANH"},
                    {"type": "COPY", "input": 1, "output": 3, "internal_index": 0},
                    {"type": "COPY", "input": 2, "output": 3, "internal_index": 3},
                    {"type": "DOT", "input": 3, "output": 4, "weights": 2, "activation": "SIGMOID"}],
                "weights": [
                    [[0.5, -0.3, 0.2, 0.1], [-0.4, 0.6, -0.1, 0.3], [0.3, -0.2, 0.4, -0.5]],
                    [[-0.2, 0.4, -0.3, 0.5], [0.1, -0.5, 0.2, -0.4], [-0.3, 0.1, -0.4, 0.2]],
                    [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]]],
                "bias": [[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1], [0.0, 0.0]]}),
            vec![
                vec![1.0, 0.5, -0.5, -1.0],
                vec![0.0; 4],
                vec![1.0; 4],
                vec![-1.0, 2.0, -3.0, 4.0],
            ],
            false,
        ),
        (
            "residual_concat",
            json!({"buffer_sizes": [4, 4, 4, 8, 4],
                "instructions": [
                    {"type": "COPY", "input": 0, "output": 1, "internal_index": 0},
                    {"type": "DOT", "input": 0, "output": 2, "weights": 0, "activation": "TANH"},
                    {"type": "COPY", "input": 1, "output": 3, "internal_index": 0},
                    {"type": "COPY", "input": 2, "output": 3, "internal_index": 4},
                    {"type": "DOT", "input": 3, "output": 4, "weights": 1, "activation": "RELU"}],
                "weights": [
                    [[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.1, 0.0], [0.0, 0.1, 0.9, 0.1], [0.0, 0.0, 0.1, 0.9]],
                    [[0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                     [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5]]],
                "bias": [vec![0.0; 4], vec![0.0; 4]]}),
            vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![-1.0, -2.0, -3.0, -4.0],
                vec![0.5, -0.5, 0.5, -0.5],
            ],
            false,
        ),
        (
            "wide_fan_out_softmax",
            json!({"buffer_sizes": [4, 2, 2, 2, 6, 3],
                "instructions": [
                    {"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "RELU"},
                    {"type": "DOT", "input": 0, "output": 2, "weights": 1, "activation": "TANH"},
                    {"type": "DOT", "input": 0, "output": 3, "weights": 2, "activation": "SIGMOID"},
                    {"type": "COPY", "input": 1, "output": 4, "internal_index": 0},
                    {"type": "COPY", "input": 2, "output": 4, "internal_index": 2},
                    {"type": "COPY", "input": 3, "output": 4, "internal_index": 4},
                    {"type": "DOT", "input": 4, "output": 5, "weights": 3, "activation": "SOFTMAX"}],
                "weights": [
                    [[0.5, -0.3, 0.2, 0.1], [-0.4, 0.6, -0.1, 0.3]],
                    [[-0.2, 0.4, -0.3, 0.5], [0.1, -0.5, 0.2, -0.4]],
                    [[0.3, -0.1, 0.4, -0.2], [-0.5, 0.2, -0.3, 0.1]],
                    [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [-0.1, -0.2, -0.3, 0.1, 0.2, 0.3], [0.2, -0.1, 0.1, -0.2, 0.3, -0.3]]],
                "bias": [[0.1, -0.1], [0.05, -0.05], [-0.1, 0.1], [0.0, 0.0, 0.0]]}),
            vec![vec![1.0, -1.0, 0.5, -0.5], vec![0.0; 4], vec![2.0; 4]],
            true,
        ),
    ];
    for (name, model_json, inputs, softmax_output) in cases {
        for output in parity(name, model_json, &inputs) {
            if softmax_output {
                assert_sums_to_one(name, &output);
            }
        }
    }
}
