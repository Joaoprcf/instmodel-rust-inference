//! CPU/GPU parity for the version-2 opcodes, authored through the graph DSL.
//! GPU halves self-skip when no adapter is available.

mod common;

use common::assert_close;
use common::gpu::gpu_predict;
use instmodel_inference::InstructionModel;
use instmodel_inference::activation::Activation;
use instmodel_inference::gpu::{GpuModel, get_instmodel_wgsl};
use instmodel_inference::graph::{Constant, Graph, ModelGraph};
use instmodel_inference::params::ParamLayout;

const EXACT_TOLERANCE: f32 = 1e-6;
const DOT_TOLERANCE: f32 = 1e-4;

/// Checks the CPU result (against `expected` when given), then GPU against CPU.
fn assert_parity(
    info: instmodel_inference::InstructionModelInfo,
    input: &[f32],
    expected: Option<&[f32]>,
    tolerance: f32,
    name: &str,
) {
    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let cpu_result = cpu_model.predict(input).expect("CPU prediction failed");
    if let Some(expected) = expected {
        assert_close(&cpu_result, expected, EXACT_TOLERANCE, name);
    }
    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
    match gpu_predict(&gpu_model, input) {
        Some(gpu_result) => assert_close(&gpu_result, &cpu_result, tolerance, name),
        None => eprintln!("{name}: skipping GPU half - no adapter available"),
    }
}

type Build = fn(&Graph) -> ModelGraph;

#[test]
fn opcode_parity() {
    let three_inputs = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0];
    let heads_input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 10.0];
    let clip_input = [-2.0, -0.5, 0.5, 2.0];
    let cases: [(&str, Build, &[f32], &[f32]); 9] = [
        (
            "copy_masked",
            |g| {
                let x = g.input(4, None);
                g.model(vec![&x], &g.gather(&x, vec![3, 1, 1, 0, 2]))
            },
            &[10.0, 20.0, 30.0, 40.0],
            &[40.0, 20.0, 20.0, 10.0, 30.0],
        ),
        (
            "clip_min_only",
            |g| {
                let x = g.input(4, None);
                g.model(vec![&x], &g.clip(&x, Some(Constant::Scalar(-1.0)), None))
            },
            &clip_input,
            &[-1.0, -0.5, 0.5, 2.0],
        ),
        (
            "clip_max_only",
            |g| {
                let x = g.input(4, None);
                let max = Constant::PerElement(vec![0.0, 0.25, 0.5, 1.0]);
                g.model(vec![&x], &g.clip(&x, None, Some(max)))
            },
            &clip_input,
            &[-2.0, -0.5, 0.5, 1.0],
        ),
        (
            "clip_both_bounds",
            |g| {
                let x = g.input(4, None);
                let y = g.clip(
                    &x,
                    Some(Constant::Scalar(-1.0)),
                    Some(Constant::Scalar(1.0)),
                );
                g.model(vec![&x], &y)
            },
            &clip_input,
            &[-1.0, -0.5, 0.5, 1.0],
        ),
        (
            "elem_wise_buffers_add",
            |g| {
                let (a, b, c) = (g.input(3, None), g.input(3, None), g.input(3, None));
                g.model(vec![&a, &b, &c], &g.add(&[&a, &b, &c]))
            },
            &three_inputs,
            &[111.0, 222.0, 333.0],
        ),
        (
            "elem_wise_buffers_mul",
            |g| {
                let (a, b, c) = (g.input(3, None), g.input(3, None), g.input(3, None));
                g.model(vec![&a, &b, &c], &g.mul(&[&a, &b, &c]))
            },
            &three_inputs,
            &[1000.0, 8000.0, 27000.0],
        ),
        (
            "multiply_buffer_heads",
            |g| {
                let (data, heads) = (g.input(6, None), g.input(2, None));
                g.model(vec![&data, &heads], &g.mul_heads(&data, &heads))
            },
            &heads_input,
            &[2.0, 4.0, 6.0, 40.0, 50.0, 60.0],
        ),
        (
            "add_buffer_heads",
            |g| {
                let (data, heads) = (g.input(6, None), g.input(2, None));
                g.model(vec![&data, &heads], &g.add_heads(&data, &heads))
            },
            &heads_input,
            &[3.0, 4.0, 5.0, 14.0, 15.0, 16.0],
        ),
        (
            "reduce_sum",
            |g| {
                let x = g.input(5, None);
                g.model(vec![&x], &g.reduce_sum(&x))
            },
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[15.0],
        ),
    ];
    for (name, build, input, expected) in cases {
        let info = build(&Graph::new())
            .compile(&[])
            .expect("graph compile failed");
        assert_parity(info, input, Some(expected), EXACT_TOLERANCE, name);
    }
}

/// Every version-2 opcode plus DOT/COPY in one model.
fn mixed_model_graph() -> (ModelGraph, Vec<f32>) {
    let graph = Graph::new();
    let x = graph.input(4, None);
    let heads = graph.input(2, None);
    let gathered = graph.gather(&x, vec![3, 2, 1, 0]);
    let normalized = graph.normalize(&gathered, vec![1.0; 4], vec![2.0; 4]);
    let clipped = graph.clip(
        &normalized,
        Some(Constant::Scalar(-0.75)),
        Some(Constant::Scalar(0.75)),
    );
    let scaled = graph.mul_heads(&clipped, &heads);
    let total = graph.reduce_sum(&scaled);
    let combined = graph.concat(&[&total, &clipped]);
    let doubled = graph.add(&[&combined, &combined]);
    let hidden = graph.dense(&doubled, 3, Some(Activation::Tanh));
    let output = graph.dense(&hidden, 2, None);
    let model_graph = graph.model(vec![&x, &heads], &output);

    let layout = model_graph.weights_map().expect("weights_map failed");
    let theta: Vec<f32> = (0..layout.total).map(|k| (k as f32) * 0.25 - 3.0).collect();
    (model_graph, theta)
}

#[test]
fn mixed_graph_pipeline_parity() {
    let (model_graph, theta) = mixed_model_graph();
    let info = model_graph.compile(&theta).expect("graph compile failed");
    let input = [0.5, -1.5, 2.5, -3.5, 2.0, -0.5];
    let cpu_result = InstructionModel::new(info.clone())
        .unwrap()
        .predict(&input)
        .unwrap();
    assert_eq!(cpu_result.len(), 2);
    assert!(cpu_result.iter().all(|value| value.is_finite()));
    assert_parity(info, &input, None, DOT_TOLERANCE, "mixed_pipeline");
}

#[test]
fn payload_params_do_not_disturb_weights_region() {
    // Pointer payloads land in the params region, so the weights region must
    // stay byte-identical to canonical flat theta.
    let (model_graph, theta) = mixed_model_graph();
    let info = model_graph.compile(&theta).expect("graph compile failed");
    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");

    let mut read_back = vec![0.0f32; gpu_model.theta_len()];
    gpu_model
        .read_theta(&mut read_back)
        .expect("read_theta failed");
    assert_eq!(read_back, theta);
    assert_eq!(ParamLayout::flatten_from_info(&info), theta);
}

#[test]
fn instmodel_wgsl_includes_new_opcodes() {
    let wgsl = get_instmodel_wgsl(1024);
    for needle in [
        "fn execute_copy_masked(",
        "fn execute_clip_elementwise(",
        "fn execute_elem_wise_buffers_add(",
        "fn execute_elem_wise_buffers_mul(",
        "fn execute_multiply_buffer_heads(",
        "fn execute_add_buffer_heads(",
        "fn execute_reduce_sum(",
        "const OPCODE_COPY_MASKED: u32 = 6u;",
        "const OPCODE_REDUCE_SUM: u32 = 12u;",
        "const CLIP_BOUND_NONE: u32 = 0xffffffffu;",
    ] {
        assert!(wgsl.contains(needle), "generated WGSL missing `{needle}`");
    }
}
