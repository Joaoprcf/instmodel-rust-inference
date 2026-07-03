//! CPU/GPU parity tests for the version-2 opcode set (0x06–0x0C).
//!
//! Models are authored with the graph DSL so these tests exercise the full
//! graph → `InstructionModelInfo` → GPU blob path. GPU halves self-skip when
//! no adapter is available; the CPU expectations always run.

use instmodel_inference::InstructionModel;
use instmodel_inference::activation::Activation;
use instmodel_inference::gpu::{GpuModel, get_instmodel_wgsl};
use instmodel_inference::graph::{Constant, Graph, ModelGraph};
use instmodel_inference::params::ParamLayout;
use pollster::FutureExt;
use wgpu::util::DeviceExt;

/// Exact ops (copies, clamps, sums over a handful of elements).
const EXACT_TOLERANCE: f32 = 1e-6;
/// Paths through DOT reductions and transcendental activations.
const DOT_TOLERANCE: f32 = 1e-4;

const TEST_SHADER_TEMPLATE: &str = include_str!("shaders/test_shader.wgsl");

fn create_test_shader(compute_buffer_size: u32) -> String {
    let instmodel_wgsl = get_instmodel_wgsl(compute_buffer_size);
    let compute_buffer_decl = format!("var compute_buffer: array<f32, {}>;", compute_buffer_size);

    TEST_SHADER_TEMPLATE
        .replace(
            "// INSTMODEL_WGSL_PLACEHOLDER - replaced at runtime with actual instmodel code",
            &instmodel_wgsl,
        )
        .replace(
            "// COMPUTE_BUFFER_DECLARATION - replaced at runtime with actual size",
            &compute_buffer_decl,
        )
}

/// Runs one prediction on the GPU; `None` when no adapter is available.
fn gpu_predict(gpu_model: &GpuModel, input: &[f32]) -> Option<Vec<f32>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .block_on()?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .block_on()
        .ok()?;

    let shader_source = create_test_shader(gpu_model.compute_buffer_size() as u32);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Opcode Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Model Buffer"),
        contents: gpu_model.as_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_size = gpu_model.output_size();
    let output_bytes = (output_size * std::mem::size_of::<f32>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: model_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_bytes);
    queue.submit(Some(encoder.finish()));

    let slice = staging_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    Some(result)
}

fn compare(reference: &[f32], candidate: &[f32], tolerance: f32, name: &str) {
    assert_eq!(
        reference.len(),
        candidate.len(),
        "{name}: output sizes differ: {} vs {}",
        reference.len(),
        candidate.len()
    );
    for (i, (expected, actual)) in reference.iter().zip(candidate.iter()).enumerate() {
        let diff = (expected - actual).abs();
        assert!(
            diff < tolerance,
            "{name}: mismatch at index {i}: expected {expected}, got {actual} (diff {diff})"
        );
    }
}

/// Compiles the graph, checks the CPU result against a hand-computed
/// expectation, then checks the GPU result against the CPU result.
fn assert_parity(
    model_graph: &ModelGraph,
    theta: &[f32],
    input: &[f32],
    expected: &[f32],
    name: &str,
) {
    let info = model_graph.compile(theta).expect("graph compile failed");
    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let cpu_result = cpu_model.predict(input).expect("CPU prediction failed");
    compare(expected, &cpu_result, EXACT_TOLERANCE, name);

    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
    match gpu_predict(&gpu_model, input) {
        Some(gpu_result) => compare(&cpu_result, &gpu_result, EXACT_TOLERANCE, name),
        None => eprintln!("{name}: skipping GPU half - no adapter available"),
    }
}

#[test]
fn copy_masked_parity() {
    let graph = Graph::new();
    let x = graph.input(4, None);
    let gathered = graph.gather(&x, vec![3, 1, 1, 0, 2]);
    let model_graph = graph.model(vec![&x], &gathered);

    assert_parity(
        &model_graph,
        &[],
        &[10.0, 20.0, 30.0, 40.0],
        &[40.0, 20.0, 20.0, 10.0, 30.0],
        "copy_masked",
    );
}

#[test]
fn clip_elementwise_parity_min_only() {
    let graph = Graph::new();
    let x = graph.input(4, None);
    let clipped = graph.clip(&x, Some(Constant::Scalar(-1.0)), None);
    let model_graph = graph.model(vec![&x], &clipped);

    assert_parity(
        &model_graph,
        &[],
        &[-2.0, -0.5, 0.5, 2.0],
        &[-1.0, -0.5, 0.5, 2.0],
        "clip_min_only",
    );
}

#[test]
fn clip_elementwise_parity_max_only() {
    let graph = Graph::new();
    let x = graph.input(4, None);
    let clipped = graph.clip(
        &x,
        None,
        Some(Constant::PerElement(vec![0.0, 0.25, 0.5, 1.0])),
    );
    let model_graph = graph.model(vec![&x], &clipped);

    assert_parity(
        &model_graph,
        &[],
        &[-2.0, -0.5, 0.5, 2.0],
        &[-2.0, -0.5, 0.5, 1.0],
        "clip_max_only",
    );
}

#[test]
fn clip_elementwise_parity_both_bounds() {
    let graph = Graph::new();
    let x = graph.input(4, None);
    let clipped = graph.clip(
        &x,
        Some(Constant::Scalar(-1.0)),
        Some(Constant::Scalar(1.0)),
    );
    let model_graph = graph.model(vec![&x], &clipped);

    assert_parity(
        &model_graph,
        &[],
        &[-2.0, -0.5, 0.5, 2.0],
        &[-1.0, -0.5, 0.5, 1.0],
        "clip_both_bounds",
    );
}

#[test]
fn elem_wise_buffers_add_parity() {
    let graph = Graph::new();
    let a = graph.input(3, None);
    let b = graph.input(3, None);
    let c = graph.input(3, None);
    let sum = graph.add(&[&a, &b, &c]);
    let model_graph = graph.model(vec![&a, &b, &c], &sum);

    assert_parity(
        &model_graph,
        &[],
        &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        &[111.0, 222.0, 333.0],
        "elem_wise_buffers_add",
    );
}

#[test]
fn elem_wise_buffers_mul_parity() {
    let graph = Graph::new();
    let a = graph.input(3, None);
    let b = graph.input(3, None);
    let c = graph.input(3, None);
    let product = graph.mul(&[&a, &b, &c]);
    let model_graph = graph.model(vec![&a, &b, &c], &product);

    assert_parity(
        &model_graph,
        &[],
        &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        &[1000.0, 8000.0, 27000.0],
        "elem_wise_buffers_mul",
    );
}

#[test]
fn multiply_buffer_heads_parity() {
    let graph = Graph::new();
    let data = graph.input(6, None);
    let heads = graph.input(2, None);
    let scaled = graph.mul_heads(&data, &heads);
    let model_graph = graph.model(vec![&data, &heads], &scaled);

    assert_parity(
        &model_graph,
        &[],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 10.0],
        &[2.0, 4.0, 6.0, 40.0, 50.0, 60.0],
        "multiply_buffer_heads",
    );
}

#[test]
fn add_buffer_heads_parity() {
    let graph = Graph::new();
    let data = graph.input(6, None);
    let heads = graph.input(2, None);
    let shifted = graph.add_heads(&data, &heads);
    let model_graph = graph.model(vec![&data, &heads], &shifted);

    assert_parity(
        &model_graph,
        &[],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 10.0],
        &[3.0, 4.0, 5.0, 14.0, 15.0, 16.0],
        "add_buffer_heads",
    );
}

#[test]
fn reduce_sum_parity() {
    let graph = Graph::new();
    let x = graph.input(5, None);
    let total = graph.reduce_sum(&x);
    let model_graph = graph.model(vec![&x], &total);

    assert_parity(
        &model_graph,
        &[],
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[15.0],
        "reduce_sum",
    );
}

/// Builds the mixed pipeline used for the end-to-end checks: every new
/// opcode plus DOT/COPY in one model.
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
    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");

    let input = [0.5, -1.5, 2.5, -3.5, 2.0, -0.5];
    let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");
    assert_eq!(cpu_result.len(), 2);
    assert!(cpu_result.iter().all(|value| value.is_finite()));

    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
    match gpu_predict(&gpu_model, &input) {
        Some(gpu_result) => compare(&cpu_result, &gpu_result, DOT_TOLERANCE, "mixed_pipeline"),
        None => eprintln!("mixed_pipeline: skipping GPU half - no adapter available"),
    }
}

#[test]
fn payload_params_do_not_disturb_weights_region() {
    // Zero-repack invariant: pointer payloads land in the params region, so
    // the weights region must stay byte-identical to canonical flat theta.
    let (model_graph, theta) = mixed_model_graph();
    let info = model_graph.compile(&theta).expect("graph compile failed");
    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");

    assert_eq!(gpu_model.theta_len(), theta.len());
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
