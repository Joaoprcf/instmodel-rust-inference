//! GPU inference tests comparing with CPU results.

use instmodel_inference::activation::Activation;
use instmodel_inference::gpu::{GpuModel, get_instmodel_wgsl};
use instmodel_inference::instruction_model::InstructionModel;
use instmodel_inference::instruction_model_info::{
    ActivationInstructionInfo, CopyInstructionInfo, DotInstructionInfo, ElemWiseAddInstructionInfo,
    ElemWiseMulInstructionInfo, InstructionInfo, InstructionModelInfo,
};
use pollster::FutureExt;
use wgpu::util::DeviceExt;

const TOLERANCE: f32 = 1e-6;

/// Test shader template - loaded from external file for IDE highlighting
const TEST_SHADER_TEMPLATE: &str = include_str!("shaders/test_shader.wgsl");

/// Create a test shader that calls instmodel functions.
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

async fn run_gpu_inference(gpu_model: &GpuModel, input: &[f32]) -> Vec<f32> {
    // Initialize wgpu
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to find adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .expect("Failed to create device");

    // Create shader
    let shader_source = create_test_shader(gpu_model.compute_buffer_size() as u32);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create buffers
    let model_data = gpu_model.as_f32_slice();
    let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Model Buffer"),
        contents: bytemuck::cast_slice(model_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_size = gpu_model.output_size();
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: (output_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (output_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create bind group layout and bind group
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

    // Create pipeline
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

    // Execute
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &staging_buffer,
        0,
        (output_size * std::mem::size_of::<f32>()) as u64,
    );
    queue.submit(Some(encoder.finish()));

    // Read results
    let slice = staging_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    result
}

fn compare_results(cpu_result: &[f32], gpu_result: &[f32], test_name: &str) {
    assert_eq!(
        cpu_result.len(),
        gpu_result.len(),
        "{}: Output sizes differ: CPU={}, GPU={}",
        test_name,
        cpu_result.len(),
        gpu_result.len()
    );

    for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
        let diff = (cpu_val - gpu_val).abs();
        assert!(
            diff < TOLERANCE,
            "{}: Mismatch at index {}: CPU={}, GPU={}, diff={}",
            test_name,
            i,
            cpu_val,
            gpu_val,
            diff
        );
    }
}

#[test]
fn test_simple_dot_product() {
    let info = InstructionModelInfo {
        features: Some(vec!["f1".to_string(), "f2".to_string()]),
        feature_size: Some(2),
        computation_buffer_sizes: vec![2, 1],
        instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
            input: 0,
            output: 1,
            weights: 0,
            activation: None,
        })],
        weights: vec![vec![vec![2.0, 0.5]]],
        bias: vec![vec![0.25]],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    // CPU inference
    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let input = vec![1.0, -1.0];
    let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");

    // GPU inference
    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
    let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

    // Expected: 2.0 * 1.0 + 0.5 * (-1.0) + 0.25 = 1.75
    compare_results(&cpu_result, &gpu_result, "simple_dot_product");
    assert!((cpu_result[0] - 1.75).abs() < TOLERANCE);
}

#[test]
fn test_dot_with_relu() {
    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(3),
        computation_buffer_sizes: vec![3, 2],
        instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
            input: 0,
            output: 1,
            weights: 0,
            activation: Some(Activation::Relu),
        })],
        weights: vec![vec![vec![1.0, -1.0, 0.5], vec![-2.0, 1.0, 0.0]]],
        bias: vec![vec![0.0, 0.0]],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let input = vec![1.0, 2.0, 3.0];
    let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");

    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
    let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

    compare_results(&cpu_result, &gpu_result, "dot_with_relu");

    // First output: 1*1 + (-1)*2 + 0.5*3 = 1 - 2 + 1.5 = 0.5, relu(0.5) = 0.5
    // Second output: (-2)*1 + 1*2 + 0*3 = -2 + 2 = 0, relu(0) = 0
    assert!((cpu_result[0] - 0.5).abs() < TOLERANCE);
    assert!((cpu_result[1] - 0.0).abs() < TOLERANCE);
}

#[test]
fn test_dot_with_sigmoid() {
    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(2),
        computation_buffer_sizes: vec![2, 1],
        instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
            input: 0,
            output: 1,
            weights: 0,
            activation: Some(Activation::Sigmoid),
        })],
        weights: vec![vec![vec![1.0, 1.0]]],
        bias: vec![vec![0.0]],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let input = vec![0.0, 0.0];
    let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");

    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
    let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

    compare_results(&cpu_result, &gpu_result, "dot_with_sigmoid");

    // sigmoid(0) = 0.5
    assert!((cpu_result[0] - 0.5).abs() < TOLERANCE);
}

#[test]
fn test_dot_with_softmax() {
    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(2),
        computation_buffer_sizes: vec![2, 3],
        instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
            input: 0,
            output: 1,
            weights: 0,
            activation: Some(Activation::Softmax),
        })],
        weights: vec![vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]]],
        bias: vec![vec![0.0, 0.0, 0.0]],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let input = vec![1.0, 2.0];
    let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");

    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
    let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

    compare_results(&cpu_result, &gpu_result, "dot_with_softmax");

    // Softmax outputs should sum to 1
    let sum: f32 = cpu_result.iter().sum();
    assert!((sum - 1.0).abs() < TOLERANCE);
}

#[test]
fn test_standalone_activation_relu() {
    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(4),
        computation_buffer_sizes: vec![4],
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

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let input = vec![-1.0, 0.0, 1.0, 2.0];
    let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");

    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
    let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

    compare_results(&cpu_result, &gpu_result, "standalone_activation_relu");

    assert!((cpu_result[0] - 0.0).abs() < TOLERANCE);
    assert!((cpu_result[1] - 0.0).abs() < TOLERANCE);
    assert!((cpu_result[2] - 1.0).abs() < TOLERANCE);
    assert!((cpu_result[3] - 2.0).abs() < TOLERANCE);
}

#[test]
fn test_standalone_activation_tanh() {
    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(3),
        computation_buffer_sizes: vec![3],
        instructions: vec![InstructionInfo::Activation(ActivationInstructionInfo {
            input: 0,
            activation: Activation::Tanh,
        })],
        weights: vec![],
        bias: vec![],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let input = vec![-1.0, 0.0, 1.0];
    let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");

    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
    let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

    compare_results(&cpu_result, &gpu_result, "standalone_activation_tanh");
}

#[test]
fn test_elem_wise_add() {
    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(3),
        computation_buffer_sizes: vec![3],
        instructions: vec![InstructionInfo::ElemWiseAdd(ElemWiseAddInstructionInfo {
            input: 0,
            parameters: 0,
        })],
        weights: vec![],
        bias: vec![],
        parameters: Some(vec![vec![1.0, 2.0, 3.0]]),
        maps: None,
        validation_data: None,
    };

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let input = vec![10.0, 20.0, 30.0];
    let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");

    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
    let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

    compare_results(&cpu_result, &gpu_result, "elem_wise_add");

    assert!((cpu_result[0] - 11.0).abs() < TOLERANCE);
    assert!((cpu_result[1] - 22.0).abs() < TOLERANCE);
    assert!((cpu_result[2] - 33.0).abs() < TOLERANCE);
}

#[test]
fn test_elem_wise_mul() {
    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(3),
        computation_buffer_sizes: vec![3],
        instructions: vec![InstructionInfo::ElemWiseMul(ElemWiseMulInstructionInfo {
            input: 0,
            parameters: 0,
        })],
        weights: vec![],
        bias: vec![],
        parameters: Some(vec![vec![2.0, 0.5, 0.0]]),
        maps: None,
        validation_data: None,
    };

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let input = vec![10.0, 20.0, 30.0];
    let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");

    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
    let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

    compare_results(&cpu_result, &gpu_result, "elem_wise_mul");

    assert!((cpu_result[0] - 20.0).abs() < TOLERANCE);
    assert!((cpu_result[1] - 10.0).abs() < TOLERANCE);
    assert!((cpu_result[2] - 0.0).abs() < TOLERANCE);
}

#[test]
fn test_copy_instruction() {
    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(2),
        computation_buffer_sizes: vec![2, 4],
        instructions: vec![InstructionInfo::Copy(CopyInstructionInfo {
            input: 0,
            output: 1,
            internal_index: 1,
        })],
        weights: vec![],
        bias: vec![],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let input = vec![5.0, 7.0];
    let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");

    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
    let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

    compare_results(&cpu_result, &gpu_result, "copy_instruction");

    // Output buffer [0, 5.0, 7.0, 0] (copied at internal_index=1)
    assert!((cpu_result[0] - 0.0).abs() < TOLERANCE);
    assert!((cpu_result[1] - 5.0).abs() < TOLERANCE);
    assert!((cpu_result[2] - 7.0).abs() < TOLERANCE);
    assert!((cpu_result[3] - 0.0).abs() < TOLERANCE);
}

#[test]
fn test_multi_layer_network() {
    // 3-layer MLP: 4 -> 8 (relu) -> 4 (relu) -> 2 (sigmoid)
    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(4),
        computation_buffer_sizes: vec![4, 8, 4, 2],
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
                activation: Some(Activation::Relu),
            }),
            InstructionInfo::Dot(DotInstructionInfo {
                input: 2,
                output: 3,
                weights: 2,
                activation: Some(Activation::Sigmoid),
            }),
        ],
        weights: vec![
            // Layer 1: 4 -> 8
            vec![
                vec![0.1, 0.2, 0.3, 0.4],
                vec![0.5, 0.6, 0.7, 0.8],
                vec![0.1, -0.2, 0.3, -0.4],
                vec![-0.5, 0.6, -0.7, 0.8],
                vec![0.2, 0.2, 0.2, 0.2],
                vec![0.3, 0.3, 0.3, 0.3],
                vec![0.4, 0.4, 0.4, 0.4],
                vec![0.5, 0.5, 0.5, 0.5],
            ],
            // Layer 2: 8 -> 4
            vec![
                vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                vec![0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                vec![0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                vec![0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
            ],
            // Layer 3: 4 -> 2
            vec![vec![0.5, 0.5, 0.5, 0.5], vec![0.6, 0.6, 0.6, 0.6]],
        ],
        bias: vec![
            vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            vec![0.1, 0.1, 0.1, 0.1],
            vec![0.0, 0.0],
        ],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");

    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
    let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

    compare_results(&cpu_result, &gpu_result, "multi_layer_network");

    // Sigmoid outputs should be in (0, 1)
    assert!(cpu_result[0] > 0.0 && cpu_result[0] < 1.0);
    assert!(cpu_result[1] > 0.0 && cpu_result[1] < 1.0);
}

#[test]
fn test_all_activations() {
    let activations = [
        Activation::Relu,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Sqrt,
        Activation::Log,
        Activation::Log10,
        Activation::Inverse,
    ];

    for activation in activations {
        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(4),
            computation_buffer_sizes: vec![4],
            instructions: vec![InstructionInfo::Activation(ActivationInstructionInfo {
                input: 0,
                activation,
            })],
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
        // Use positive values for sqrt/log
        let input = vec![0.5, 1.0, 2.0, 4.0];
        let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");

        let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
        let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

        compare_results(
            &cpu_result,
            &gpu_result,
            &format!("activation_{:?}", activation),
        );
    }
}

#[test]
fn test_standalone_softmax() {
    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(4),
        computation_buffer_sizes: vec![4],
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

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");

    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");
    let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

    compare_results(&cpu_result, &gpu_result, "standalone_softmax");

    // Softmax outputs should sum to 1
    let cpu_sum: f32 = cpu_result.iter().sum();
    let gpu_sum: f32 = gpu_result.iter().sum();
    assert!((cpu_sum - 1.0).abs() < TOLERANCE);
    assert!((gpu_sum - 1.0).abs() < TOLERANCE);
}

#[test]
fn test_gpu_model_full_size() {
    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(2),
        computation_buffer_sizes: vec![2, 1],
        instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
            input: 0,
            output: 1,
            weights: 0,
            activation: None,
        })],
        weights: vec![vec![vec![1.0, 2.0]]],
        bias: vec![vec![0.5]],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");

    // Verify full_model_size is stored in header correctly
    let data = gpu_model.as_f32_slice();
    let full_model_size_from_header = data[12].to_bits() as usize;
    assert_eq!(full_model_size_from_header, gpu_model.full_size());
}

// =============================================================================
// CPU/GPU Parity Tests - Same InstructionModelInfo, Multiple Inputs
// =============================================================================

/// Test CPU/GPU parity with multiple different inputs on the same model
#[test]
fn test_cpu_gpu_parity_multiple_inputs() {
    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(4),
        computation_buffer_sizes: vec![4, 8, 2],
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
                activation: Some(Activation::Sigmoid),
            }),
        ],
        weights: vec![
            vec![
                vec![0.1, -0.2, 0.3, -0.4],
                vec![-0.5, 0.6, -0.7, 0.8],
                vec![0.9, -0.1, 0.2, -0.3],
                vec![-0.4, 0.5, -0.6, 0.7],
                vec![0.8, -0.9, 0.1, -0.2],
                vec![-0.3, 0.4, -0.5, 0.6],
                vec![0.7, -0.8, 0.9, -0.1],
                vec![-0.2, 0.3, -0.4, 0.5],
            ],
            vec![
                vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                vec![-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
            ],
        ],
        bias: vec![vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], vec![0.0, 0.0]],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    // Create both models from the SAME InstructionModelInfo
    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");

    // Test with multiple different inputs
    let test_inputs = [
        vec![0.0, 0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![-1.0, -1.0, -1.0, -1.0],
        vec![1.0, -1.0, 1.0, -1.0],
        vec![0.5, -0.5, 0.25, -0.25],
        vec![100.0, -100.0, 50.0, -50.0],
        vec![0.001, 0.002, 0.003, 0.004],
    ];

    for (i, input) in test_inputs.iter().enumerate() {
        let cpu_result = cpu_model.predict(input).expect("CPU prediction failed");
        let gpu_result = run_gpu_inference(&gpu_model, input).block_on();
        compare_results(
            &cpu_result,
            &gpu_result,
            &format!("cpu_gpu_parity_input_{}", i),
        );
    }
}

/// Test CPU/GPU parity with a deep network (5 layers)
#[test]
fn test_cpu_gpu_parity_deep_network() {
    // 5-layer network: 8 -> 16 -> 16 -> 8 -> 4 -> 2
    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(8),
        computation_buffer_sizes: vec![8, 16, 16, 8, 4, 2],
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
                activation: Some(Activation::Tanh),
            }),
            InstructionInfo::Dot(DotInstructionInfo {
                input: 2,
                output: 3,
                weights: 2,
                activation: Some(Activation::Relu),
            }),
            InstructionInfo::Dot(DotInstructionInfo {
                input: 3,
                output: 4,
                weights: 3,
                activation: Some(Activation::Sigmoid),
            }),
            InstructionInfo::Dot(DotInstructionInfo {
                input: 4,
                output: 5,
                weights: 4,
                activation: Some(Activation::Softmax),
            }),
        ],
        weights: vec![
            // Layer 1: 8 -> 16
            (0..16)
                .map(|i| (0..8).map(|j| ((i * 8 + j) as f32 * 0.01) - 0.5).collect())
                .collect(),
            // Layer 2: 16 -> 16
            (0..16)
                .map(|i| {
                    (0..16)
                        .map(|j| ((i * 16 + j) as f32 * 0.005) - 0.4)
                        .collect()
                })
                .collect(),
            // Layer 3: 16 -> 8
            (0..8)
                .map(|i| {
                    (0..16)
                        .map(|j| ((i * 16 + j) as f32 * 0.008) - 0.3)
                        .collect()
                })
                .collect(),
            // Layer 4: 8 -> 4
            (0..4)
                .map(|i| (0..8).map(|j| ((i * 8 + j) as f32 * 0.02) - 0.2).collect())
                .collect(),
            // Layer 5: 4 -> 2
            (0..2)
                .map(|i| (0..4).map(|j| ((i * 4 + j) as f32 * 0.05) - 0.1).collect())
                .collect(),
        ],
        bias: vec![
            vec![0.1; 16],
            vec![0.05; 16],
            vec![0.02; 8],
            vec![0.01; 4],
            vec![0.0; 2],
        ],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");

    let input = vec![0.5, -0.3, 0.8, -0.2, 0.1, 0.9, -0.7, 0.4];
    let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");
    let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

    compare_results(&cpu_result, &gpu_result, "deep_network_parity");

    // Verify softmax output sums to 1
    let sum: f32 = cpu_result.iter().sum();
    assert!((sum - 1.0).abs() < TOLERANCE, "Softmax sum {} != 1.0", sum);
}

/// Test CPU/GPU parity with mixed instruction types
#[test]
fn test_cpu_gpu_parity_mixed_instructions() {
    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(4),
        computation_buffer_sizes: vec![4, 4, 4, 2],
        instructions: vec![
            // Element-wise multiply with parameters
            InstructionInfo::ElemWiseMul(ElemWiseMulInstructionInfo {
                input: 0,
                parameters: 0,
            }),
            // Element-wise add with parameters
            InstructionInfo::ElemWiseAdd(ElemWiseAddInstructionInfo {
                input: 0,
                parameters: 1,
            }),
            // Copy to buffer 1
            InstructionInfo::Copy(CopyInstructionInfo {
                input: 0,
                output: 1,
                internal_index: 0,
            }),
            // Activation on buffer 1
            InstructionInfo::Activation(ActivationInstructionInfo {
                input: 1,
                activation: Activation::Tanh,
            }),
            // Copy to buffer 2
            InstructionInfo::Copy(CopyInstructionInfo {
                input: 1,
                output: 2,
                internal_index: 0,
            }),
            // DOT from buffer 2 to output
            InstructionInfo::Dot(DotInstructionInfo {
                input: 2,
                output: 3,
                weights: 0,
                activation: Some(Activation::Sigmoid),
            }),
        ],
        weights: vec![vec![vec![0.5, 0.5, 0.5, 0.5], vec![-0.5, -0.5, -0.5, -0.5]]],
        bias: vec![vec![0.1, -0.1]],
        parameters: Some(vec![
            vec![2.0, 0.5, 1.5, 0.25], // mul params
            vec![0.1, 0.2, 0.3, 0.4],  // add params
        ]),
        maps: None,
        validation_data: None,
    };

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");

    let input = vec![1.0, 2.0, 3.0, 4.0];
    let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");
    let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

    compare_results(&cpu_result, &gpu_result, "mixed_instructions_parity");
}

/// Test CPU/GPU parity with edge case values
#[test]
fn test_cpu_gpu_parity_edge_cases() {
    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(4),
        computation_buffer_sizes: vec![4, 4],
        instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
            input: 0,
            output: 1,
            weights: 0,
            activation: Some(Activation::Sigmoid),
        })],
        weights: vec![vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ]],
        bias: vec![vec![0.0, 0.0, 0.0, 0.0]],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");

    // Test edge cases for sigmoid
    let edge_inputs = [
        vec![0.0, 0.0, 0.0, 0.0],           // sigmoid(0) = 0.5
        vec![10.0, 10.0, 10.0, 10.0],       // Large positive - sigmoid near 1
        vec![-10.0, -10.0, -10.0, -10.0],   // Large negative - sigmoid near 0
        vec![1e-10, -1e-10, 1e-10, -1e-10], // Very small values
    ];

    for (i, input) in edge_inputs.iter().enumerate() {
        let cpu_result = cpu_model.predict(input).expect("CPU prediction failed");
        let gpu_result = run_gpu_inference(&gpu_model, input).block_on();
        compare_results(&cpu_result, &gpu_result, &format!("edge_case_input_{}", i));
    }
}

/// Test CPU/GPU parity: verify identical models produce identical metadata
#[test]
fn test_cpu_gpu_parity_metadata() {
    let info = InstructionModelInfo {
        features: Some(vec![
            "feature_a".to_string(),
            "feature_b[3]".to_string(),
            "feature_c".to_string(),
        ]),
        feature_size: Some(5), // 1 + 3 + 1
        computation_buffer_sizes: vec![5, 3],
        instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
            input: 0,
            output: 1,
            weights: 0,
            activation: None,
        })],
        weights: vec![vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
        ]],
        bias: vec![vec![0.1, 0.2, 0.3]],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");

    // Verify metadata matches
    assert_eq!(cpu_model.get_feature_size(), gpu_model.feature_size());
    assert_eq!(cpu_model.get_output_size(), gpu_model.output_size());

    // Verify prediction results match
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");
    let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

    compare_results(&cpu_result, &gpu_result, "metadata_parity");
}

/// Test CPU/GPU parity with fan-out, fan-in configuration
/// Input splits into two branches, processed separately, then combined
#[test]
fn test_cpu_gpu_parity_fan_out_fan_in() {
    // Architecture:
    // Input (4) ─┬─> Branch A: DOT (4->3) with Relu ─┐
    //            │                                    ├─> Concatenate (6) -> DOT (6->2) with Sigmoid
    //            └─> Branch B: DOT (4->3) with Tanh ─┘
    //
    // Buffer layout:
    // 0: input (4)
    // 1: branch_a output (3)
    // 2: branch_b output (3)
    // 3: concatenated (6) - branch_a copied to [0..3], branch_b copied to [3..6]
    // 4: final output (2)

    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(4),
        computation_buffer_sizes: vec![4, 3, 3, 6, 2],
        instructions: vec![
            // Fan-out: Input -> Branch A
            InstructionInfo::Dot(DotInstructionInfo {
                input: 0,
                output: 1,
                weights: 0,
                activation: Some(Activation::Relu),
            }),
            // Fan-out: Input -> Branch B
            InstructionInfo::Dot(DotInstructionInfo {
                input: 0,
                output: 2,
                weights: 1,
                activation: Some(Activation::Tanh),
            }),
            // Fan-in: Copy Branch A to concatenation buffer [0..3]
            InstructionInfo::Copy(CopyInstructionInfo {
                input: 1,
                output: 3,
                internal_index: 0,
            }),
            // Fan-in: Copy Branch B to concatenation buffer [3..6]
            InstructionInfo::Copy(CopyInstructionInfo {
                input: 2,
                output: 3,
                internal_index: 3,
            }),
            // Final layer: DOT from concatenated to output
            InstructionInfo::Dot(DotInstructionInfo {
                input: 3,
                output: 4,
                weights: 2,
                activation: Some(Activation::Sigmoid),
            }),
        ],
        weights: vec![
            // Branch A weights: 4 -> 3
            vec![
                vec![0.5, -0.3, 0.2, 0.1],
                vec![-0.4, 0.6, -0.1, 0.3],
                vec![0.3, -0.2, 0.4, -0.5],
            ],
            // Branch B weights: 4 -> 3
            vec![
                vec![-0.2, 0.4, -0.3, 0.5],
                vec![0.1, -0.5, 0.2, -0.4],
                vec![-0.3, 0.1, -0.4, 0.2],
            ],
            // Final layer weights: 6 -> 2
            vec![
                vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                vec![-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
            ],
        ],
        bias: vec![
            vec![0.1, 0.1, 0.1],    // Branch A bias
            vec![-0.1, -0.1, -0.1], // Branch B bias
            vec![0.0, 0.0],         // Final layer bias
        ],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");

    // Test with multiple inputs
    let test_inputs = [
        vec![1.0, 0.5, -0.5, -1.0],
        vec![0.0, 0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![-1.0, 2.0, -3.0, 4.0],
    ];

    for (i, input) in test_inputs.iter().enumerate() {
        let cpu_result = cpu_model.predict(input).expect("CPU prediction failed");
        let gpu_result = run_gpu_inference(&gpu_model, input).block_on();
        compare_results(
            &cpu_result,
            &gpu_result,
            &format!("fan_out_fan_in_input_{}", i),
        );
    }
}

/// Test CPU/GPU parity with residual-style skip connection pattern
/// Input is processed through two parallel paths then combined
#[test]
fn test_cpu_gpu_parity_residual_connection() {
    // Architecture (residual-style with concatenation):
    // Input (4) ─┬─> Copy to buffer 1 (identity path) ─┐
    //            │                                      ├─> Concat (8) -> DOT (8->4) Relu
    //            └─> DOT (4->4) Tanh to buffer 2 ──────┘
    //
    // Buffer layout:
    // 0: input (4)
    // 1: identity copy (4)
    // 2: transformed (4)
    // 3: concatenated (8) - identity + transformed
    // 4: output (4)

    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(4),
        computation_buffer_sizes: vec![4, 4, 4, 8, 4],
        instructions: vec![
            // Identity path: copy input to buffer 1
            InstructionInfo::Copy(CopyInstructionInfo {
                input: 0,
                output: 1,
                internal_index: 0,
            }),
            // Transform path: DOT input -> buffer 2
            InstructionInfo::Dot(DotInstructionInfo {
                input: 0,
                output: 2,
                weights: 0,
                activation: Some(Activation::Tanh),
            }),
            // Concatenate: copy identity to [0..4]
            InstructionInfo::Copy(CopyInstructionInfo {
                input: 1,
                output: 3,
                internal_index: 0,
            }),
            // Concatenate: copy transformed to [4..8]
            InstructionInfo::Copy(CopyInstructionInfo {
                input: 2,
                output: 3,
                internal_index: 4,
            }),
            // Final: combine both paths
            InstructionInfo::Dot(DotInstructionInfo {
                input: 3,
                output: 4,
                weights: 1,
                activation: Some(Activation::Relu),
            }),
        ],
        weights: vec![
            // Transform weights: 4 -> 4
            vec![
                vec![0.9, 0.1, 0.0, 0.0],
                vec![0.1, 0.9, 0.1, 0.0],
                vec![0.0, 0.1, 0.9, 0.1],
                vec![0.0, 0.0, 0.1, 0.9],
            ],
            // Combine weights: 8 -> 4 (learns to add identity + transformed)
            vec![
                vec![0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                vec![0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
                vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
            ],
        ],
        bias: vec![vec![0.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 0.0]],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");

    let test_inputs = [
        vec![1.0, 2.0, 3.0, 4.0],
        vec![-1.0, -2.0, -3.0, -4.0],
        vec![0.5, -0.5, 0.5, -0.5],
    ];

    for (i, input) in test_inputs.iter().enumerate() {
        let cpu_result = cpu_model.predict(input).expect("CPU prediction failed");
        let gpu_result = run_gpu_inference(&gpu_model, input).block_on();
        compare_results(
            &cpu_result,
            &gpu_result,
            &format!("residual_connection_input_{}", i),
        );
    }
}

/// Test CPU/GPU parity with wider fan-out (1 input to 3 branches)
#[test]
fn test_cpu_gpu_parity_wide_fan_out() {
    // Architecture:
    // Input (4) ─┬─> Branch A: DOT (4->2) Relu ──┐
    //            ├─> Branch B: DOT (4->2) Tanh ──┼─> Concat (6) -> DOT (6->3) Softmax
    //            └─> Branch C: DOT (4->2) Sigmoid┘
    //
    // Buffer layout:
    // 0: input (4)
    // 1: branch_a (2)
    // 2: branch_b (2)
    // 3: branch_c (2)
    // 4: concatenated (6)
    // 5: output (3)

    let info = InstructionModelInfo {
        features: None,
        feature_size: Some(4),
        computation_buffer_sizes: vec![4, 2, 2, 2, 6, 3],
        instructions: vec![
            // Branch A
            InstructionInfo::Dot(DotInstructionInfo {
                input: 0,
                output: 1,
                weights: 0,
                activation: Some(Activation::Relu),
            }),
            // Branch B
            InstructionInfo::Dot(DotInstructionInfo {
                input: 0,
                output: 2,
                weights: 1,
                activation: Some(Activation::Tanh),
            }),
            // Branch C
            InstructionInfo::Dot(DotInstructionInfo {
                input: 0,
                output: 3,
                weights: 2,
                activation: Some(Activation::Sigmoid),
            }),
            // Concatenate: copy A to [0..2]
            InstructionInfo::Copy(CopyInstructionInfo {
                input: 1,
                output: 4,
                internal_index: 0,
            }),
            // Concatenate: copy B to [2..4]
            InstructionInfo::Copy(CopyInstructionInfo {
                input: 2,
                output: 4,
                internal_index: 2,
            }),
            // Concatenate: copy C to [4..6]
            InstructionInfo::Copy(CopyInstructionInfo {
                input: 3,
                output: 4,
                internal_index: 4,
            }),
            // Final: concatenated -> output
            InstructionInfo::Dot(DotInstructionInfo {
                input: 4,
                output: 5,
                weights: 3,
                activation: Some(Activation::Softmax),
            }),
        ],
        weights: vec![
            // Branch A: 4 -> 2
            vec![vec![0.5, -0.3, 0.2, 0.1], vec![-0.4, 0.6, -0.1, 0.3]],
            // Branch B: 4 -> 2
            vec![vec![-0.2, 0.4, -0.3, 0.5], vec![0.1, -0.5, 0.2, -0.4]],
            // Branch C: 4 -> 2
            vec![vec![0.3, -0.1, 0.4, -0.2], vec![-0.5, 0.2, -0.3, 0.1]],
            // Final: 6 -> 3
            vec![
                vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                vec![-0.1, -0.2, -0.3, 0.1, 0.2, 0.3],
                vec![0.2, -0.1, 0.1, -0.2, 0.3, -0.3],
            ],
        ],
        bias: vec![
            vec![0.1, -0.1],
            vec![0.05, -0.05],
            vec![-0.1, 0.1],
            vec![0.0, 0.0, 0.0],
        ],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
    let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");

    let test_inputs = [
        vec![1.0, -1.0, 0.5, -0.5],
        vec![0.0, 0.0, 0.0, 0.0],
        vec![2.0, 2.0, 2.0, 2.0],
    ];

    for (i, input) in test_inputs.iter().enumerate() {
        let cpu_result = cpu_model.predict(input).expect("CPU prediction failed");
        let gpu_result = run_gpu_inference(&gpu_model, input).block_on();
        compare_results(
            &cpu_result,
            &gpu_result,
            &format!("wide_fan_out_input_{}", i),
        );

        // Verify softmax sums to 1
        let sum: f32 = cpu_result.iter().sum();
        assert!((sum - 1.0).abs() < TOLERANCE, "Softmax sum {} != 1.0", sum);
    }
}

/// Test CPU/GPU parity with all activation functions in DOT instructions
#[test]
fn test_cpu_gpu_parity_all_dot_activations() {
    let activations = [
        None,
        Some(Activation::Relu),
        Some(Activation::Sigmoid),
        Some(Activation::Tanh),
        Some(Activation::Softmax),
    ];

    for activation in activations {
        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(3),
            computation_buffer_sizes: vec![3, 4],
            instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
                input: 0,
                output: 1,
                weights: 0,
                activation,
            })],
            weights: vec![vec![
                vec![0.5, -0.3, 0.2],
                vec![-0.4, 0.6, -0.1],
                vec![0.3, -0.2, 0.4],
                vec![-0.1, 0.5, -0.3],
            ]],
            bias: vec![vec![0.1, -0.1, 0.05, -0.05]],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let cpu_model = InstructionModel::new(info.clone()).expect("CPU model creation failed");
        let gpu_model = GpuModel::from_info(&info).expect("GPU model creation failed");

        let input = vec![1.0, -0.5, 0.25];
        let cpu_result = cpu_model.predict(&input).expect("CPU prediction failed");
        let gpu_result = run_gpu_inference(&gpu_model, &input).block_on();

        compare_results(
            &cpu_result,
            &gpu_result,
            &format!("dot_activation_{:?}", activation),
        );
    }
}
