//! GPU vs CPU bulk inference benchmark
//!
//! Compares performance of:
//! - CPU inference (single-threaded and parallel)
//! - GPU bulk inference
//!
//! Run with: cargo run --release --bin gpu_benchmark

use instmodel_inference::activation::Activation;
use instmodel_inference::gpu::{GpuModel, get_instmodel_wgsl};
use instmodel_inference::instruction_model::InstructionModel;
use instmodel_inference::instruction_model_info::{
    DotInstructionInfo, InstructionInfo, InstructionModelInfo,
};
use instmodel_inference::parallel_predict::PredictConfig;
use pollster::FutureExt;
use std::time::Instant;
use wgpu::util::DeviceExt;

const TOTAL_INPUTS: usize = 10_000_000;
const FEATURE_SIZE: usize = 40;
const OUTPUT_SIZE: usize = 2;
const WORKGROUP_SIZE: u32 = 256;
const INPUTS_PER_THREAD: u32 = 16; // Each GPU thread processes 16 inputs

// We'll request max buffer size from GPU and calculate batch size dynamically
const DEFAULT_GPU_BATCH_SIZE: usize = 10_000_000; // Will be adjusted based on actual limits

/// Shader template for bulk inference
const BULK_SHADER_TEMPLATE: &str = include_str!("bulk_inference.wgsl");

const HIDDEN_SIZE: usize = 300;

fn create_benchmark_model() -> InstructionModelInfo {
    // MLP: 40 -> 300 (relu) -> 32 (relu) -> 2 (sigmoid)
    // This is a bigger model to make GPU advantage more noticeable
    InstructionModelInfo {
        features: None,
        feature_size: Some(FEATURE_SIZE),
        computation_buffer_sizes: vec![FEATURE_SIZE, HIDDEN_SIZE, 32, OUTPUT_SIZE],
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
            // Layer 1: 40 -> 300
            (0..HIDDEN_SIZE)
                .map(|i| {
                    (0..FEATURE_SIZE)
                        .map(|j| ((i * FEATURE_SIZE + j) as f32 * 0.001) - 0.2)
                        .collect()
                })
                .collect(),
            // Layer 2: 300 -> 32
            (0..32)
                .map(|i| {
                    (0..HIDDEN_SIZE)
                        .map(|j| ((i * HIDDEN_SIZE + j) as f32 * 0.0005) - 0.15)
                        .collect()
                })
                .collect(),
            // Layer 3: 32 -> 2
            (0..OUTPUT_SIZE)
                .map(|i| {
                    (0..32)
                        .map(|j| ((i * 32 + j) as f32 * 0.02) - 0.3)
                        .collect()
                })
                .collect(),
        ],
        bias: vec![
            vec![0.1; HIDDEN_SIZE],
            vec![0.05; 32],
            vec![0.0; OUTPUT_SIZE],
        ],
        parameters: None,
        maps: None,
        validation_data: None,
    }
}

fn generate_random_inputs(count: usize, feature_size: usize) -> Vec<f32> {
    // Simple pseudo-random generation for reproducibility
    let mut data = Vec::with_capacity(count * feature_size);
    let mut seed: u64 = 12345;

    for _ in 0..(count * feature_size) {
        // LCG random number generator
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        data.push(val);
    }

    data
}

fn benchmark_cpu_parallel(
    model: &InstructionModel,
    inputs: &[f32],
    num_threads: usize,
) -> (Vec<f32>, f64) {
    let config = PredictConfig::new().with_threads(num_threads);

    let start = Instant::now();
    let result = model
        .predict_parallel(inputs, config)
        .expect("CPU parallel prediction failed");
    let elapsed = start.elapsed().as_secs_f64();

    (result.as_slice().to_vec(), elapsed)
}

fn create_gpu_shader(compute_buffer_size: u32) -> String {
    let instmodel_wgsl = get_instmodel_wgsl(compute_buffer_size);
    let compute_buffer_decl = format!("var compute_buffer: array<f32, {}>;", compute_buffer_size);

    BULK_SHADER_TEMPLATE
        .replace("// INSTMODEL_WGSL_PLACEHOLDER", &instmodel_wgsl)
        .replace("// COMPUTE_BUFFER_DECLARATION", &compute_buffer_decl)
}

async fn benchmark_gpu(gpu_model: &GpuModel, inputs: &[f32]) -> (Vec<f32>, f64, f64) {
    // Initialize wgpu
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("Failed to find adapter");

    let adapter_info = adapter.get_info();
    println!("  GPU: {} ({:?})", adapter_info.name, adapter_info.backend);

    // Request higher buffer size limits
    let limits = adapter.limits();
    let required_limits = wgpu::Limits {
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_buffer_size: limits.max_buffer_size,
        ..Default::default()
    };

    println!(
        "  Max buffer size: {} MB",
        limits.max_storage_buffer_binding_size / 1_000_000
    );

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_limits,
                ..Default::default()
            },
            None,
        )
        .await
        .expect("Failed to create device");

    // Calculate batch size based on actual buffer limits
    let input_bytes_per_sample = FEATURE_SIZE * std::mem::size_of::<f32>();
    let max_inputs_per_batch =
        (limits.max_storage_buffer_binding_size as usize) / input_bytes_per_sample;
    let gpu_batch_size = std::cmp::min(max_inputs_per_batch, DEFAULT_GPU_BATCH_SIZE);

    // Create shader
    let shader_source = create_gpu_shader(gpu_model.compute_buffer_size() as u32);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Bulk Inference Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
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
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
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

    // Model buffer (shared across batches)
    let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Model Buffer"),
        contents: gpu_model.as_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Process in batches to stay within buffer size limits
    let num_batches = TOTAL_INPUTS.div_ceil(gpu_batch_size);
    println!(
        "  Processing in {} batch(es) of up to {} inputs each",
        num_batches, gpu_batch_size
    );

    let mut all_outputs = vec![0.0f32; TOTAL_INPUTS * OUTPUT_SIZE];
    let mut total_gpu_time = 0.0;
    let mut total_read_time = 0.0;

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * gpu_batch_size;
        let batch_end = std::cmp::min(batch_start + gpu_batch_size, TOTAL_INPUTS);
        let batch_size = batch_end - batch_start;

        // Calculate dispatch parameters for this batch
        let total_threads = (batch_size as u32).div_ceil(INPUTS_PER_THREAD);
        let num_workgroups = total_threads.div_ceil(WORKGROUP_SIZE);

        if batch_idx == 0 {
            println!(
                "  Dispatch per batch: {} workgroups x {} threads/workgroup",
                num_workgroups, WORKGROUP_SIZE
            );
            println!("  Each thread processes {} inputs", INPUTS_PER_THREAD);
        }

        // Create batch-specific buffers
        let input_start = batch_start * FEATURE_SIZE;
        let input_end = batch_end * FEATURE_SIZE;
        let batch_inputs = &inputs[input_start..input_end];

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(batch_inputs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer_size = (batch_size * OUTPUT_SIZE * std::mem::size_of::<f32>()) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Params uniform buffer
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct BenchmarkParams {
            total_inputs: u32,
            feature_size: u32,
            output_size: u32,
            inputs_per_thread: u32,
        }

        let params = BenchmarkParams {
            total_inputs: batch_size as u32,
            feature_size: FEATURE_SIZE as u32,
            output_size: OUTPUT_SIZE as u32,
            inputs_per_thread: INPUTS_PER_THREAD,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Warmup on first batch only
        if batch_idx == 0 {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
        }

        // Timed run
        let start = Instant::now();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        total_gpu_time += start.elapsed().as_secs_f64();

        // Read results
        let start_read = Instant::now();
        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let batch_outputs: &[f32] = bytemuck::cast_slice(&data);

        let output_start = batch_start * OUTPUT_SIZE;
        all_outputs[output_start..output_start + batch_outputs.len()]
            .copy_from_slice(batch_outputs);

        drop(data);
        staging_buffer.unmap();

        total_read_time += start_read.elapsed().as_secs_f64();
    }

    (all_outputs, total_gpu_time, total_read_time)
}

fn verify_results(cpu_outputs: &[f32], gpu_outputs: &[f32], tolerance: f32) -> (bool, f32) {
    if cpu_outputs.len() != gpu_outputs.len() {
        return (false, f32::MAX);
    }

    let mut max_diff: f32 = 0.0;
    for (cpu, gpu) in cpu_outputs.iter().zip(gpu_outputs.iter()) {
        let diff = (cpu - gpu).abs();
        max_diff = max_diff.max(diff);
    }

    (max_diff < tolerance, max_diff)
}

fn main() {
    println!("=== GPU vs CPU Bulk Inference Benchmark ===\n");
    println!("Configuration:");
    println!("  Total inputs: {}", TOTAL_INPUTS);
    println!("  Feature size: {}", FEATURE_SIZE);
    println!("  Hidden size:  {}", HIDDEN_SIZE);
    println!("  Output size:  {}", OUTPUT_SIZE);
    println!(
        "  Model: MLP {} -> {} (relu) -> 32 (relu) -> {} (sigmoid)\n",
        FEATURE_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
    );

    // Create model
    println!("Creating model...");
    let info = create_benchmark_model();
    let cpu_model = InstructionModel::new(info.clone()).expect("Failed to create CPU model");
    let gpu_model = GpuModel::from_info(&info).expect("Failed to create GPU model");
    println!("  Model data size: {} bytes\n", gpu_model.as_bytes().len());

    // Generate inputs
    println!("Generating {} random inputs...", TOTAL_INPUTS);
    let inputs = generate_random_inputs(TOTAL_INPUTS, FEATURE_SIZE);
    println!(
        "  Input data size: {:.2} MB\n",
        (inputs.len() * 4) as f64 / 1_000_000.0
    );

    // Get number of CPU cores
    let num_cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    // CPU single-threaded benchmark (using predict_parallel with 1 thread)
    println!("Running CPU single-threaded benchmark...");
    let (cpu_single_outputs, cpu_single_time) = benchmark_cpu_parallel(&cpu_model, &inputs, 1);
    let cpu_single_throughput = TOTAL_INPUTS as f64 / cpu_single_time;
    println!("  Time: {:.3} s", cpu_single_time);
    println!(
        "  Throughput: {:.0} inferences/sec\n",
        cpu_single_throughput
    );

    // CPU Parallel benchmark
    println!("Running CPU parallel benchmark ({} threads)...", num_cores);
    let (cpu_par_outputs, cpu_par_time) = benchmark_cpu_parallel(&cpu_model, &inputs, num_cores);
    let cpu_par_throughput = TOTAL_INPUTS as f64 / cpu_par_time;
    println!("  Time: {:.3} s", cpu_par_time);
    println!("  Throughput: {:.0} inferences/sec\n", cpu_par_throughput);

    // Verify CPU parallel matches CPU single-threaded
    let (cpu_match, cpu_diff) = verify_results(&cpu_single_outputs, &cpu_par_outputs, 1e-6);
    println!(
        "  CPU parallel matches single-threaded: {} (max diff: {:.2e})\n",
        if cpu_match { "YES" } else { "NO" },
        cpu_diff
    );

    // GPU benchmark
    println!("Running GPU benchmark...");
    let (gpu_outputs, gpu_time, read_time) = benchmark_gpu(&gpu_model, &inputs).block_on();
    let gpu_throughput = TOTAL_INPUTS as f64 / gpu_time;
    let gpu_total_time = gpu_time + read_time;
    let gpu_total_throughput = TOTAL_INPUTS as f64 / gpu_total_time;
    println!("  Compute time: {:.3} s", gpu_time);
    println!("  Read time:    {:.3} s", read_time);
    println!("  Total time:   {:.3} s", gpu_total_time);
    println!(
        "  Throughput (compute only): {:.0} inferences/sec",
        gpu_throughput
    );
    println!(
        "  Throughput (with readback): {:.0} inferences/sec\n",
        gpu_total_throughput
    );

    // Verify GPU matches CPU (use 1e-4 tolerance for GPU floating point differences)
    let (gpu_match, gpu_diff) = verify_results(&cpu_single_outputs, &gpu_outputs, 1e-4);
    println!(
        "GPU matches CPU: {} (max diff: {:.2e})\n",
        if gpu_match { "YES" } else { "NO" },
        gpu_diff
    );

    // Summary
    println!("=== Summary ===");
    println!(
        "CPU single-thread:  {:>10.0} inf/s (baseline)",
        cpu_single_throughput
    );
    println!(
        "CPU parallel ({}t): {:>10.0} inf/s ({:.1}x speedup)",
        num_cores,
        cpu_par_throughput,
        cpu_par_throughput / cpu_single_throughput
    );
    println!(
        "GPU (compute only): {:>10.0} inf/s ({:.1}x vs single, {:.1}x vs parallel)",
        gpu_throughput,
        gpu_throughput / cpu_single_throughput,
        gpu_throughput / cpu_par_throughput
    );
    println!(
        "GPU (with readback):{:>10.0} inf/s ({:.1}x vs single, {:.1}x vs parallel)",
        gpu_total_throughput,
        gpu_total_throughput / cpu_single_throughput,
        gpu_total_throughput / cpu_par_throughput
    );

    println!("\n=== Note ===");
    println!("This benchmark measures BULK inference with CPU<->GPU data transfer.");
    println!("The GPU library is designed for EMBEDDED inference within GPU kernels");
    println!("(e.g., RL simulations) where model data stays on GPU and no transfers occur.");
    println!("In embedded scenarios, GPU advantage would be significantly higher.");
}
