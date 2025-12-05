//! Performance benchmark binary comparing framework vs manual neural network inference.
//!
//! This binary measures the performance of the instruction-based neural network framework
//! against manual implementations to understand the overhead and efficiency characteristics.

use instmodel_inference::benchmarks::instructions::dot_product::manual::DotProductTestData;
use instmodel_inference::instruction_model_info::*;
use instmodel_inference::{Activation, InstructionModel};
use log::{error, info};
use std::time::Instant;

/// Manual implementation of the same neural network for comparison
struct ManualNeuralNetwork {
    weights_layer1: Vec<Vec<f32>>,
    bias_layer1: Vec<f32>,
    weights_layer2: Vec<Vec<f32>>,
    bias_layer2: Vec<f32>,
}

impl ManualNeuralNetwork {
    fn new(
        weights_layer1: Vec<Vec<f32>>,
        bias_layer1: Vec<f32>,
        weights_layer2: Vec<Vec<f32>>,
        bias_layer2: Vec<f32>,
    ) -> Self {
        Self {
            weights_layer1,
            bias_layer1,
            weights_layer2,
            bias_layer2,
        }
    }

    /// Manual forward pass: input -> hidden (ReLU) -> output (Sigmoid)
    fn predict(&self, input: &[f32]) -> Vec<f32> {
        // Layer 1: Dense + ReLU
        let mut hidden = vec![0.0f32; self.weights_layer1.len()];
        for (i, weights_row) in self.weights_layer1.iter().enumerate() {
            let mut sum = self.bias_layer1[i];
            for (j, &weight) in weights_row.iter().enumerate() {
                sum += weight * input[j];
            }
            // ReLU activation
            hidden[i] = if sum > 0.0 { sum } else { 0.0 };
        }

        // Layer 2: Dense + Sigmoid
        let mut output = vec![0.0f32; self.weights_layer2.len()];
        for (i, weights_row) in self.weights_layer2.iter().enumerate() {
            let mut sum = self.bias_layer2[i];
            for (j, &weight) in weights_row.iter().enumerate() {
                sum += weight * hidden[j];
            }
            // Sigmoid activation
            output[i] = 1.0 / (1.0 + (-sum).exp());
        }

        output
    }

    /// Manual forward pass with pre-allocated buffers for fair comparison
    fn predict_with_buffers(
        &self,
        input: &[f32],
        hidden_buffer: &mut [f32],
        output_buffer: &mut [f32],
    ) {
        // Layer 1: Dense + ReLU
        for (i, weights_row) in self.weights_layer1.iter().enumerate() {
            let mut sum = self.bias_layer1[i];
            for (j, &weight) in weights_row.iter().enumerate() {
                sum += weight * input[j];
            }
            // ReLU activation
            hidden_buffer[i] = if sum > 0.0 { sum } else { 0.0 };
        }

        // Layer 2: Dense + Sigmoid
        for (i, weights_row) in self.weights_layer2.iter().enumerate() {
            let mut sum = self.bias_layer2[i];
            for (j, &weight) in weights_row.iter().enumerate() {
                sum += weight * hidden_buffer[j];
            }
            // Sigmoid activation
            output_buffer[i] = 1.0 / (1.0 + (-sum).exp());
        }
    }
}

/// Performance measurement structure
#[derive(Debug)]
struct PerformanceResults {
    method: String,
    total_time_ns: u128,
    average_time_ns: u128,
    average_time_ms: f64,
    num_executions: u32,
}

impl PerformanceResults {
    fn new(method: String, total_time_ns: u128, num_executions: u32) -> Self {
        let average_time_ns = total_time_ns / num_executions as u128;
        let average_time_ms = average_time_ns as f64 / 1_000_000.0;

        Self {
            method,
            total_time_ns,
            average_time_ns,
            average_time_ms,
            num_executions,
        }
    }

    fn overhead_ratio(&self, baseline: &PerformanceResults) -> f64 {
        self.average_time_ns as f64 / baseline.average_time_ns as f64
    }

    fn overhead_percentage(&self, baseline: &PerformanceResults) -> f64 {
        (self.overhead_ratio(baseline) - 1.0) * 100.0
    }
}

fn create_test_data() -> DotProductTestData {
    // Create weights for layer 1: 10000 outputs x 2 inputs
    let weights_layer1: Vec<Vec<f32>> = (0..10000)
        .map(|i| vec![0.1 + (i as f32) * 0.0001, 0.2 + (i as f32) * 0.0001])
        .collect();

    // Create bias for layer 1: 10000 values
    let bias_layer1: Vec<f32> = (0..10000).map(|i| 0.01 + (i as f32) * 0.00001).collect();

    // Create weights for layer 2: 10 outputs x 10000 inputs
    let weights_layer2: Vec<Vec<f32>> = (0..10)
        .map(|i| {
            (0..10000)
                .map(|j| 0.001 + (i as f32 + j as f32) * 0.000001)
                .collect()
        })
        .collect();

    // Create bias for layer 2: 10 values
    let bias_layer2: Vec<f32> = (0..10).map(|i| 0.1 + (i as f32) * 0.01).collect();

    // Test inputs
    let inputs = vec![0.5, -0.3];

    (
        weights_layer1,
        bias_layer1,
        weights_layer2,
        bias_layer2,
        inputs,
    )
}

fn create_framework_model(
    weights_layer1: &[Vec<f32>],
    bias_layer1: &[f32],
    weights_layer2: &[Vec<f32>],
    bias_layer2: &[f32],
) -> InstructionModel {
    let computation_buffer_sizes = vec![2, 10000, 10];

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
        features: Some(vec!["input1".to_string(), "input2".to_string()]),
        feature_size: None,
        computation_buffer_sizes,
        instructions,
        weights: vec![weights_layer1.to_vec(), weights_layer2.to_vec()],
        bias: vec![bias_layer1.to_vec(), bias_layer2.to_vec()],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    InstructionModel::new(model_info).expect("Model creation should succeed")
}

fn benchmark_method<F>(name: &str, num_executions: u32, mut benchmark_fn: F) -> PerformanceResults
where
    F: FnMut(),
{
    println!("Benchmarking {} ({} executions)...", name, num_executions);

    // Warm-up
    for _ in 0..5 {
        benchmark_fn();
    }

    let start = Instant::now();
    for i in 0..num_executions {
        benchmark_fn();
        if (i + 1) % (num_executions / 10) == 0 {
            println!("  Progress: {}/{}", i + 1, num_executions);
        }
    }
    let duration = start.elapsed();

    PerformanceResults::new(name.to_string(), duration.as_nanos(), num_executions)
}

fn verify_outputs_match(manual_output: &[f32], framework_output: &[f32]) -> bool {
    const EPSILON: f32 = 1e-6;
    if manual_output.len() != framework_output.len() {
        return false;
    }
    for (manual, framework) in manual_output.iter().zip(framework_output.iter()) {
        if (manual - framework).abs() > EPSILON {
            println!(
                "Output mismatch: manual={}, framework={}, diff={}",
                manual,
                framework,
                (manual - framework).abs()
            );
            return false;
        }
    }
    true
}

fn main() {
    // Initialize logger
    env_logger::init();

    info!("{}", "=".repeat(80));
    info!("Neural Network Performance Benchmark");
    info!("Network Architecture: 2 inputs -> 10000 hidden (ReLU) -> 10 outputs (Sigmoid)");
    info!("{}", "=".repeat(80));

    let num_executions = 1000; // More executions for better statistical accuracy
    let (weights_layer1, bias_layer1, weights_layer2, bias_layer2, inputs) = create_test_data();

    // Create models
    let manual_model = ManualNeuralNetwork::new(
        weights_layer1.clone(),
        bias_layer1.clone(),
        weights_layer2.clone(),
        bias_layer2.clone(),
    );
    let framework_model =
        create_framework_model(&weights_layer1, &bias_layer1, &weights_layer2, &bias_layer2);

    // Verify outputs match
    info!("Verifying output consistency between manual and framework implementations...");
    let manual_result = manual_model.predict(&inputs);
    let framework_result = framework_model
        .predict(&inputs)
        .expect("Framework prediction failed");

    if verify_outputs_match(&manual_result, &framework_result) {
        info!("✅ Outputs match - implementations are consistent");
        info!(
            "   Sample values: {:?}",
            &manual_result[..5.min(manual_result.len())]
        );
    } else {
        error!("❌ Outputs do not match - there may be an implementation bug");
        return;
    }

    info!("{}", "=".repeat(80));
    info!("Performance Benchmarks");
    info!("{}", "=".repeat(80));

    // 1. Manual implementation (baseline)
    let manual_results =
        benchmark_method("Manual Implementation (Baseline)", num_executions, || {
            let _result = manual_model.predict(&inputs);
        });

    // 2. Manual implementation with pre-allocated buffers
    let mut hidden_buffer = vec![0.0f32; 10000];
    let mut output_buffer = vec![0.0f32; 10];
    let manual_buffered_results = benchmark_method(
        "Manual Implementation (Pre-allocated buffers)",
        num_executions,
        || {
            manual_model.predict_with_buffers(&inputs, &mut hidden_buffer, &mut output_buffer);
        },
    );

    // 3. Framework with buffer allocation
    let framework_alloc_results = benchmark_method(
        "Framework Implementation (Buffer allocation)",
        num_executions,
        || {
            let _result = framework_model.predict(&inputs).expect("Prediction failed");
        },
    );

    // 4. Framework with pre-allocated buffer
    let required_memory = framework_model.required_memory();
    let mut computation_buffer = vec![0.0f32; required_memory];
    let framework_buffer_results = benchmark_method(
        "Framework Implementation (Pre-allocated buffer)",
        num_executions,
        || {
            // Copy input to buffer
            for (i, &value) in inputs.iter().enumerate() {
                computation_buffer[i] = value;
            }
            framework_model
                .predict_with_buffer(&mut computation_buffer)
                .expect("Prediction failed");
        },
    );

    // Print detailed results
    println!("\n{}", "=".repeat(80));
    println!("Detailed Results");
    println!("{}", "=".repeat(80));

    let results = vec![
        &manual_results,
        &manual_buffered_results,
        &framework_alloc_results,
        &framework_buffer_results,
    ];

    for result in &results {
        println!("\n📊 {}", result.method);
        println!(
            "   Average time: {:.3} ms ({} ns)",
            result.average_time_ms, result.average_time_ns
        );
        println!(
            "   Total time: {:.3} ms",
            result.total_time_ns as f64 / 1_000_000.0
        );
        println!("   Executions: {}", result.num_executions);

        if result.method != manual_results.method {
            println!(
                "   Overhead vs baseline: {:.2}x ({:.1}%)",
                result.overhead_ratio(&manual_results),
                result.overhead_percentage(&manual_results)
            );
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("Performance Analysis");
    println!("{}", "=".repeat(80));

    println!("\n🚀 Speed Rankings (fastest to slowest):");
    let mut sorted_results = results.clone();
    sorted_results.sort_by_key(|r| r.average_time_ns);

    for (i, result) in sorted_results.iter().enumerate() {
        let rank_emoji = match i {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        println!(
            "   {} {}: {:.3} ms",
            rank_emoji, result.method, result.average_time_ms
        );
    }

    println!("\n📈 Framework Overhead Analysis:");
    println!(
        "   Framework (alloc) vs Manual (baseline): {:.2}x overhead ({:.1}%)",
        framework_alloc_results.overhead_ratio(&manual_results),
        framework_alloc_results.overhead_percentage(&manual_results)
    );

    println!(
        "   Framework (buffer) vs Manual (baseline): {:.2}x overhead ({:.1}%)",
        framework_buffer_results.overhead_ratio(&manual_results),
        framework_buffer_results.overhead_percentage(&manual_results)
    );

    println!(
        "   Framework (buffer) vs Manual (buffer): {:.2}x overhead ({:.1}%)",
        framework_buffer_results.overhead_ratio(&manual_buffered_results),
        framework_buffer_results.overhead_percentage(&manual_buffered_results)
    );

    println!("\n💾 Memory Requirements:");
    println!(
        "   Framework buffer size: {} floats ({} KB)",
        required_memory,
        (required_memory * 4) / 1024
    );
    println!(
        "   Manual buffer size: {} floats ({} KB)",
        10000 + 10,
        ((10000 + 10) * 4) / 1024
    );

    println!("\n{}", "=".repeat(80));
    println!("Benchmark Complete");
    println!("{}", "=".repeat(80));
}
