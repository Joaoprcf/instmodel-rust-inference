use instmodel_inference::instruction_model_info::{DotInstructionInfo, InstructionInfo};
use instmodel_inference::{Activation, InstructionModel, InstructionModelInfo, PredictConfig};
use std::time::Instant;

fn create_benchmark_model() -> InstructionModel {
    let input_size = 300;
    let hidden_size = 500;
    let output_size = 5;

    let weights_layer0: Vec<Vec<f32>> = (0..hidden_size)
        .map(|i| {
            (0..input_size)
                .map(|j| ((i * input_size + j) as f32 * 0.001).sin() * 0.1)
                .collect()
        })
        .collect();

    let weights_layer1: Vec<Vec<f32>> = (0..output_size)
        .map(|i| {
            (0..hidden_size)
                .map(|j| ((i * hidden_size + j) as f32 * 0.002).cos() * 0.1)
                .collect()
        })
        .collect();

    let bias_layer0: Vec<f32> = (0..hidden_size)
        .map(|i| (i as f32 * 0.01).sin() * 0.01)
        .collect();
    let bias_layer1: Vec<f32> = (0..output_size)
        .map(|i| (i as f32 * 0.01).cos() * 0.01)
        .collect();

    let model_info = InstructionModelInfo {
        features: None,
        feature_size: Some(input_size),
        computation_buffer_sizes: vec![input_size, hidden_size, output_size],
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
        weights: vec![weights_layer0, weights_layer1],
        bias: vec![bias_layer0, bias_layer1],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    InstructionModel::new(model_info).expect("Model creation should succeed")
}

fn generate_inputs(num_samples: usize, feature_size: usize) -> Vec<f32> {
    (0..num_samples * feature_size)
        .map(|i| (i as f32 * 0.001).sin())
        .collect()
}

fn main() {
    let model = create_benchmark_model();
    let num_samples = 50_000;
    let feature_size = model.get_feature_size();
    let output_size = model.get_output_size();

    println!("Model: {} -> {} -> {}", feature_size, 500, output_size);
    println!("Samples: {}", num_samples);
    println!(
        "Required memory per inference: {} f32 values",
        model.required_memory()
    );
    println!();

    let inputs = generate_inputs(num_samples, feature_size);

    // Warmup
    let _ = model.predict(&inputs[..feature_size]);

    // Sequential benchmark
    println!("Running sequential inference...");
    let start = Instant::now();
    let mut sequential_results = Vec::with_capacity(num_samples * output_size);
    for i in 0..num_samples {
        let input_start = i * feature_size;
        let input_end = input_start + feature_size;
        let result = model.predict(&inputs[input_start..input_end]).unwrap();
        sequential_results.extend(result);
    }
    let sequential_duration = start.elapsed();
    println!(
        "Sequential: {:.3?} ({:.3} inferences/sec)",
        sequential_duration,
        num_samples as f64 / sequential_duration.as_secs_f64()
    );

    // Parallel benchmark with default config (uses all CPU cores)
    println!("\nRunning parallel inference (default threads)...");
    let config = PredictConfig::new();
    let start = Instant::now();
    let parallel_result = model.predict_parallel(&inputs, config).unwrap();
    let parallel_duration = start.elapsed();
    println!(
        "Parallel:   {:.3?} ({:.3} inferences/sec)",
        parallel_duration,
        num_samples as f64 / parallel_duration.as_secs_f64()
    );

    // Verify results match
    let parallel_buffer = parallel_result.as_slice();
    let mut max_diff: f32 = 0.0;
    for (seq, par) in sequential_results.iter().zip(parallel_buffer.iter()) {
        max_diff = max_diff.max((seq - par).abs());
    }
    println!(
        "\nMax difference between sequential and parallel: {:.2e}",
        max_diff
    );

    // Speedup
    let speedup = sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64();
    println!("\nSpeedup: {:.3}x", speedup);
}
