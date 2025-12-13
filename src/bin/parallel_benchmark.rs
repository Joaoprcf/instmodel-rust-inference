use instmodel_inference::instruction_model_info::{DotInstructionInfo, InstructionInfo};
use instmodel_inference::{Activation, InstructionModel, InstructionModelInfo, PredictConfig};
use std::time::Instant;

const INPUT_SIZE: usize = 250;
const HIDDEN_SIZE: usize = 300;
const OUTPUT_SIZE: usize = 200;
const NUM_SAMPLES: usize = 200_000;

fn create_benchmark_model() -> InstructionModel {
    let weights_layer0: Vec<Vec<f32>> = (0..HIDDEN_SIZE)
        .map(|i| {
            (0..INPUT_SIZE)
                .map(|j| ((i * INPUT_SIZE + j) as f32 * 0.001).sin() * 0.1)
                .collect()
        })
        .collect();

    let weights_layer1: Vec<Vec<f32>> = (0..OUTPUT_SIZE)
        .map(|i| {
            (0..HIDDEN_SIZE)
                .map(|j| ((i * HIDDEN_SIZE + j) as f32 * 0.002).cos() * 0.1)
                .collect()
        })
        .collect();

    let bias_layer0: Vec<f32> = (0..HIDDEN_SIZE)
        .map(|i| (i as f32 * 0.01).sin() * 0.01)
        .collect();
    let bias_layer1: Vec<f32> = (0..OUTPUT_SIZE)
        .map(|i| (i as f32 * 0.01).cos() * 0.01)
        .collect();

    let model_info = InstructionModelInfo {
        features: None,
        feature_size: Some(INPUT_SIZE),
        computation_buffer_sizes: vec![INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE],
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
    let num_samples = NUM_SAMPLES;
    let feature_size = model.get_feature_size();
    let output_size = model.get_output_size();

    println!(
        "Model: {} -> {} -> {}",
        feature_size, HIDDEN_SIZE, output_size
    );
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
    let output_start = model.get_output_index_start();
    let mut sequential_buffer = vec![0.0f32; model.required_memory()];
    let mut sequential_results = Vec::with_capacity(num_samples * output_size);
    for i in 0..num_samples {
        let input_start = i * feature_size;
        let input_end = input_start + feature_size;
        sequential_buffer[feature_size..].fill(0.0f32);
        sequential_buffer[..feature_size].copy_from_slice(&inputs[input_start..input_end]);
        model
            .predict_with_buffer(sequential_buffer.as_mut_slice())
            .unwrap();
        sequential_results
            .extend_from_slice(&sequential_buffer[output_start..output_start + output_size]);
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
