use instmodel_inference::instruction_model_info::{DotInstructionInfo, InstructionInfo};
use instmodel_inference::{Activation, InstructionModel, InstructionModelInfo, PredictConfig};

const DELTA: f32 = 1e-6;

fn create_test_model() -> InstructionModel {
    let weights = vec![vec![0.5, -0.3, 0.8], vec![-0.2, 0.7, 0.1]];
    let bias = vec![0.1, -0.1];

    let model_info = InstructionModelInfo {
        features: Some(vec!["f1".to_string(), "f2".to_string(), "f3".to_string()]),
        feature_size: Some(3),
        computation_buffer_sizes: vec![3, 2],
        instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
            input: 0,
            output: 1,
            weights: 0,
            activation: Some(Activation::Sigmoid),
        })],
        weights: vec![weights],
        bias: vec![bias],
        parameters: None,
        maps: None,
        validation_data: None,
    };

    InstructionModel::new(model_info).expect("Model creation should succeed")
}

fn generate_test_inputs(num_samples: usize, feature_size: usize) -> Vec<f32> {
    let mut inputs = Vec::with_capacity(num_samples * feature_size);
    for i in 0..num_samples {
        for j in 0..feature_size {
            let value = ((i * feature_size + j) as f32 * 0.01).sin();
            inputs.push(value);
        }
    }
    inputs
}

#[test]
fn test_parallel_predict_matches_sequential() {
    let model = create_test_model();
    let num_samples = 400;
    let feature_size = model.get_feature_size();
    let output_size = model.get_output_size();

    let inputs = generate_test_inputs(num_samples, feature_size);

    let mut expected_results = Vec::with_capacity(num_samples * output_size);
    for sample_idx in 0..num_samples {
        let start = sample_idx * feature_size;
        let end = start + feature_size;
        let sample_input = &inputs[start..end];
        let result = model
            .predict(sample_input)
            .expect("Sequential predict should succeed");
        expected_results.extend(result);
    }

    let config = PredictConfig::new().with_threads(4);
    let parallel_result = model
        .predict_parallel(&inputs, config)
        .expect("Parallel predict should succeed");

    assert_eq!(parallel_result.num_samples(), num_samples);
    assert_eq!(parallel_result.slice_size(), output_size);

    let parallel_buffer = parallel_result.as_slice();
    assert_eq!(parallel_buffer.len(), expected_results.len());

    for (i, (expected, actual)) in expected_results
        .iter()
        .zip(parallel_buffer.iter())
        .enumerate()
    {
        assert!(
            (expected - actual).abs() < DELTA,
            "Mismatch at index {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_parallel_predict_copy_results() {
    let model = create_test_model();
    let num_samples = 10;
    let feature_size = model.get_feature_size();
    let output_size = model.get_output_size();

    let inputs = generate_test_inputs(num_samples, feature_size);

    let config = PredictConfig::new();
    let result = model
        .predict_parallel(&inputs, config)
        .expect("Parallel predict should succeed");

    let mut dest = vec![0.0f32; num_samples * output_size];
    result.copy_results(&mut dest).expect("Copy should succeed");
    assert_eq!(dest, result.as_slice());

    let vec_results = result.copy_results_to_vec();
    assert_eq!(vec_results.len(), num_samples);
    for v in &vec_results {
        assert_eq!(v.len(), output_size);
    }
}

#[test]
fn test_parallel_predict_get_result() {
    let model = create_test_model();
    let num_samples = 5;
    let feature_size = model.get_feature_size();

    let inputs = generate_test_inputs(num_samples, feature_size);

    let config = PredictConfig::new();
    let result = model
        .predict_parallel(&inputs, config)
        .expect("Parallel predict should succeed");

    for i in 0..num_samples {
        let sample_result = result.get_result(i).expect("Get result should succeed");
        assert_eq!(sample_result.len(), model.get_output_size());
    }

    let out_of_bounds = result.get_result(num_samples);
    assert!(out_of_bounds.is_err());
}

#[test]
fn test_parallel_predict_custom_slice() {
    let model = create_test_model();
    let num_samples = 10;
    let feature_size = model.get_feature_size();

    let inputs = generate_test_inputs(num_samples, feature_size);

    let config = PredictConfig::new().with_slice_result_buffer(0, model.required_memory());

    let result = model
        .predict_parallel(&inputs, config)
        .expect("Parallel predict should succeed");

    assert_eq!(result.slice_size(), model.required_memory());
}

#[test]
fn test_parallel_predict_single_thread() {
    let model = create_test_model();
    let num_samples = 50;
    let feature_size = model.get_feature_size();

    let inputs = generate_test_inputs(num_samples, feature_size);

    let config = PredictConfig::new().with_threads(1);
    let result = model
        .predict_parallel(&inputs, config)
        .expect("Single-thread parallel predict should succeed");

    assert_eq!(result.num_samples(), num_samples);
}

#[test]
fn test_parallel_predict_empty_input() {
    let model = create_test_model();
    let inputs: Vec<f32> = vec![];

    let config = PredictConfig::new();
    let result = model
        .predict_parallel(&inputs, config)
        .expect("Empty input should succeed");

    assert_eq!(result.num_samples(), 0);
    assert!(result.as_slice().is_empty());
}

#[test]
fn test_parallel_predict_input_size_mismatch() {
    let model = create_test_model();
    let inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    let config = PredictConfig::new();
    let result = model.predict_parallel(&inputs, config);

    assert!(result.is_err());
}

#[test]
fn test_parallel_predict_invalid_slice_range() {
    let model = create_test_model();
    let inputs = generate_test_inputs(10, model.get_feature_size());

    let config = PredictConfig::new().with_slice_result_buffer(5, 3);
    let result = model.predict_parallel(&inputs, config);
    assert!(result.is_err());

    let config = PredictConfig::new().with_slice_result_buffer(0, model.required_memory() + 10);
    let result = model.predict_parallel(&inputs, config);
    assert!(result.is_err());
}

#[test]
fn test_parallel_predict_many_threads() {
    let model = create_test_model();
    let num_samples = 100;
    let feature_size = model.get_feature_size();

    let inputs = generate_test_inputs(num_samples, feature_size);

    let config = PredictConfig::new().with_threads(200);
    let result = model
        .predict_parallel(&inputs, config)
        .expect("Many threads should work");

    assert_eq!(result.num_samples(), num_samples);
}

#[test]
fn test_parallel_predict_destination_size_mismatch() {
    let model = create_test_model();
    let num_samples = 10;
    let feature_size = model.get_feature_size();

    let inputs = generate_test_inputs(num_samples, feature_size);

    let config = PredictConfig::new();
    let result = model
        .predict_parallel(&inputs, config)
        .expect("Parallel predict should succeed");

    let mut wrong_dest = vec![0.0f32; 5];
    let copy_result = result.copy_results(&mut wrong_dest);
    assert!(copy_result.is_err());
}
