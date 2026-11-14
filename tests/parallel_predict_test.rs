mod common;

use instmodel_inference::{InstructionModel, PredictConfig};
use serde_json::json;

fn test_model() -> InstructionModel {
    common::model(json!({"buffer_sizes": [3, 2],
        "instructions": [{"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "SIGMOID"}],
        "weights": [[[0.5, -0.3, 0.8], [-0.2, 0.7, 0.1]]], "bias": [[0.1, -0.1]]}))
}

fn test_inputs(num_samples: usize) -> Vec<f32> {
    (0..num_samples * 3)
        .map(|i| (i as f32 * 0.01).sin())
        .collect()
}

#[test]
fn matches_sequential_for_any_thread_count() {
    let model = test_model();
    let inputs = test_inputs(400);
    let expected: Vec<f32> = inputs
        .chunks(3)
        .flat_map(|sample| model.predict(sample).unwrap())
        .collect();

    for config in [
        PredictConfig::new(),
        PredictConfig::new().with_threads(1),
        PredictConfig::new().with_threads(4),
        PredictConfig::new().with_threads(200),
    ] {
        let result = model.predict_parallel(&inputs, config).unwrap();
        assert_eq!(result.num_samples(), 400);
        assert_eq!(result.slice_size(), model.get_output_size());
        common::assert_close(result.as_slice(), &expected, 1e-6, "parallel");
    }
}

#[test]
fn output_accessors() {
    let model = test_model();
    let result = model
        .predict_parallel(&test_inputs(10), PredictConfig::new())
        .unwrap();

    let mut dest = vec![0.0f32; 10 * model.get_output_size()];
    result.copy_results(&mut dest).unwrap();
    assert_eq!(dest, result.as_slice());
    assert!(result.copy_results(&mut [0.0; 5]).is_err());

    let rows = result.copy_results_to_vec();
    assert_eq!(rows.len(), 10);
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(result.get_result(i).unwrap(), row.as_slice());
    }
    assert!(result.get_result(10).is_err());
}

#[test]
fn custom_slice_returns_whole_buffer() {
    let model = test_model();
    let inputs = test_inputs(10);
    let config = PredictConfig::new().with_slice_result_buffer(0, model.required_memory());
    let result = model.predict_parallel(&inputs, config).unwrap();
    assert_eq!(result.slice_size(), model.required_memory());
    assert_eq!(&result.get_result(2).unwrap()[..3], &inputs[6..9]);
}

#[test]
fn invalid_requests() {
    let model = test_model();
    let empty = model.predict_parallel(&[], PredictConfig::new()).unwrap();
    assert_eq!(empty.num_samples(), 0);
    assert!(empty.as_slice().is_empty());

    assert!(
        model
            .predict_parallel(&[1.0; 7], PredictConfig::new())
            .is_err()
    );

    let inputs = test_inputs(10);
    for (start, end) in [(5, 3), (0, model.required_memory() + 10)] {
        let config = PredictConfig::new().with_slice_result_buffer(start, end);
        assert!(model.predict_parallel(&inputs, config).is_err());
    }
}
