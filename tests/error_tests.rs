mod common;

use instmodel_inference::errors::{InstructionModelError as E, ValidationError};
use instmodel_inference::instruction_model::InstructionModel;
use serde_json::{Value, json};

fn copy_model(extra: Value) -> Value {
    let mut model = json!({"buffer_sizes": [2, 2],
        "instructions": [{"type": "COPY", "input": 0, "output": 1, "internal_index": 0}]});
    model
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    model
}

#[test]
fn construction_errors() {
    type IsExpected = fn(&E) -> bool;
    let cases: [(&str, Value, IsExpected); 5] = [
        (
            "buffer_limit",
            copy_model(json!({"feature_size": 1, "buffer_sizes": [1, 2_000_000]})),
            |e| matches!(e, E::ComputationBufferSizeExceedsLimit { .. }),
        ),
        (
            "unused_weights",
            copy_model(json!({"weights": [[[1.0, 2.0], [3.0, 4.0]]], "bias": [[1.0, 2.0]]})),
            |e| matches!(e, E::UnusedWeights { index: 0 }),
        ),
        (
            "unused_parameters",
            copy_model(json!({"parameters": [[1.0, 2.0]]})),
            |e| matches!(e, E::UnusedParameters { index: 0 }),
        ),
        (
            "unused_map",
            copy_model(json!({"maps": [{"1.0": [2.0, 3.0]}]})),
            |e| matches!(e, E::UnusedMap { index: 0 }),
        ),
        (
            "validation_mismatch",
            copy_model(
                json!({"validation_data": {"inputs": [[1.0, 2.0]], "expected_outputs": [[5.0, 6.0]]}}),
            ),
            |e| matches!(e, E::ValidationMismatch { .. }),
        ),
    ];
    for (name, model_json, is_expected) in cases {
        match InstructionModel::new(common::info(model_json)) {
            Err(error) => assert!(is_expected(&error), "{name}: unexpected error {error}"),
            Ok(_) => panic!("{name}: model was accepted"),
        }
    }

    assert!(matches!(
        InstructionModel::new_for_test(vec![3, 2], vec![], 10),
        Err(E::InvalidFeatureSize { .. })
    ));
}

#[test]
fn error_messages() {
    let cases = [
        (
            E::MissingFeatures.to_string(),
            "Features or feature size must be provided",
        ),
        (
            E::NoLayersProvided.to_string(),
            "At least one layer is required",
        ),
        (
            E::NoInstructionsProvided.to_string(),
            "At least one instruction is required",
        ),
        (
            E::BiasWeightsMismatch.to_string(),
            "The numbers of bias and weights must be the same",
        ),
        (
            E::TooManyWeightsForInstructions.to_string(),
            "The number of weights/bias must not exceed the number instructions",
        ),
        (
            E::InvalidLayerSize.to_string(),
            "The size of the layer must be greater than 0",
        ),
        (
            E::InvalidUnifiedBufferSize.to_string(),
            "The size of the unified computation buffer must be greater than 0",
        ),
        (
            E::ValidationInputOutputMismatch.to_string(),
            "The number of inputs must match the number of outputs in the validation data",
        ),
        (
            E::InputOutputCountMismatch.to_string(),
            "The number of inputs must match the number of outputs",
        ),
        (
            E::InvalidFeatureFormat {
                feature: "invalid[".to_string(),
            }
            .to_string(),
            "Invalid feature format: invalid[",
        ),
        (
            E::FeatureSizeMismatch {
                expected: 5,
                actual: 3,
            }
            .to_string(),
            "Feature size mismatch: expected 5 but got 3 from features",
        ),
        (
            E::WeightSizeExceedsLimit {
                actual: 15000000,
                max: 10000000,
            }
            .to_string(),
            "The size of the weights exceeds the maximum allowed size: 15000000 > 10000000",
        ),
        (
            ValidationError::ValidationFailed {
                message: "Test validation failure".to_string(),
            }
            .to_string(),
            "Validation failed: Test validation failure",
        ),
        (
            ValidationError::InvalidValidationData {
                reason: "lengths differ".to_string(),
            }
            .to_string(),
            "Invalid validation data: lengths differ",
        ),
        (
            ValidationError::ToleranceExceeded {
                expected: 1.0,
                actual: 1.5,
                tolerance: 0.1,
            }
            .to_string(),
            "Validation tolerance exceeded: expected 1, got 1.5, tolerance 0.1",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, expected);
    }
}
