//! Tests for error types in the neural inference library.
//!
//! This module tests error variants that are actually implemented and validated
//! in the current codebase to ensure proper error handling.

use instmodel_inference::errors::{InstructionModelError, ValidationError};
use instmodel_inference::instruction_model::InstructionModel;
use instmodel_inference::instruction_model_info::{
    CopyInstructionInfo, InstructionInfo, InstructionModelInfo, ValidationData,
};
use std::collections::HashMap;

#[cfg(test)]
mod instruction_model_error_tests {
    use super::*;

    #[test]
    fn test_computation_buffer_size_exceeds_limit() {
        // Create buffer sizes that exceed the maximum allowed computation buffer size
        let huge_buffer_size = 2_000_000; // Exceeds MAX_COMPUTATION_BUFFER_SIZE

        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(1),
            computation_buffer_sizes: vec![1, huge_buffer_size],
            instructions: vec![InstructionInfo::Copy(CopyInstructionInfo {
                input: 0,
                output: 1,
                internal_index: 0,
            })],
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = InstructionModel::new(info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::ComputationBufferSizeExceedsLimit { .. })
        ));
    }

    #[test]
    fn test_unused_weights() {
        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(2),
            computation_buffer_sizes: vec![2, 2],
            instructions: vec![InstructionInfo::Copy(CopyInstructionInfo {
                input: 0,
                output: 1,
                internal_index: 0,
            })],
            weights: vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]], // Unused weights
            bias: vec![vec![1.0, 2.0]],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = InstructionModel::new(info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::UnusedWeights { index: 0 })
        ));
    }

    #[test]
    fn test_unused_parameters() {
        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(2),
            computation_buffer_sizes: vec![2, 2],
            instructions: vec![InstructionInfo::Copy(CopyInstructionInfo {
                input: 0,
                output: 1,
                internal_index: 0,
            })],
            weights: vec![],
            bias: vec![],
            parameters: Some(vec![vec![1.0, 2.0]]), // Unused parameters
            maps: None,
            validation_data: None,
        };

        let result = InstructionModel::new(info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::UnusedParameters { index: 0 })
        ));
    }

    #[test]
    fn test_unused_map() {
        let mut map = HashMap::new();
        map.insert("1.0".to_string(), vec![2.0, 3.0]);

        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(2),
            computation_buffer_sizes: vec![2, 2],
            instructions: vec![InstructionInfo::Copy(CopyInstructionInfo {
                input: 0,
                output: 1,
                internal_index: 0,
            })],
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: Some(vec![map]), // Unused map
            validation_data: None,
        };

        let result = InstructionModel::new(info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::UnusedMap { index: 0 })
        ));
    }

    #[test]
    fn test_validation_mismatch() {
        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(2),
            computation_buffer_sizes: vec![2, 2],
            instructions: vec![InstructionInfo::Copy(CopyInstructionInfo {
                input: 0,
                output: 1,
                internal_index: 0,
            })],
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: Some(ValidationData {
                inputs: vec![vec![1.0, 2.0]],
                expected_outputs: vec![vec![5.0, 6.0]], // Different from copy result
            }),
        };

        let result = InstructionModel::new(info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::ValidationMismatch { .. })
        ));
    }

    #[test]
    fn test_feature_size_exceeds_buffer_size() {
        let result = InstructionModel::new_for_test(
            vec![3, 2], // Total buffer needed is 5, but feature size is 10
            vec![],
            10,
        );
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::InvalidFeatureSize { .. })
        ));
    }
}

#[cfg(test)]
mod validation_error_tests {
    use super::*;

    #[test]
    fn test_validation_failed() {
        let error = ValidationError::ValidationFailed {
            message: "Test validation failure".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Validation failed: Test validation failure"
        );
    }

    #[test]
    fn test_invalid_validation_data() {
        let error = ValidationError::InvalidValidationData {
            reason: "Input and output lengths don't match".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Invalid validation data: Input and output lengths don't match"
        );
    }

    #[test]
    fn test_tolerance_exceeded() {
        let error = ValidationError::ToleranceExceeded {
            expected: 1.0,
            actual: 1.5,
            tolerance: 0.1,
        };
        assert_eq!(
            error.to_string(),
            "Validation tolerance exceeded: expected 1, got 1.5, tolerance 0.1"
        );
    }
}

#[cfg(test)]
mod error_display_tests {
    use super::*;

    #[test]
    fn test_basic_error_displays() {
        assert_eq!(
            InstructionModelError::MissingFeatures.to_string(),
            "Features or feature size must be provided"
        );
        assert_eq!(
            InstructionModelError::NoLayersProvided.to_string(),
            "At least one layer is required"
        );
        assert_eq!(
            InstructionModelError::NoInstructionsProvided.to_string(),
            "At least one instruction is required"
        );
        assert_eq!(
            InstructionModelError::BiasWeightsMismatch.to_string(),
            "The numbers of bias and weights must be the same"
        );
        assert_eq!(
            InstructionModelError::TooManyWeightsForInstructions.to_string(),
            "The number of weights/bias must not exceed the number instructions"
        );
        assert_eq!(
            InstructionModelError::InvalidLayerSize.to_string(),
            "The size of the layer must be greater than 0"
        );
        assert_eq!(
            InstructionModelError::InvalidUnifiedBufferSize.to_string(),
            "The size of the unified computation buffer must be greater than 0"
        );
        assert_eq!(
            InstructionModelError::ValidationInputOutputMismatch.to_string(),
            "The number of inputs must match the number of outputs in the validation data"
        );
        assert_eq!(
            InstructionModelError::InputOutputCountMismatch.to_string(),
            "The number of inputs must match the number of outputs"
        );
    }

    #[test]
    fn test_complex_error_displays() {
        let error = InstructionModelError::InvalidFeatureFormat {
            feature: "invalid[".to_string(),
        };
        assert_eq!(error.to_string(), "Invalid feature format: invalid[");

        let error = InstructionModelError::FeatureSizeMismatch {
            expected: 5,
            actual: 3,
        };
        assert_eq!(
            error.to_string(),
            "Feature size mismatch: expected 5 but got 3 from features"
        );

        let error = InstructionModelError::WeightSizeExceedsLimit {
            actual: 15000000,
            max: 10000000,
        };
        assert_eq!(
            error.to_string(),
            "The size of the weights exceeds the maximum allowed size: 15000000 > 10000000"
        );
    }
}
