//! Construction-time validation helpers for [`crate::InstructionModel`].
//!
//! Extracted from `instruction_model.rs` to keep that file focused on the
//! model's runtime behavior; everything here runs once during
//! [`crate::InstructionModel::new`].

use crate::errors::{
    BufferIndexOutOfBoundsError, ComputationBufferSizeExceedsLimitError, FeatureSizeMismatchError,
    InstructionModelError, InstructionModelResult, InvalidFeatureSizeError,
    ValidationInputOutputMismatchError,
};
use crate::instruction_model_info::{InstructionModelInfo, ValidationData};

/// Maximum computation buffer size (default configuration)
pub(crate) const MAX_COMPUTATION_BUFFER_SIZE: usize = 1_000_000;

/// Maximum weight size (default configuration)
pub(crate) const MAX_WEIGHT_SIZE: usize = 10_000_000;

/// Computes the total feature size based on the provided instruction model information.
pub(crate) fn calculate_feature_size(
    instruction_model_info: &InstructionModelInfo,
) -> InstructionModelResult<usize> {
    if let Some(features) = &instruction_model_info.features {
        let mut total_size = 0;
        for feature in features {
            if let Some(open_bracket) = feature.find('[') {
                if feature.ends_with(']') {
                    // Check for exactly one '[' and one ']'
                    if feature.chars().filter(|&c| c == '[' || c == ']').count() != 2 {
                        return Err(InstructionModelError::InvalidFeatureFormat {
                            feature: feature.clone(),
                        });
                    }

                    let prefix = &feature[..open_bracket];
                    let number_str = &feature[open_bracket + 1..feature.len() - 1];

                    if prefix.is_empty() || number_str.is_empty() {
                        return Err(InstructionModelError::InvalidFeatureFormat {
                            feature: feature.clone(),
                        });
                    }

                    match number_str.parse::<usize>() {
                        Ok(number) if number > 0 => total_size += number,
                        _ => {
                            return Err(InstructionModelError::InvalidFeatureFormat {
                                feature: feature.clone(),
                            });
                        }
                    }
                } else {
                    return Err(InstructionModelError::InvalidFeatureFormat {
                        feature: feature.clone(),
                    });
                }
            } else if feature.contains('[') || feature.contains(']') {
                return Err(InstructionModelError::InvalidFeatureFormat {
                    feature: feature.clone(),
                });
            } else {
                total_size += 1;
            }
        }

        validate_declared_feature_size(instruction_model_info.feature_size, total_size)?;
        Ok(total_size)
    } else if let Some(feature_size) = instruction_model_info.feature_size {
        Ok(feature_size)
    } else {
        Err(InstructionModelError::MissingFeatures)
    }
}

/// Validates that the specified feature size exactly fills one or more complete input buffers.
pub(crate) fn validate_feature_size(
    feature_size: usize,
    buffer_sizes: &[usize],
) -> std::result::Result<(), InvalidFeatureSizeError> {
    let mut accumulated = 0;
    let mut accumulated_capacities = Vec::new();

    for &capacity in buffer_sizes {
        accumulated += capacity;
        accumulated_capacities.push(accumulated);
        match accumulated.cmp(&feature_size) {
            std::cmp::Ordering::Equal => return Ok(()),
            std::cmp::Ordering::Greater => break,
            std::cmp::Ordering::Less => {}
        }
    }

    Err(InvalidFeatureSizeError {
        expected: feature_size,
        actual: accumulated,
        capacities: accumulated_capacities,
    })
}

/// Performs basic initial validation of the inputs of the model.
pub(crate) fn validate_inputs(
    instruction_model_info: &InstructionModelInfo,
) -> InstructionModelResult<()> {
    if instruction_model_info.features.is_none() && instruction_model_info.feature_size.is_none() {
        return Err(InstructionModelError::MissingFeatures);
    }

    if instruction_model_info.computation_buffer_sizes.is_empty() {
        return Err(InstructionModelError::NoLayersProvided);
    }

    if instruction_model_info.bias.len() != instruction_model_info.weights.len() {
        return Err(InstructionModelError::BiasWeightsMismatch);
    }

    if instruction_model_info.instructions.is_empty() {
        return Err(InstructionModelError::NoInstructionsProvided);
    }

    if instruction_model_info.bias.len() > instruction_model_info.instructions.len() {
        return Err(InstructionModelError::TooManyWeightsForInstructions);
    }

    let mut calculated_size = 0;
    for (i, bias_vec) in instruction_model_info.bias.iter().enumerate() {
        calculated_size += bias_vec.len();
        if bias_vec.len() != instruction_model_info.weights[i].len() {
            return Err(InstructionModelError::BiasWeightsSizeMismatch {
                index: i,
                bias_size: bias_vec.len(),
                weights_size: instruction_model_info.weights[i].len(),
            });
        }
        for weights_column in &instruction_model_info.weights[i] {
            calculated_size += weights_column.len();
        }
    }

    if calculated_size > MAX_WEIGHT_SIZE {
        return Err(InstructionModelError::WeightSizeExceedsLimit {
            actual: calculated_size,
            max: MAX_WEIGHT_SIZE,
        });
    }

    validate_feature_size(
        calculate_feature_size(instruction_model_info)?,
        &instruction_model_info.computation_buffer_sizes,
    )?;

    validate_validation_data_lengths(instruction_model_info.validation_data.as_ref())?;
    Ok(())
}

/// Validate if the model required memory is within the maximum allowed.
pub(crate) fn validate_required_memory(
    output_index_end: usize,
) -> std::result::Result<(), ComputationBufferSizeExceedsLimitError> {
    if output_index_end > MAX_COMPUTATION_BUFFER_SIZE {
        return Err(ComputationBufferSizeExceedsLimitError {
            actual: output_index_end,
            max: MAX_COMPUTATION_BUFFER_SIZE,
        });
    }
    Ok(())
}

/// Validates that declared feature size matches computed size.
pub(crate) fn validate_declared_feature_size(
    declared: Option<usize>,
    computed: usize,
) -> std::result::Result<(), FeatureSizeMismatchError> {
    if let Some(expected) = declared.filter(|&fs| fs != computed) {
        return Err(FeatureSizeMismatchError {
            expected,
            actual: computed,
        });
    }
    Ok(())
}

/// Validates that validation data has matching input/output lengths.
pub(crate) fn validate_validation_data_lengths(
    validation_data: Option<&ValidationData>,
) -> std::result::Result<(), ValidationInputOutputMismatchError> {
    if validation_data.is_some_and(|vd| vd.inputs.len() != vd.expected_outputs.len()) {
        return Err(ValidationInputOutputMismatchError);
    }
    Ok(())
}

/// Fills `computation_buffer_indexes` with the start offset of each buffer and
/// returns the total unified buffer size.
pub(crate) fn calculate_computation_buffer_indexes(
    computation_buffer_sizes: &[usize],
    computation_buffer_indexes: &mut Vec<usize>,
) -> InstructionModelResult<usize> {
    computation_buffer_indexes.push(0);
    let input_layer_size = computation_buffer_sizes[0];
    let mut index = input_layer_size;

    for &computation_buffer_size in computation_buffer_sizes.iter().skip(1) {
        if computation_buffer_size == 0 {
            return Err(InstructionModelError::InvalidLayerSize);
        }
        computation_buffer_indexes.push(index);
        index += computation_buffer_size;
    }

    if index == 0 {
        return Err(InstructionModelError::InvalidUnifiedBufferSize);
    }

    Ok(index)
}

/// Validates buffer index.
pub(crate) fn validate_buffer_index(
    label: &str,
    buffer_index: usize,
    max_size: usize,
) -> std::result::Result<(), BufferIndexOutOfBoundsError> {
    if buffer_index >= max_size {
        return Err(BufferIndexOutOfBoundsError {
            label: label.to_string(),
            index: buffer_index,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_feature_size_from_features() {
        let info = InstructionModelInfo {
            features: Some(vec![
                "feature1".to_string(),
                "feature2[3]".to_string(),
                "feature3".to_string(),
            ]),
            feature_size: None,
            computation_buffer_sizes: vec![5],
            instructions: vec![],
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = calculate_feature_size(&info).unwrap();
        assert_eq!(result, 5); // 1 + 3 + 1
    }

    #[test]
    fn test_calculate_feature_size_from_size() {
        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(10),
            computation_buffer_sizes: vec![10],
            instructions: vec![],
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = calculate_feature_size(&info).unwrap();
        assert_eq!(result, 10);
    }

    #[test]
    fn test_invalid_feature_format_empty_brackets() {
        let info = InstructionModelInfo {
            features: Some(vec!["feature[]".to_string()]),
            feature_size: None,
            computation_buffer_sizes: vec![1],
            instructions: vec![],
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = calculate_feature_size(&info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::InvalidFeatureFormat { .. })
        ));
    }

    #[test]
    fn test_invalid_feature_format_multiple_brackets() {
        let info = InstructionModelInfo {
            features: Some(vec!["feature[[5]]".to_string()]),
            feature_size: None,
            computation_buffer_sizes: vec![1],
            instructions: vec![],
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = calculate_feature_size(&info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::InvalidFeatureFormat { .. })
        ));
    }

    #[test]
    fn test_invalid_feature_format_no_closing_bracket() {
        let info = InstructionModelInfo {
            features: Some(vec!["feature[5".to_string()]),
            feature_size: None,
            computation_buffer_sizes: vec![1],
            instructions: vec![],
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = calculate_feature_size(&info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::InvalidFeatureFormat { .. })
        ));
    }

    #[test]
    fn test_invalid_feature_format_invalid_number() {
        let info = InstructionModelInfo {
            features: Some(vec!["feature[abc]".to_string()]),
            feature_size: None,
            computation_buffer_sizes: vec![1],
            instructions: vec![],
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = calculate_feature_size(&info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::InvalidFeatureFormat { .. })
        ));
    }

    #[test]
    fn test_invalid_feature_format_zero_number() {
        let info = InstructionModelInfo {
            features: Some(vec!["feature[0]".to_string()]),
            feature_size: None,
            computation_buffer_sizes: vec![1],
            instructions: vec![],
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = calculate_feature_size(&info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::InvalidFeatureFormat { .. })
        ));
    }

    #[test]
    fn test_feature_size_mismatch() {
        let info = InstructionModelInfo {
            features: Some(vec!["feature1".to_string(), "feature2".to_string()]),
            feature_size: Some(5), // Should be 2
            computation_buffer_sizes: vec![2],
            instructions: vec![],
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = calculate_feature_size(&info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::FeatureSizeMismatch {
                expected: 5,
                actual: 2
            })
        ));
    }

    #[test]
    fn test_missing_features() {
        let info = InstructionModelInfo {
            features: None,
            feature_size: None,
            computation_buffer_sizes: vec![1],
            instructions: vec![],
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = calculate_feature_size(&info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::MissingFeatures)
        ));
    }

    #[test]
    fn test_invalid_feature_size() {
        let result = validate_feature_size(5, &[2, 2]); // 5 doesn't fit in 2+2=4
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InvalidFeatureSizeError {
                expected: 5,
                actual: 4,
                ..
            })
        ));
    }

    #[test]
    fn test_no_layers_provided() {
        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(1),
            computation_buffer_sizes: vec![], // Empty
            instructions: vec![],
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = validate_inputs(&info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::NoLayersProvided)
        ));
    }

    #[test]
    fn test_no_instructions_provided() {
        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(1),
            computation_buffer_sizes: vec![1],
            instructions: vec![], // Empty
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = validate_inputs(&info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::NoInstructionsProvided)
        ));
    }

    #[test]
    fn test_bias_weights_mismatch() {
        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(1),
            computation_buffer_sizes: vec![1],
            instructions: vec![],
            weights: vec![vec![vec![1.0]]], // 1 weight
            bias: vec![],                   // 0 bias
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = validate_inputs(&info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::BiasWeightsMismatch)
        ));
    }

    #[test]
    fn test_too_many_weights_for_instructions() {
        use crate::instruction_model_info::{CopyInstructionInfo, InstructionInfo};

        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(1),
            computation_buffer_sizes: vec![1, 1],
            instructions: vec![InstructionInfo::Copy(CopyInstructionInfo {
                input: 0,
                output: 1,
                internal_index: 0,
            })], // 1 instruction
            weights: vec![vec![vec![1.0]], vec![vec![2.0]]], // 2 weights
            bias: vec![vec![1.0], vec![2.0]],                // 2 bias
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = validate_inputs(&info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::TooManyWeightsForInstructions)
        ));
    }

    #[test]
    fn test_bias_weights_size_mismatch() {
        use crate::instruction_model_info::{DotInstructionInfo, InstructionInfo};

        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(1),
            computation_buffer_sizes: vec![1, 2],
            instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
                input: 0,
                output: 1,
                weights: 0,
                activation: None,
            })], // Need instruction to pass earlier validation
            weights: vec![vec![vec![1.0], vec![2.0]]], // 2 rows, 1 column each
            bias: vec![vec![1.0]], // 1 bias element, should be 2 to match 2 rows
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let result = validate_inputs(&info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::BiasWeightsSizeMismatch { index: 0, .. })
        ));
    }

    #[test]
    fn test_validation_input_output_mismatch() {
        use crate::instruction_model_info::{CopyInstructionInfo, InstructionInfo};

        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(1),
            computation_buffer_sizes: vec![1, 1],
            instructions: vec![InstructionInfo::Copy(CopyInstructionInfo {
                input: 0,
                output: 1,
                internal_index: 0,
            })], // Need instruction to pass earlier validation
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: Some(ValidationData {
                inputs: vec![vec![1.0]],  // 1 input
                expected_outputs: vec![], // 0 outputs
            }),
        };

        let result = validate_inputs(&info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::ValidationInputOutputMismatch)
        ));
    }

    #[test]
    fn test_invalid_layer_size() {
        let mut buffer_indexes = Vec::new();
        let result = calculate_computation_buffer_indexes(
            &[1, 0], // Second layer has size 0
            &mut buffer_indexes,
        );
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::InvalidLayerSize)
        ));
    }

    #[test]
    fn test_invalid_unified_buffer_size() {
        let mut buffer_indexes = Vec::new();
        let result = calculate_computation_buffer_indexes(
            &[0], // Only layer has size 0
            &mut buffer_indexes,
        );
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::InvalidUnifiedBufferSize)
        ));
    }

    #[test]
    fn test_buffer_index_out_of_bounds() {
        let result = validate_buffer_index("test", 5, 3);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(BufferIndexOutOfBoundsError { label, index: 5 }) if label == "test"
        ));
    }
}
