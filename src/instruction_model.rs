//! Core instruction model for neural network inference.
//!
//! This module contains the main InstructionModel struct which orchestrates
//! the execution of neural network inference through a sequence of instructions
//! operating on computation buffers.

use crate::errors::{InstructionModelError, Result};
use crate::instruction_model_info::InstructionModelInfo;
use crate::instructions::{Instruction, create_instruction};

/// Maximum computation buffer size (default configuration)
const MAX_COMPUTATION_BUFFER_SIZE: usize = 1_000_000;

/// Maximum weight size (default configuration)  
const MAX_WEIGHT_SIZE: usize = 10_000_000;

/// Represents a model that is configured to follow optimized computation following a sequence of instructions.
/// This model configuration can be represented via a JSON file that corresponds to an InstructionModelInfo record.
/// The model is generally used as an inference computation engine generated from a trained neural network.
pub struct InstructionModel {
    instructions: Vec<Box<dyn Instruction>>,
    feature_size: usize,
    computation_buffer_sizes: Vec<usize>,
    computation_buffer_indexes: Vec<usize>,
    output_index_start: usize,
    output_index_end: usize,
}

impl InstructionModel {
    /// Creates a new InstructionModel from InstructionModelInfo.
    pub fn new(instruction_model_info: InstructionModelInfo) -> Result<Self> {
        Self::validate_inputs(&instruction_model_info)?;

        let computation_buffer_sizes = instruction_model_info.computation_buffer_sizes.clone();
        let feature_size = Self::calculate_feature_size(&instruction_model_info)?;
        Self::validate_feature_size(feature_size, &computation_buffer_sizes)?;

        let mut computation_buffer_indexes = Vec::new();
        let output_index_end = Self::calculate_computation_buffer_indexes(
            &computation_buffer_sizes,
            &mut computation_buffer_indexes,
        )?;
        let output_index_start =
            output_index_end - computation_buffer_sizes[computation_buffer_sizes.len() - 1];

        Self::validate_required_memory(output_index_end)?;

        let instructions = Self::validate_and_create_instructions(
            &instruction_model_info,
            &computation_buffer_indexes,
            &computation_buffer_sizes,
        )?;

        let mut model = InstructionModel {
            instructions,
            feature_size,
            computation_buffer_sizes,
            computation_buffer_indexes,
            output_index_start,
            output_index_end,
        };

        // Validate with provided validation data
        if let Some(validation_data) = &instruction_model_info.validation_data {
            model.validate_model(
                &validation_data.inputs,
                &validation_data.expected_outputs,
                1e-5,
            )?;
        }

        Ok(model)
    }

    /// Creates a new InstructionModel for test purposes only.
    pub fn new_for_test(
        computation_buffer_sizes: Vec<usize>,
        instructions: Vec<Box<dyn Instruction>>,
        feature_size: usize,
    ) -> Result<Self> {
        if computation_buffer_sizes.is_empty() {
            return Err(InstructionModelError::NoLayersProvided);
        }
        Self::validate_feature_size(feature_size, &computation_buffer_sizes)?;

        let mut computation_buffer_indexes = Vec::new();
        let output_index_end = Self::calculate_computation_buffer_indexes(
            &computation_buffer_sizes,
            &mut computation_buffer_indexes,
        )?;
        let output_index_start =
            output_index_end - computation_buffer_sizes[computation_buffer_sizes.len() - 1];

        Ok(InstructionModel {
            instructions,
            feature_size,
            computation_buffer_sizes,
            computation_buffer_indexes,
            output_index_start,
            output_index_end,
        })
    }

    /// Computes the total feature size based on the provided instruction model information.
    pub fn calculate_feature_size(instruction_model_info: &InstructionModelInfo) -> Result<usize> {
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

            if let Some(feature_size) = instruction_model_info.feature_size {
                if feature_size != total_size {
                    return Err(InstructionModelError::FeatureSizeMismatch {
                        expected: feature_size,
                        actual: total_size,
                    });
                }
            }

            Ok(total_size)
        } else if let Some(feature_size) = instruction_model_info.feature_size {
            Ok(feature_size)
        } else {
            Err(InstructionModelError::MissingFeatures)
        }
    }

    /// Validates that the specified feature size exactly fills one or more complete input buffers.
    fn validate_feature_size(feature_size: usize, buffer_sizes: &[usize]) -> Result<()> {
        let mut accumulated = 0;
        let mut accumulated_capacities = Vec::new();

        for &capacity in buffer_sizes {
            accumulated += capacity;
            accumulated_capacities.push(accumulated);
            if accumulated == feature_size {
                return Ok(());
            } else if accumulated > feature_size {
                break;
            }
        }

        Err(InstructionModelError::InvalidFeatureSize {
            expected: feature_size,
            actual: accumulated,
            capacities: accumulated_capacities,
        })
    }

    /// Performs basic initial validation of the inputs of the model.
    fn validate_inputs(instruction_model_info: &InstructionModelInfo) -> Result<()> {
        if instruction_model_info.features.is_none()
            && instruction_model_info.feature_size.is_none()
        {
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

        Self::validate_feature_size(
            Self::calculate_feature_size(instruction_model_info)?,
            &instruction_model_info.computation_buffer_sizes,
        )?;

        if let Some(validation_data) = &instruction_model_info.validation_data {
            if validation_data.inputs.len() != validation_data.expected_outputs.len() {
                return Err(InstructionModelError::ValidationInputOutputMismatch);
            }
        }

        Ok(())
    }

    /// Validate if the model required memory is within the maximum allowed.
    fn validate_required_memory(output_index_end: usize) -> Result<()> {
        if output_index_end > MAX_COMPUTATION_BUFFER_SIZE {
            return Err(InstructionModelError::ComputationBufferSizeExceedsLimit {
                actual: output_index_end,
                max: MAX_COMPUTATION_BUFFER_SIZE,
            });
        }
        Ok(())
    }

    fn calculate_computation_buffer_indexes(
        computation_buffer_sizes: &[usize],
        computation_buffer_indexes: &mut Vec<usize>,
    ) -> Result<usize> {
        computation_buffer_indexes.push(0);
        let input_layer_size = computation_buffer_sizes[0];
        let mut index = input_layer_size;

        for i in 1..computation_buffer_sizes.len() {
            let computation_buffer_size = computation_buffer_sizes[i];
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

    fn validate_and_create_instructions(
        instruction_model_info: &InstructionModelInfo,
        computation_buffer_indexes: &[usize],
        computation_buffer_sizes: &[usize],
    ) -> Result<Vec<Box<dyn Instruction>>> {
        let weights = &instruction_model_info.weights;
        let bias = &instruction_model_info.bias;
        let parameters = instruction_model_info.parameters.as_deref().unwrap_or(&[]);
        let maps = instruction_model_info.maps.as_deref().unwrap_or(&[]);

        let mut instructions = Vec::new();
        let mut used_weights = vec![false; weights.len()];
        let mut used_parameters = vec![false; parameters.len()];
        let mut used_maps = vec![false; maps.len()];

        for instruction_info in &instruction_model_info.instructions {
            // Validate input and output buffer indices
            for &input_index in &instruction_info.get_inputs() {
                Self::validate_buffer_index("input", input_index, computation_buffer_sizes.len())?;
            }
            Self::validate_buffer_index(
                "output",
                instruction_info.output(),
                computation_buffer_sizes.len(),
            )?;

            let instruction = create_instruction(
                instruction_info,
                computation_buffer_indexes,
                computation_buffer_sizes,
                weights,
                bias,
                parameters,
                maps,
            )?;

            // Mark resources as used
            match instruction_info {
                crate::instruction_model_info::InstructionInfo::Dot(info) => {
                    used_weights[info.weights] = true;
                }
                crate::instruction_model_info::InstructionInfo::ElemWiseAdd(info) => {
                    used_parameters[info.parameters] = true;
                }
                crate::instruction_model_info::InstructionInfo::ElemWiseMul(info) => {
                    used_parameters[info.parameters] = true;
                }
                crate::instruction_model_info::InstructionInfo::MapTransform(info) => {
                    used_maps[info.map] = true;
                }
                _ => {}
            }

            instructions.push(instruction);
        }

        // Check for unused resources
        for (i, &used) in used_weights.iter().enumerate() {
            if !used {
                return Err(InstructionModelError::UnusedWeights { index: i });
            }
        }
        for (i, &used) in used_parameters.iter().enumerate() {
            if !used {
                return Err(InstructionModelError::UnusedParameters { index: i });
            }
        }
        for (i, &used) in used_maps.iter().enumerate() {
            if !used {
                return Err(InstructionModelError::UnusedMap { index: i });
            }
        }

        Ok(instructions)
    }

    /// Validates buffer index.
    fn validate_buffer_index(label: &str, buffer_index: usize, max_size: usize) -> Result<()> {
        if buffer_index >= max_size {
            return Err(InstructionModelError::BufferIndexOutOfBounds {
                label: label.to_string(),
                index: buffer_index,
            });
        }
        Ok(())
    }

    /// Validates the model using inference on input data and comparing the expected outputs with the predicted outputs.
    pub fn validate_model(
        &mut self,
        inputs: &[Vec<f32>],
        outputs: &[Vec<f32>],
        delta: f32,
    ) -> Result<()> {
        if inputs.len() != outputs.len() {
            return Err(InstructionModelError::InputOutputCountMismatch);
        }

        let mut temporary_buffer = vec![0.0f32; self.output_index_end];
        let last_layer_size =
            self.computation_buffer_sizes[self.computation_buffer_sizes.len() - 1];

        for (i, (input, expected_output)) in inputs.iter().zip(outputs.iter()).enumerate() {
            if input.len() != self.feature_size {
                return Err(InstructionModelError::ValidationInputSizeMismatch {
                    provided: input.len(),
                    expected: self.feature_size,
                });
            }
            if expected_output.len() != last_layer_size {
                return Err(InstructionModelError::ValidationOutputSizeMismatch {
                    index: i,
                    provided: expected_output.len(),
                    expected: last_layer_size,
                });
            }

            // Copy input to buffer
            for (j, &value) in input.iter().enumerate() {
                temporary_buffer[j] = value;
            }

            // Run prediction
            self.predict_with_buffer(&mut temporary_buffer)?;

            // Check results
            let mut computed_output = Vec::new();
            for j in 0..expected_output.len() {
                let computed = temporary_buffer[self.output_index_start + j];
                computed_output.push(computed);
                if (expected_output[j] - computed).abs() > delta {
                    return Err(InstructionModelError::ValidationMismatch {
                        case_number: i,
                        inputs: input.clone(),
                        expected: expected_output.clone(),
                        computed: computed_output,
                    });
                }
            }
        }

        Ok(())
    }

    /// Returns the required memory size.
    pub fn required_memory(&self) -> usize {
        self.output_index_end
    }

    /// Predicts output using the provided computation buffer.
    pub fn predict_with_buffer(&self, unified_computation_buffer: &mut [f32]) -> Result<()> {
        if unified_computation_buffer.len() < self.output_index_end {
            return Err(InstructionModelError::ComputationBufferTooSmall {
                buffer_size: unified_computation_buffer.len(),
                required_size: self.output_index_end,
            });
        }

        for instruction in &self.instructions {
            instruction.apply(unified_computation_buffer)?;
        }

        Ok(())
    }

    /// Predicts output, allocating a new computation buffer.
    pub fn predict(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.feature_size {
            return Err(InstructionModelError::ValidationInputSizeMismatch {
                provided: input.len(),
                expected: self.feature_size,
            });
        }

        let mut unified_computation_buffer = vec![0.0f32; self.output_index_end];

        // Copy input to buffer
        for (i, &value) in input.iter().enumerate() {
            unified_computation_buffer[i] = value;
        }

        self.predict_with_buffer(&mut unified_computation_buffer)?;

        // Extract output
        let output_size = self.computation_buffer_sizes[self.computation_buffer_sizes.len() - 1];
        let mut result = vec![0.0f32; output_size];
        for i in 0..output_size {
            result[i] = unified_computation_buffer[self.output_index_start + i];
        }

        Ok(result)
    }

    /// Predicts a single output value.
    pub fn predict_single(&self, input: &[f32]) -> Result<f32> {
        let output = self.predict(input)?;
        Ok(output[output.len() - 1])
    }

    /// Gets the output value at a specific index.
    pub fn get_output(&self, unified_computation_buffer: &[f32], index: usize) -> f32 {
        unified_computation_buffer[self.output_index_start + index]
    }

    /// Returns the size of the input layer.
    pub fn get_feature_size(&self) -> usize {
        self.feature_size
    }

    /// Returns the size of the output layer.
    pub fn get_output_size(&self) -> usize {
        self.computation_buffer_sizes[self.computation_buffer_sizes.len() - 1]
    }

    /// Returns the start index of the output layer in the computation buffer.
    pub fn get_output_index_start(&self) -> usize {
        self.output_index_start
    }

    /// Returns the computation buffer sizes.
    pub fn get_computation_buffer_sizes(&self) -> &[usize] {
        &self.computation_buffer_sizes
    }

    /// Returns the computation buffer indexes.
    pub fn get_computation_buffer_indexes(&self) -> &[usize] {
        &self.computation_buffer_indexes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction_model_info::{InstructionInfo, ValidationData};

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

        let result = InstructionModel::calculate_feature_size(&info).unwrap();
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

        let result = InstructionModel::calculate_feature_size(&info).unwrap();
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

        let result = InstructionModel::calculate_feature_size(&info);
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

        let result = InstructionModel::calculate_feature_size(&info);
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

        let result = InstructionModel::calculate_feature_size(&info);
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

        let result = InstructionModel::calculate_feature_size(&info);
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

        let result = InstructionModel::calculate_feature_size(&info);
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

        let result = InstructionModel::calculate_feature_size(&info);
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

        let result = InstructionModel::calculate_feature_size(&info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::MissingFeatures)
        ));
    }

    #[test]
    fn test_invalid_feature_size() {
        let result = InstructionModel::validate_feature_size(5, &[2, 2]); // 5 doesn't fit in 2+2=4
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::InvalidFeatureSize {
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

        let result = InstructionModel::validate_inputs(&info);
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

        let result = InstructionModel::validate_inputs(&info);
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

        let result = InstructionModel::validate_inputs(&info);
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

        let result = InstructionModel::validate_inputs(&info);
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

        let result = InstructionModel::validate_inputs(&info);
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

        let result = InstructionModel::validate_inputs(&info);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::ValidationInputOutputMismatch)
        ));
    }

    #[test]
    fn test_invalid_layer_size() {
        let mut buffer_indexes = Vec::new();
        let result = InstructionModel::calculate_computation_buffer_indexes(
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
        let result = InstructionModel::calculate_computation_buffer_indexes(
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
    fn test_computation_buffer_too_small() {
        let model = InstructionModel::new_for_test(vec![2, 2], vec![], 2).unwrap();

        let mut small_buffer = vec![0.0; 2]; // Too small
        let result = model.predict_with_buffer(&mut small_buffer);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::ComputationBufferTooSmall { .. })
        ));
    }

    #[test]
    fn test_validation_input_size_mismatch() {
        let model = InstructionModel::new_for_test(vec![2, 2], vec![], 2).unwrap();

        let wrong_input = vec![1.0]; // Size 1, should be 2
        let result = model.predict(&wrong_input);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::ValidationInputSizeMismatch {
                provided: 1,
                expected: 2
            })
        ));
    }

    #[test]
    fn test_input_output_count_mismatch() {
        let mut model = InstructionModel::new_for_test(vec![1, 1], vec![], 1).unwrap();

        let inputs = vec![vec![1.0]]; // 1 input
        let outputs = vec![]; // 0 outputs
        let result = model.validate_model(&inputs, &outputs, 1e-5);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::InputOutputCountMismatch)
        ));
    }

    #[test]
    fn test_validation_output_size_mismatch() {
        let mut model = InstructionModel::new_for_test(
            vec![1, 2], // Output size is 2
            vec![],
            1,
        )
        .unwrap();

        let inputs = vec![vec![1.0]];
        let outputs = vec![vec![1.0]]; // Size 1, should be 2
        let result = model.validate_model(&inputs, &outputs, 1e-5);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::ValidationOutputSizeMismatch {
                index: 0,
                provided: 1,
                expected: 2
            })
        ));
    }

    #[test]
    fn test_buffer_index_out_of_bounds() {
        let result = InstructionModel::validate_buffer_index("test", 5, 3);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InstructionModelError::BufferIndexOutOfBounds { label, index: 5 }) if label == "test"
        ));
    }
}
