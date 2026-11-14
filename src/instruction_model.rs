//! [`InstructionModel`]: validated, executable form of an [`InstructionModelInfo`].

use crate::errors::{InstructionModelError, InstructionModelResult, ParallelPredictResult};
use crate::high_performance_execution_utils::ParallelExecutionGraph;
use crate::instruction_model_info::{InstructionInfo, InstructionModelInfo};
use crate::instruction_model_validation as validation;
use crate::instructions::{Instruction, create_instruction};
use crate::parallel_predict::{ParallelPredictOutput, PredictConfig, execute_parallel_predict};
use crate::params::{ParamLayout, ThetaBinding};

thread_local! {
    static PREDICT_SCRATCH_BUFFER: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The instructions plus, per weight slot, the indices of the instructions that
/// own a copy of that slot's parameters.
type BuiltInstructions = (Vec<Box<dyn Instruction>>, Vec<Vec<usize>>);

/// Inference engine executing a validated instruction sequence over one unified buffer.
pub struct InstructionModel {
    pub(crate) instructions: Vec<Box<dyn Instruction>>,
    pub(crate) feature_size: usize,
    pub(crate) computation_buffer_sizes: Vec<usize>,
    pub(crate) computation_buffer_indexes: Vec<usize>,
    pub(crate) output_index_start: usize,
    pub(crate) output_index_end: usize,
    pub(crate) parallel_graph: ParallelExecutionGraph,
    pub(crate) theta_binding: Option<ThetaBinding>,
}

impl InstructionModel {
    /// Validates `instruction_model_info` (including its `validation_data`, if any) and builds the model.
    pub fn new(instruction_model_info: InstructionModelInfo) -> InstructionModelResult<Self> {
        let feature_size = validation::validate_inputs(&instruction_model_info)?;
        let mut model = Self::with_layout(
            instruction_model_info.computation_buffer_sizes.clone(),
            Vec::new(),
            feature_size,
        )?;
        validation::validate_required_memory(model.output_index_end)?;

        let (instructions, slot_owners) = Self::validate_and_create_instructions(
            &instruction_model_info,
            &model.computation_buffer_indexes,
            &model.computation_buffer_sizes,
        )?;
        model.instructions = instructions;
        model.theta_binding = Some(ThetaBinding {
            layout: ParamLayout::from_info(&instruction_model_info)?,
            slot_owners,
        });
        model.parallel_graph = ParallelExecutionGraph::build(
            &instruction_model_info.instructions,
            model.computation_buffer_sizes.len(),
        )?;

        if let Some(validation_data) = &instruction_model_info.validation_data {
            model.validate_model(
                &validation_data.inputs,
                &validation_data.expected_outputs,
                1e-5,
            )?;
        }

        Ok(model)
    }

    /// Builds a model from pre-built instructions, skipping instruction validation (tests only).
    pub fn new_for_test(
        computation_buffer_sizes: Vec<usize>,
        instructions: Vec<Box<dyn Instruction>>,
        feature_size: usize,
    ) -> InstructionModelResult<Self> {
        if computation_buffer_sizes.is_empty() {
            return Err(InstructionModelError::NoLayersProvided);
        }
        validation::validate_feature_size(feature_size, &computation_buffer_sizes)?;
        Self::with_layout(computation_buffer_sizes, instructions, feature_size)
    }

    fn with_layout(
        computation_buffer_sizes: Vec<usize>,
        instructions: Vec<Box<dyn Instruction>>,
        feature_size: usize,
    ) -> InstructionModelResult<Self> {
        let mut computation_buffer_indexes = Vec::new();
        let output_index_end = validation::calculate_computation_buffer_indexes(
            &computation_buffer_sizes,
            &mut computation_buffer_indexes,
        )?;
        let output_size = computation_buffer_sizes[computation_buffer_sizes.len() - 1];
        Ok(InstructionModel {
            instructions,
            feature_size,
            output_index_start: output_index_end.saturating_sub(output_size),
            output_index_end,
            computation_buffer_sizes,
            computation_buffer_indexes,
            parallel_graph: ParallelExecutionGraph {
                nodes: Vec::new(),
                root_indices: Vec::new(),
                is_parallelizable: false,
                buffer_last_nodes: Vec::new(),
            },
            theta_binding: None,
        })
    }

    /// Feature count declared by `features` (`name[n]` counts n) or `feature_size`.
    pub fn calculate_feature_size(
        instruction_model_info: &InstructionModelInfo,
    ) -> InstructionModelResult<usize> {
        validation::calculate_feature_size(instruction_model_info)
    }

    fn validate_and_create_instructions(
        instruction_model_info: &InstructionModelInfo,
        computation_buffer_indexes: &[usize],
        computation_buffer_sizes: &[usize],
    ) -> InstructionModelResult<BuiltInstructions> {
        let weights = &instruction_model_info.weights;
        let parameters = instruction_model_info.parameters.as_deref().unwrap_or(&[]);
        let maps = instruction_model_info.maps.as_deref().unwrap_or(&[]);

        let mut instructions = Vec::new();
        let mut slot_owners: Vec<Vec<usize>> = vec![Vec::new(); weights.len()];
        let mut used_parameters = vec![false; parameters.len()];
        let mut used_maps = vec![false; maps.len()];

        for (instruction_index, instruction_info) in
            instruction_model_info.instructions.iter().enumerate()
        {
            instructions.push(create_instruction(
                instruction_info,
                computation_buffer_indexes,
                computation_buffer_sizes,
                weights,
                &instruction_model_info.bias,
                parameters,
                maps,
            )?);

            match instruction_info {
                InstructionInfo::Dot(info) => slot_owners[info.weights].push(instruction_index),
                InstructionInfo::Attention(info) => {
                    slot_owners[info.weights].push(instruction_index)
                }
                InstructionInfo::ElemWiseAdd(info) => used_parameters[info.parameters] = true,
                InstructionInfo::ElemWiseMul(info) => used_parameters[info.parameters] = true,
                InstructionInfo::ClipElementwise(info) => {
                    for idx in [info.parameters_min, info.parameters_max]
                        .into_iter()
                        .flatten()
                    {
                        used_parameters[idx] = true;
                    }
                }
                InstructionInfo::MapTransform(info) => used_maps[info.map] = true,
                _ => {}
            }
        }

        if let Some(index) = slot_owners.iter().position(Vec::is_empty) {
            return Err(InstructionModelError::UnusedWeights { index });
        }
        if let Some(index) = used_parameters.iter().position(|used| !used) {
            return Err(InstructionModelError::UnusedParameters { index });
        }
        if let Some(index) = used_maps.iter().position(|used| !used) {
            return Err(InstructionModelError::UnusedMap { index });
        }

        Ok((instructions, slot_owners))
    }

    /// Runs `inputs` and fails on the first output differing from `outputs` by more than `delta`.
    pub fn validate_model(
        &mut self,
        inputs: &[Vec<f32>],
        outputs: &[Vec<f32>],
        delta: f32,
    ) -> InstructionModelResult<()> {
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

            temporary_buffer[..input.len()].copy_from_slice(input);
            self.predict_with_buffer(&mut temporary_buffer)?;

            // The error reports outputs up to and including the first mismatch.
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

    /// Unified buffer length required by [`Self::predict_with_buffer`].
    pub fn required_memory(&self) -> usize {
        self.output_index_end
    }

    /// Runs the model in place; inputs occupy the buffer start, outputs the last buffer.
    pub fn predict_with_buffer(
        &self,
        unified_computation_buffer: &mut [f32],
    ) -> InstructionModelResult<()> {
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

    /// Runs one sample using a thread-local scratch buffer and returns the output.
    pub fn predict(&self, input: &[f32]) -> InstructionModelResult<Vec<f32>> {
        if input.len() != self.feature_size {
            return Err(InstructionModelError::ValidationInputSizeMismatch {
                provided: input.len(),
                expected: self.feature_size,
            });
        }

        let output_size = self.computation_buffer_sizes[self.computation_buffer_sizes.len() - 1];

        PREDICT_SCRATCH_BUFFER.with(|scratch| {
            let mut scratch_buffer = scratch.borrow_mut();
            if scratch_buffer.len() < self.output_index_end {
                scratch_buffer.resize(self.output_index_end, 0.0f32);
            }
            scratch_buffer[..self.output_index_end].fill(0.0f32);
            scratch_buffer[..self.feature_size].copy_from_slice(input);

            self.predict_with_buffer(scratch_buffer.as_mut_slice())?;

            let output_start = self.output_index_start;
            let output_end = output_start + output_size;
            Ok(scratch_buffer[output_start..output_end].to_vec())
        })
    }

    /// Last output value.
    pub fn predict_single(&self, input: &[f32]) -> InstructionModelResult<f32> {
        let output = self.predict(input)?;
        Ok(output[output.len() - 1])
    }

    /// Predicts outputs for multiple samples in parallel.
    pub fn predict_parallel(
        &self,
        inputs: &[f32],
        config: PredictConfig,
    ) -> ParallelPredictResult<ParallelPredictOutput> {
        execute_parallel_predict(self, inputs, &config)
    }

    /// Output `index` read from a buffer filled by [`Self::predict_with_buffer`].
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

    /// Returns whether the model can benefit from parallel execution.
    pub fn is_parallelizable(&self) -> bool {
        self.parallel_graph.is_parallelizable
    }

    /// Returns a reference to the parallel execution graph.
    pub fn get_parallel_graph(&self) -> &ParallelExecutionGraph {
        &self.parallel_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
