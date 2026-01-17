//! Instruction implementations for neural network operations.
//!
//! This module contains all the instruction types that can be executed
//! by the neural inference engine. Each instruction operates on computation
//! buffers to perform specific neural network operations.

use crate::errors::InstructionModelError;

pub mod activation_instruction;
pub mod add_buffer_heads_instruction;
pub mod attention_instruction;
pub mod copy_instruction;
pub mod copy_masked_instruction;
pub mod dot_instruction;
pub mod elem_wise_add_instruction;
pub mod elem_wise_buffers_add_instruction;
pub mod elem_wise_buffers_mul_instruction;
pub mod elem_wise_mul_instruction;
pub mod map_transform_instruction;
pub mod multiply_buffer_heads_instruction;
pub mod reduce_sum_instruction;

pub use activation_instruction::ActivationInstruction;
pub use add_buffer_heads_instruction::AddBufferHeadsInstruction;
pub use attention_instruction::AttentionInstruction;
pub use copy_instruction::CopyInstruction;
pub use copy_masked_instruction::CopyMaskedInstruction;
pub use dot_instruction::DotInstruction;
pub use elem_wise_add_instruction::ElemWiseAddInstruction;
pub use elem_wise_buffers_add_instruction::ElemWiseBuffersAddInstruction;
pub use elem_wise_buffers_mul_instruction::ElemWiseBuffersMulInstruction;
pub use elem_wise_mul_instruction::ElemWiseMulInstruction;
pub use map_transform_instruction::MapTransformInstruction;
pub use multiply_buffer_heads_instruction::MultiplyBufferHeadsInstruction;
pub use reduce_sum_instruction::ReduceSumInstruction;

/// Base trait for all instruction types.
///
/// All instructions have default information about the output pointer and the data size.
/// These are essential to any operation as they represent the minimum information needed
/// to identify where the result of the operation will be stored.
pub trait Instruction: Send + Sync {
    /// Returns the output pointer that represents the first index in the buffer to copy to.
    fn output_ptr(&self) -> usize;

    /// Returns the size of the data that will be stored at the output pointer.
    fn data_size(&self) -> usize;

    /// Applies the instruction to the computation buffer.
    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError>;
}

/// Creates an instruction from instruction info and model context.
pub fn create_instruction(
    instruction_info: &crate::instruction_model_info::InstructionInfo,
    computation_buffer_indexes: &[usize],
    computation_buffer_sizes: &[usize],
    weights: &[Vec<Vec<f32>>],
    bias: &[Vec<f32>],
    parameters: &[Vec<f32>],
    maps: &[std::collections::HashMap<String, Vec<f32>>],
) -> Result<Box<dyn Instruction>, InstructionModelError> {
    use crate::instruction_model_info::InstructionInfo;

    match instruction_info {
        InstructionInfo::Dot(info) => {
            if info.input == info.output {
                return Err(InstructionModelError::SameInputOutputIndexes {
                    instruction_type: "DOT".to_string(),
                });
            }
            if info.weights >= weights.len() {
                return Err(InstructionModelError::WeightsIndexOutOfBounds {
                    index: info.weights,
                });
            }

            let weights_matrix = &weights[info.weights];
            let bias_vector =
                bias.get(info.weights)
                    .ok_or(InstructionModelError::WeightsIndexOutOfBounds {
                        index: info.weights,
                    })?;

            let input_size = computation_buffer_sizes[info.input];
            let output_size = computation_buffer_sizes[info.output];

            if weights_matrix.len() != output_size {
                return Err(InstructionModelError::WeightsRowSizeMismatch {
                    weights_rows: weights_matrix.len(),
                    output_size,
                });
            }
            if bias_vector.len() != output_size {
                return Err(InstructionModelError::BiasOutputSizeMismatch {
                    bias_index: info.weights,
                    output_index: info.output,
                    bias_size: bias_vector.len(),
                    output_size,
                });
            }
            for row in weights_matrix {
                if row.len() != input_size {
                    return Err(InstructionModelError::WeightsColumnSizeMismatch {
                        input_index: info.input,
                        weights_columns: row.len(),
                        input_size,
                    });
                }
            }

            let instruction = DotInstruction::new(
                computation_buffer_indexes[info.input],
                computation_buffer_indexes[info.output],
                computation_buffer_sizes[info.output],
                weights_matrix,
                bias_vector,
                info.activation,
            )?;
            Ok(Box::new(instruction))
        }
        InstructionInfo::Copy(info) => {
            let instruction = CopyInstruction::new(
                computation_buffer_indexes[info.input],
                computation_buffer_indexes[info.output] + info.internal_index,
                computation_buffer_sizes[info.input],
            );
            Ok(Box::new(instruction))
        }
        InstructionInfo::CopyMasked(info) => {
            let instruction = CopyMaskedInstruction::new(
                computation_buffer_indexes[info.input],
                computation_buffer_indexes[info.output],
                &info.indexes,
            );
            Ok(Box::new(instruction))
        }
        InstructionInfo::Activation(info) => {
            let instruction = ActivationInstruction::new(
                info.activation,
                computation_buffer_indexes[info.input],
                computation_buffer_sizes[info.input],
            );
            Ok(Box::new(instruction))
        }
        InstructionInfo::ElemWiseAdd(info) => {
            let instruction = ElemWiseAddInstruction::new(
                computation_buffer_indexes[info.input],
                computation_buffer_sizes[info.input],
                &parameters[info.parameters],
            );
            Ok(Box::new(instruction))
        }
        InstructionInfo::ElemWiseMul(info) => {
            let instruction = ElemWiseMulInstruction::new(
                computation_buffer_indexes[info.input],
                computation_buffer_sizes[info.input],
                &parameters[info.parameters],
            );
            Ok(Box::new(instruction))
        }
        InstructionInfo::MapTransform(info) => {
            let instruction = MapTransformInstruction::new(
                computation_buffer_indexes[info.input] + info.internal_input_index,
                computation_buffer_indexes[info.output] + info.internal_output_index,
                info.size,
                &maps[info.map],
                &info.default_value,
            );
            Ok(Box::new(instruction))
        }
        InstructionInfo::ElemWiseBuffersAdd(info) => {
            let input_ptrs: Vec<usize> = info
                .input
                .iter()
                .map(|&idx| computation_buffer_indexes[idx])
                .collect();
            let instruction = ElemWiseBuffersAddInstruction::new(
                input_ptrs,
                computation_buffer_indexes[info.output],
                computation_buffer_sizes[info.output],
            );
            Ok(Box::new(instruction))
        }
        InstructionInfo::ElemWiseBuffersMul(info) => {
            let input_ptrs: Vec<usize> = info
                .input
                .iter()
                .map(|&idx| computation_buffer_indexes[idx])
                .collect();
            let instruction = ElemWiseBuffersMulInstruction::new(
                input_ptrs,
                computation_buffer_indexes[info.output],
                computation_buffer_sizes[info.output],
            );
            Ok(Box::new(instruction))
        }
        InstructionInfo::ReduceSum(info) => {
            let instruction = ReduceSumInstruction::new(
                computation_buffer_indexes[info.input],
                computation_buffer_indexes[info.output],
                computation_buffer_sizes[info.input],
            );
            Ok(Box::new(instruction))
        }
        InstructionInfo::Attention(info) => {
            if info.weights >= weights.len() {
                return Err(InstructionModelError::WeightsIndexOutOfBounds {
                    index: info.weights,
                });
            }

            let weights_matrix = &weights[info.weights];
            let bias_vector =
                bias.get(info.weights)
                    .ok_or(InstructionModelError::WeightsIndexOutOfBounds {
                        index: info.weights,
                    })?;

            let query_size = computation_buffer_sizes[info.input];
            let key_size = computation_buffer_sizes[info.key];
            let output_size = computation_buffer_sizes[info.output];

            if query_size != output_size {
                return Err(InstructionModelError::InputBufferSizeMismatch {
                    index: info.input,
                    actual_size: query_size,
                    expected_size: output_size,
                });
            }
            if weights_matrix.len() != output_size {
                return Err(InstructionModelError::WeightsRowSizeMismatch {
                    weights_rows: weights_matrix.len(),
                    output_size,
                });
            }
            if bias_vector.len() != output_size {
                return Err(InstructionModelError::BiasOutputSizeMismatch {
                    bias_index: info.weights,
                    output_index: info.output,
                    bias_size: bias_vector.len(),
                    output_size,
                });
            }
            for row in weights_matrix {
                if row.len() != key_size {
                    return Err(InstructionModelError::WeightsColumnSizeMismatch {
                        input_index: info.key,
                        weights_columns: row.len(),
                        input_size: key_size,
                    });
                }
            }

            let instruction = AttentionInstruction::new(
                computation_buffer_indexes[info.input],
                computation_buffer_indexes[info.key],
                computation_buffer_indexes[info.output],
                computation_buffer_sizes[info.output],
                weights_matrix,
                bias_vector,
            );
            Ok(Box::new(instruction))
        }
        InstructionInfo::MultiplyBufferHeads(info) => {
            if info.input.len() != 2 {
                return Err(InstructionModelError::InsufficientInputBuffers);
            }
            let data_idx = info.input[0];
            let heads_idx = info.input[1];
            let data_ptr = computation_buffer_indexes[data_idx];
            let heads_ptr = computation_buffer_indexes[heads_idx];
            let output_ptr = computation_buffer_indexes[info.output];
            let data_size = computation_buffer_sizes[data_idx];
            let heads_size = computation_buffer_sizes[heads_idx];

            let instruction = MultiplyBufferHeadsInstruction::new(
                data_ptr, heads_ptr, output_ptr, data_size, heads_size,
            )?;
            Ok(Box::new(instruction))
        }
        InstructionInfo::AddBufferHeads(info) => {
            if info.input.len() != 2 {
                return Err(InstructionModelError::InsufficientInputBuffers);
            }
            let data_idx = info.input[0];
            let heads_idx = info.input[1];
            let data_ptr = computation_buffer_indexes[data_idx];
            let heads_ptr = computation_buffer_indexes[heads_idx];
            let output_ptr = computation_buffer_indexes[info.output];
            let data_size = computation_buffer_sizes[data_idx];
            let heads_size = computation_buffer_sizes[heads_idx];

            let instruction = AddBufferHeadsInstruction::new(
                data_ptr, heads_ptr, output_ptr, data_size, heads_size,
            )?;
            Ok(Box::new(instruction))
        }
    }
}
