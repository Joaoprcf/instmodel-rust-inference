//! Instruction implementations for neural network operations.
//!
//! This module contains all the instruction types that can be executed
//! by the neural inference engine. Each instruction operates on computation
//! buffers to perform specific neural network operations.

use crate::errors::InstructionModelError;

pub mod activation_instruction;
pub mod attention_instruction;
pub mod copy_instruction;
pub mod copy_masked_instruction;
pub mod dot_instruction;
pub mod elem_wise_add_instruction;
pub mod elem_wise_buffers_add_instruction;
pub mod elem_wise_buffers_mul_instruction;
pub mod elem_wise_mul_instruction;
pub mod map_transform_instruction;
pub mod reduce_sum_instruction;

pub use activation_instruction::ActivationInstruction;
pub use attention_instruction::AttentionInstruction;
pub use copy_instruction::CopyInstruction;
pub use copy_masked_instruction::CopyMaskedInstruction;
pub use dot_instruction::DotInstruction;
pub use elem_wise_add_instruction::ElemWiseAddInstruction;
pub use elem_wise_buffers_add_instruction::ElemWiseBuffersAddInstruction;
pub use elem_wise_buffers_mul_instruction::ElemWiseBuffersMulInstruction;
pub use elem_wise_mul_instruction::ElemWiseMulInstruction;
pub use map_transform_instruction::MapTransformInstruction;
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
            let instruction = DotInstruction::new(
                computation_buffer_indexes[info.input],
                computation_buffer_indexes[info.output],
                computation_buffer_sizes[info.output],
                &weights[info.weights],
                &bias[info.weights],
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
            let instruction = AttentionInstruction::new(
                computation_buffer_indexes[info.input],
                computation_buffer_indexes[info.key],
                computation_buffer_indexes[info.output],
                computation_buffer_sizes[info.output],
                &weights[info.weights],
                &bias[info.weights],
            );
            Ok(Box::new(instruction))
        }
    }
}
