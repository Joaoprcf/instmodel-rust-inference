//! Executable instructions operating on a unified computation buffer.

use std::collections::HashMap;

use crate::errors::InstructionModelError;
use crate::instruction_model_info::InstructionInfo;
use crate::instruction_model_validation::validate_instruction;

pub mod activation_instruction;
pub mod add_buffer_heads_instruction;
pub mod attention_instruction;
pub mod clip_elementwise_instruction;
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
pub use clip_elementwise_instruction::ClipElementwiseInstruction;
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

    /// Overwrites this instruction's dense parameters in place: `weights` is the
    /// flattened row-major weight matrix and `bias` the bias vector.
    ///
    /// Only instructions that own dense parameters (dot and attention) support this;
    /// the default implementation reports [`InstructionModelError::ThetaUnsupported`].
    fn set_dense_params(
        &mut self,
        _weights: &[f32],
        _bias: &[f32],
    ) -> Result<(), InstructionModelError> {
        Err(InstructionModelError::ThetaUnsupported {
            reason: "instruction owns no dense parameters".to_string(),
        })
    }

    /// Reads this instruction's dense parameters into `weights` (flattened row-major)
    /// and `bias`. Same support surface as [`Instruction::set_dense_params`].
    fn read_dense_params(
        &self,
        _weights: &mut [f32],
        _bias: &mut [f32],
    ) -> Result<(), InstructionModelError> {
        Err(InstructionModelError::ThetaUnsupported {
            reason: "instruction owns no dense parameters".to_string(),
        })
    }

    /// Returns a boxed copy of this instruction, or `None` when the concrete type
    /// does not support cloning (possible for external trait implementations, which
    /// predate this method and inherit the default).
    fn clone_box(&self) -> Option<Box<dyn Instruction>> {
        None
    }
}

/// Validates `instruction_info` against the model context and builds the executable instruction.
pub fn create_instruction(
    instruction_info: &InstructionInfo,
    computation_buffer_indexes: &[usize],
    computation_buffer_sizes: &[usize],
    weights: &[Vec<Vec<f32>>],
    bias: &[Vec<f32>],
    parameters: &[Vec<f32>],
    maps: &[HashMap<String, Vec<f32>>],
) -> Result<Box<dyn Instruction>, InstructionModelError> {
    validate_instruction(
        instruction_info,
        computation_buffer_sizes,
        weights,
        bias,
        parameters,
        maps,
    )?;
    let ptr = |buffer: usize| computation_buffer_indexes[buffer];
    let size = |buffer: usize| computation_buffer_sizes[buffer];

    Ok(match instruction_info {
        InstructionInfo::Dot(info) => Box::new(DotInstruction::new(
            ptr(info.input),
            ptr(info.output),
            size(info.output),
            &weights[info.weights],
            &bias[info.weights],
            info.activation,
        )?),
        InstructionInfo::Copy(info) => Box::new(CopyInstruction::new(
            ptr(info.input),
            ptr(info.output) + info.internal_index,
            size(info.input),
        )),
        InstructionInfo::CopyMasked(info) => Box::new(CopyMaskedInstruction::new(
            ptr(info.input),
            ptr(info.output),
            &info.indexes,
        )),
        InstructionInfo::Activation(info) => Box::new(ActivationInstruction::new(
            info.activation,
            ptr(info.input),
            size(info.input),
        )),
        InstructionInfo::ElemWiseAdd(info) => Box::new(ElemWiseAddInstruction::new(
            ptr(info.input),
            size(info.input),
            &parameters[info.parameters],
        )),
        InstructionInfo::ElemWiseMul(info) => Box::new(ElemWiseMulInstruction::new(
            ptr(info.input),
            size(info.input),
            &parameters[info.parameters],
        )),
        InstructionInfo::ClipElementwise(info) => Box::new(ClipElementwiseInstruction::new(
            ptr(info.input),
            size(info.input),
            info.parameters_min.map(|idx| parameters[idx].clone()),
            info.parameters_max.map(|idx| parameters[idx].clone()),
        )),
        InstructionInfo::MapTransform(info) => Box::new(MapTransformInstruction::new(
            ptr(info.input) + info.internal_input_index,
            ptr(info.output) + info.internal_output_index,
            info.size,
            &maps[info.map],
            &info.default_value,
        )),
        InstructionInfo::ElemWiseBuffersAdd(info) => Box::new(ElemWiseBuffersAddInstruction::new(
            info.input.iter().map(|&idx| ptr(idx)).collect(),
            ptr(info.output),
            size(info.output),
        )),
        InstructionInfo::ElemWiseBuffersMul(info) => Box::new(ElemWiseBuffersMulInstruction::new(
            info.input.iter().map(|&idx| ptr(idx)).collect(),
            ptr(info.output),
            size(info.output),
        )),
        InstructionInfo::ReduceSum(info) => Box::new(ReduceSumInstruction::new(
            ptr(info.input),
            ptr(info.output),
            size(info.input),
        )),
        InstructionInfo::Attention(info) => Box::new(AttentionInstruction::new(
            ptr(info.input),
            ptr(info.key),
            ptr(info.output),
            size(info.output),
            &weights[info.weights],
            &bias[info.weights],
        )),
        InstructionInfo::MultiplyBufferHeads(info) => {
            let (data, heads) = (info.input[0], info.input[1]);
            Box::new(MultiplyBufferHeadsInstruction::new(
                ptr(data),
                ptr(heads),
                ptr(info.output),
                size(data),
                size(heads),
            )?)
        }
        InstructionInfo::AddBufferHeads(info) => {
            let (data, heads) = (info.input[0], info.input[1]);
            Box::new(AddBufferHeadsInstruction::new(
                ptr(data),
                ptr(heads),
                ptr(info.output),
                size(data),
                size(heads),
            )?)
        }
    })
}

/// Borrows `buf[src..src + src_len]` and `buf[dst..dst + dst_len]` together, or `None`
/// when the ranges overlap.
pub(crate) fn split_src_dst(
    buf: &mut [f32],
    src: usize,
    src_len: usize,
    dst: usize,
    dst_len: usize,
) -> Option<(&[f32], &mut [f32])> {
    if src + src_len <= dst {
        let (head, tail) = buf.split_at_mut(dst);
        Some((&head[src..src + src_len], &mut tail[..dst_len]))
    } else if dst + dst_len <= src {
        let (head, tail) = buf.split_at_mut(src);
        Some((&tail[..src_len], &mut head[dst..dst + dst_len]))
    } else {
        None
    }
}
