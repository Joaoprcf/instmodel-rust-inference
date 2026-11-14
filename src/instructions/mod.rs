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

/// One executable step over the unified computation buffer.
pub trait Instruction: Send + Sync {
    /// First buffer index written by [`Instruction::apply`].
    fn output_ptr(&self) -> usize;

    /// Number of values written at [`Instruction::output_ptr`].
    fn data_size(&self) -> usize;

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError>;

    /// Overwrites dense parameters (row-major weights, then bias); only DOT and ATTENTION
    /// support it, others return [`InstructionModelError::ThetaUnsupported`].
    fn set_dense_params(
        &mut self,
        _weights: &[f32],
        _bias: &[f32],
    ) -> Result<(), InstructionModelError> {
        Err(InstructionModelError::ThetaUnsupported {
            reason: "instruction owns no dense parameters".to_string(),
        })
    }

    /// Reads dense parameters; same support as [`Instruction::set_dense_params`].
    fn read_dense_params(
        &self,
        _weights: &mut [f32],
        _bias: &mut [f32],
    ) -> Result<(), InstructionModelError> {
        Err(InstructionModelError::ThetaUnsupported {
            reason: "instruction owns no dense parameters".to_string(),
        })
    }

    /// Boxed copy; `None` for external implementations that do not override it.
    fn clone_box(&self) -> Option<Box<dyn Instruction>> {
        None
    }
}

/// `output_ptr`, `data_size` and `clone_box` for structs with those two fields.
macro_rules! instruction_common {
    () => {
        fn output_ptr(&self) -> usize {
            self.output_ptr
        }

        fn data_size(&self) -> usize {
            self.data_size
        }

        fn clone_box(&self) -> Option<Box<dyn $crate::instructions::Instruction>> {
            Some(Box::new(self.clone()))
        }
    };
}
use instruction_common;

macro_rules! param_elementwise_instruction {
    ($(#[$doc:meta])* $name:ident, $op:tt) => {
        $(#[$doc])*
        #[derive(Clone)]
        pub struct $name {
            output_ptr: usize,
            data_size: usize,
            parameters: Vec<f32>,
        }

        impl $name {
            pub fn new(output_ptr: usize, data_size: usize, parameters: &[f32]) -> Self {
                Self {
                    output_ptr,
                    data_size,
                    parameters: parameters.to_vec(),
                }
            }
        }

        impl $crate::instructions::Instruction for $name {
            $crate::instructions::instruction_common!();

            fn apply(&self, buffer: &mut [f32]) -> Result<(), $crate::errors::InstructionModelError> {
                let values = &mut buffer[self.output_ptr..self.output_ptr + self.data_size];
                for (v, &p) in values.iter_mut().zip(&self.parameters) {
                    *v $op p;
                }
                Ok(())
            }
        }
    };
}
use param_elementwise_instruction;

macro_rules! buffers_elementwise_instruction {
    ($(#[$doc:meta])* $name:ident, $op:tt) => {
        $(#[$doc])*
        #[derive(Clone)]
        pub struct $name {
            input_ptrs: Vec<usize>,
            output_ptr: usize,
            data_size: usize,
        }

        impl $name {
            pub fn new(input_ptrs: Vec<usize>, output_ptr: usize, data_size: usize) -> Self {
                Self {
                    input_ptrs,
                    output_ptr,
                    data_size,
                }
            }
        }

        impl $crate::instructions::Instruction for $name {
            $crate::instructions::instruction_common!();

            fn apply(&self, buffer: &mut [f32]) -> Result<(), $crate::errors::InstructionModelError> {
                let [first, rest @ ..] = self.input_ptrs.as_slice() else {
                    return Err($crate::errors::InstructionModelError::InsufficientInputBuffers);
                };
                if rest.is_empty() {
                    return Err($crate::errors::InstructionModelError::InsufficientInputBuffers);
                }
                for i in 0..self.data_size {
                    let mut acc = buffer[first + i];
                    for &ptr in rest {
                        acc $op buffer[ptr + i];
                    }
                    buffer[self.output_ptr + i] = acc;
                }
                Ok(())
            }
        }
    };
}
use buffers_elementwise_instruction;

macro_rules! buffer_heads_instruction {
    ($(#[$doc:meta])* $name:ident, $op:tt) => {
        $(#[$doc])*
        #[derive(Clone)]
        pub struct $name {
            data_ptr: usize,
            heads_ptr: usize,
            output_ptr: usize,
            data_size: usize,
            num_heads: usize,
            head_dim: usize,
        }

        impl $name {
            pub fn new(
                data_ptr: usize,
                heads_ptr: usize,
                output_ptr: usize,
                data_size: usize,
                heads_size: usize,
            ) -> Result<Self, $crate::errors::InstructionModelError> {
                if heads_size == 0 || !data_size.is_multiple_of(heads_size) {
                    return Err($crate::errors::InstructionModelError::InvalidBufferHeadsSize {
                        data_size,
                        heads_size,
                    });
                }
                Ok(Self {
                    data_ptr,
                    heads_ptr,
                    output_ptr,
                    data_size,
                    num_heads: heads_size,
                    head_dim: data_size / heads_size,
                })
            }
        }

        impl $crate::instructions::Instruction for $name {
            $crate::instructions::instruction_common!();

            fn apply(&self, buffer: &mut [f32]) -> Result<(), $crate::errors::InstructionModelError> {
                for head in 0..self.num_heads {
                    let head_value = buffer[self.heads_ptr + head];
                    for i in head * self.head_dim..(head + 1) * self.head_dim {
                        buffer[self.output_ptr + i] = buffer[self.data_ptr + i] $op head_value;
                    }
                }
                Ok(())
            }
        }
    };
}
use buffer_heads_instruction;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn run(instruction: &dyn Instruction, mut buffer: Vec<f32>) -> Vec<f32> {
        instruction.apply(&mut buffer).unwrap();
        buffer
    }

    #[test]
    fn small_instructions_compute_expected_buffers() {
        let map = HashMap::from([("2".to_string(), vec![7.0, 8.0])]);
        type Case = (Box<dyn Instruction>, Vec<f32>, Vec<f32>);
        let cases: Vec<Case> = vec![
            (
                Box::new(CopyInstruction::new(0, 3, 2)),
                vec![1.0, 2.0, 0.0, 0.0, 0.0],
                vec![1.0, 2.0, 0.0, 1.0, 2.0],
            ),
            (
                Box::new(CopyMaskedInstruction::new(0, 3, &[2, 0])),
                vec![1.0, 2.0, 3.0, 0.0, 0.0],
                vec![1.0, 2.0, 3.0, 3.0, 1.0],
            ),
            (
                Box::new(ActivationInstruction::new(crate::Activation::Relu, 1, 2)),
                vec![-1.0, -2.0, 3.0],
                vec![-1.0, 0.0, 3.0],
            ),
            (
                Box::new(ElemWiseAddInstruction::new(0, 2, &[1.0, -1.0])),
                vec![1.0, 1.0, 9.0],
                vec![2.0, 0.0, 9.0],
            ),
            (
                Box::new(ElemWiseMulInstruction::new(1, 2, &[2.0, 3.0])),
                vec![9.0, 1.0, 2.0],
                vec![9.0, 2.0, 6.0],
            ),
            (
                Box::new(ClipElementwiseInstruction::new(
                    0,
                    3,
                    Some(vec![0.0; 3]),
                    Some(vec![1.0; 3]),
                )),
                vec![-1.0, 0.5, 2.0],
                vec![0.0, 0.5, 1.0],
            ),
            (
                Box::new(ClipElementwiseInstruction::new(
                    0,
                    2,
                    None,
                    Some(vec![1.0, 1.0]),
                )),
                vec![-5.0, 5.0],
                vec![-5.0, 1.0],
            ),
            (
                Box::new(MapTransformInstruction::new(0, 1, 2, &map, &[0.5, 0.5])),
                vec![2.2, 0.0, 0.0],
                vec![2.2, 7.0, 8.0],
            ),
            (
                Box::new(MapTransformInstruction::new(0, 1, 2, &map, &[0.5, 0.5])),
                vec![3.0, 0.0, 0.0],
                vec![3.0, 0.5, 0.5],
            ),
            (
                Box::new(ElemWiseBuffersAddInstruction::new(vec![0, 2, 4], 6, 2)),
                vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0, 0.0, 0.0],
                vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0, 111.0, 222.0],
            ),
            (
                Box::new(ElemWiseBuffersMulInstruction::new(vec![0, 2], 4, 2)),
                vec![2.0, 3.0, 4.0, 5.0, 0.0, 0.0],
                vec![2.0, 3.0, 4.0, 5.0, 8.0, 15.0],
            ),
            (
                Box::new(ReduceSumInstruction::new(1, 4, 3)),
                vec![9.0, -1.0, 2.0, 4.0, 0.0],
                vec![9.0, -1.0, 2.0, 4.0, 5.0],
            ),
            (
                Box::new(AddBufferHeadsInstruction::new(0, 4, 6, 4, 2).unwrap()),
                vec![1.0, 2.0, -3.0, 4.0, 10.0, -20.0, 0.0, 0.0, 0.0, 0.0],
                vec![1.0, 2.0, -3.0, 4.0, 10.0, -20.0, 11.0, 12.0, -23.0, -16.0],
            ),
            (
                Box::new(MultiplyBufferHeadsInstruction::new(0, 4, 5, 4, 1).unwrap()),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 10.0, 15.0, 20.0],
            ),
        ];
        for (i, (instruction, input, expected)) in cases.into_iter().enumerate() {
            assert_eq!(
                run(instruction.as_ref(), input.clone()),
                expected,
                "case {i}"
            );
            let cloned = instruction
                .clone_box()
                .expect("built-in instructions clone");
            assert_eq!(run(cloned.as_ref(), input), expected, "clone of case {i}");
        }
    }

    #[test]
    fn construction_and_runtime_errors() {
        for (data, heads) in [(7, 3), (0, 0)] {
            assert!(matches!(
                AddBufferHeadsInstruction::new(0, 0, 0, data, heads),
                Err(InstructionModelError::InvalidBufferHeadsSize { .. })
            ));
            assert!(MultiplyBufferHeadsInstruction::new(0, 0, 0, data, heads).is_err());
        }
        assert!(matches!(
            ElemWiseBuffersAddInstruction::new(vec![0], 1, 1).apply(&mut [0.0; 2]),
            Err(InstructionModelError::InsufficientInputBuffers)
        ));
        assert!(matches!(
            CopyInstruction::new(0, 1, 1).set_dense_params(&[], &[]),
            Err(InstructionModelError::ThetaUnsupported { .. })
        ));
    }
}
