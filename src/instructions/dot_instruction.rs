//! Dot product instruction implementation.
//!
//! Represents an instruction that performs a complete dot product operation,
//! similar to the Dense layer in common deep learning frameworks.

use crate::activation::Activation;
use crate::errors::InstructionModelError;
use crate::instructions::{Instruction, activation_instruction::ActivationInstruction};

/// Instruction that performs a dense (matrix-vector) operation followed by bias and activation.
///
/// The implementation focuses on minimizing repeated bounds calculations inside the hot loop while
/// keeping the code safe (no `unsafe` blocks).
pub struct DotInstruction {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    input_ptr: usize,
    output_ptr: usize,
    data_size: usize,
    activation_instruction: Option<ActivationInstruction>,
}

impl DotInstruction {
    /// Creates a new DotInstruction.
    pub fn new(
        input_ptr: usize,
        output_ptr: usize,
        data_size: usize,
        weights: &[Vec<f32>],
        bias: &[f32],
        activation: Option<Activation>,
    ) -> Result<Self, InstructionModelError> {
        let activation_instruction =
            activation.map(|act| ActivationInstruction::new(act, output_ptr, data_size));

        Ok(Self {
            weights: weights.to_vec(),
            bias: bias.to_vec(),
            input_ptr,
            output_ptr,
            data_size,
            activation_instruction,
        })
    }

    #[inline(always)]
    fn apply_forward_pass(&self, unified_computation_buffer: &mut [f32]) {
        let input_start = self.input_ptr;
        let output_start = self.output_ptr;

        for (row_index, (weights_row, &bias_value)) in
            self.weights.iter().zip(self.bias.iter()).enumerate()
        {
            let mut sum = bias_value;
            for (col_index, &weight) in weights_row.iter().enumerate() {
                sum += weight * unified_computation_buffer[input_start + col_index];
            }
            unified_computation_buffer[output_start + row_index] = sum;
        }
    }
}

impl Instruction for DotInstruction {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        self.data_size
    }

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        self.apply_forward_pass(unified_computation_buffer);

        if let Some(ref activation_instruction) = self.activation_instruction {
            activation_instruction.apply(unified_computation_buffer)?;
        }

        Ok(())
    }
}
