//! Dot product instruction implementation.
//!
//! Represents an instruction that performs a complete dot product operation,
//! similar to the Dense layer in deep learning frameworks such as Keras.

use crate::activation::Activation;
use crate::errors::InstructionModelError;
use crate::instructions::{Instruction, activation_instruction::ActivationInstruction};

/// Represents an instruction that performs a complete dot product operation.
/// This operation comprises three sequential steps:
/// 1. Matrix-vector multiplication: computing the dot product of an input vector with a weight matrix.
/// 2. Addition of a bias vector: adding a bias value to each computed output element.
/// 3. Application of an activation function: transforming the resulting values using the specified activation.
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

    /// Performs the forward pass computation: a dot product followed by a bias addition.
    fn apply_forward_pass(&self, unified_computation_buffer: &mut [f32]) {
        // OPTIMIZATION: Use local accumulation instead of repeated memory writes
        // Perform matrix-vector multiplication with local accumulation (like manual implementation)
        for (i, (weights_row, &bias_value)) in self.weights.iter().zip(self.bias.iter()).enumerate()
        {
            let mut sum = bias_value; // Start with bias

            // Accumulate dot product locally to minimize memory writes
            for (j, &weight) in weights_row.iter().enumerate() {
                sum += weight * unified_computation_buffer[self.input_ptr + j];
            }

            // Single write to output
            unified_computation_buffer[self.output_ptr + i] = sum;
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
