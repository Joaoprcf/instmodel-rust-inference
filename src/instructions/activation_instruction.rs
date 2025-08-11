//! Activation instruction implementation.

use crate::activation::Activation;
use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Represents an instruction that applies an activation function to values in place.
pub struct ActivationInstruction {
    activation: Activation,
    output_ptr: usize,
    data_size: usize,
}

impl ActivationInstruction {
    pub fn new(activation: Activation, output_ptr: usize, data_size: usize) -> Self {
        Self {
            activation,
            output_ptr,
            data_size,
        }
    }
}

impl Instruction for ActivationInstruction {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        self.data_size
    }

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        let start = self.output_ptr;
        let end = start + self.data_size;
        let slice = &mut unified_computation_buffer[start..end];
        self.activation.apply_in_place(slice);
        Ok(())
    }
}
