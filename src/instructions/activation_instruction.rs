//! ACTIVATION.

use crate::activation::Activation;
use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Applies an activation in place over one buffer.
#[derive(Clone)]
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
    super::instruction_common!();

    fn apply(&self, buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        self.activation
            .apply_in_place(&mut buffer[self.output_ptr..self.output_ptr + self.data_size]);
        Ok(())
    }
}
