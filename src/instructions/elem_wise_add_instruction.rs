//! Element-wise addition instruction implementation.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Represents an instruction that performs element-wise addition with parameters.
pub struct ElemWiseAddInstruction {
    output_ptr: usize,
    data_size: usize,
    parameters: Vec<f32>,
}

impl ElemWiseAddInstruction {
    pub fn new(output_ptr: usize, data_size: usize, parameters: &[f32]) -> Self {
        Self {
            output_ptr,
            data_size,
            parameters: parameters.to_vec(),
        }
    }
}

impl Instruction for ElemWiseAddInstruction {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        self.data_size
    }

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        debug_assert_eq!(self.parameters.len(), self.data_size);

        let output_start = self.output_ptr;

        // Simple direct loop - let the compiler optimize with --release
        for i in 0..self.data_size {
            unified_computation_buffer[output_start + i] += self.parameters[i];
        }

        Ok(())
    }
}
