//! Element-wise multiplication instruction implementation.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Represents an instruction that performs element-wise multiplication with parameters.
pub struct ElemWiseMulInstruction {
    output_ptr: usize,
    data_size: usize,
    parameters: Vec<f32>,
}

impl ElemWiseMulInstruction {
    pub fn new(output_ptr: usize, data_size: usize, parameters: &[f32]) -> Self {
        Self {
            output_ptr,
            data_size,
            parameters: parameters.to_vec(),
        }
    }
}

impl Instruction for ElemWiseMulInstruction {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        self.data_size
    }

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        for i in 0..self.data_size {
            unified_computation_buffer[self.output_ptr + i] *= self.parameters[i];
        }
        Ok(())
    }
}
