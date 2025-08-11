//! Copy instruction implementation.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Represents an instruction that copies data from one buffer location to another.
pub struct CopyInstruction {
    input_ptr: usize,
    output_ptr: usize,
    data_size: usize,
}

impl CopyInstruction {
    pub fn new(input_ptr: usize, output_ptr: usize, data_size: usize) -> Self {
        Self {
            input_ptr,
            output_ptr,
            data_size,
        }
    }
}

impl Instruction for CopyInstruction {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        self.data_size
    }

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        for i in 0..self.data_size {
            unified_computation_buffer[self.output_ptr + i] =
                unified_computation_buffer[self.input_ptr + i];
        }
        Ok(())
    }
}
