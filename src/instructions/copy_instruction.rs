//! COPY.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Copies `data_size` values from `input_ptr` to `output_ptr`.
#[derive(Clone)]
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
    super::instruction_common!();

    fn apply(&self, buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        buffer.copy_within(
            self.input_ptr..self.input_ptr + self.data_size,
            self.output_ptr,
        );
        Ok(())
    }
}
