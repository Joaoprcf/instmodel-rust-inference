//! REDUCE_SUM.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Writes the sum of the input buffer into one output slot.
#[derive(Clone)]
pub struct ReduceSumInstruction {
    input_ptr: usize,
    output_ptr: usize,
    input_size: usize,
}

impl ReduceSumInstruction {
    pub fn new(input_ptr: usize, output_ptr: usize, input_size: usize) -> Self {
        Self {
            input_ptr,
            output_ptr,
            input_size,
        }
    }
}

impl Instruction for ReduceSumInstruction {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        1
    }

    fn apply(&self, buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        buffer[self.output_ptr] = buffer[self.input_ptr..self.input_ptr + self.input_size]
            .iter()
            .sum();
        Ok(())
    }

    fn clone_box(&self) -> Option<Box<dyn Instruction>> {
        Some(Box::new(self.clone()))
    }
}
