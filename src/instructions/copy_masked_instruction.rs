//! Copy masked instruction implementation.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Represents an instruction that copies data from specific indexed positions.
pub struct CopyMaskedInstruction {
    output_ptr: usize,
    input_pointers: Vec<usize>,
}

impl CopyMaskedInstruction {
    pub fn new(input_ptr: usize, output_ptr: usize, indexes: &[usize]) -> Self {
        let input_pointers: Vec<usize> = indexes.iter().map(|&idx| input_ptr + idx).collect();

        Self {
            output_ptr,
            input_pointers,
        }
    }
}

impl Instruction for CopyMaskedInstruction {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        self.input_pointers.len()
    }

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        for (i, &input_ptr) in self.input_pointers.iter().enumerate() {
            unified_computation_buffer[self.output_ptr + i] = unified_computation_buffer[input_ptr];
        }
        Ok(())
    }
}
