//! COPY_MASKED.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Gathers the selected input positions into consecutive output slots.
#[derive(Clone)]
pub struct CopyMaskedInstruction {
    output_ptr: usize,
    input_pointers: Vec<usize>,
}

impl CopyMaskedInstruction {
    pub fn new(input_ptr: usize, output_ptr: usize, indexes: &[usize]) -> Self {
        Self {
            output_ptr,
            input_pointers: indexes.iter().map(|&idx| input_ptr + idx).collect(),
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

    fn apply(&self, buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        for (i, &input_ptr) in self.input_pointers.iter().enumerate() {
            buffer[self.output_ptr + i] = buffer[input_ptr];
        }
        Ok(())
    }

    fn clone_box(&self) -> Option<Box<dyn Instruction>> {
        Some(Box::new(self.clone()))
    }
}
