//! Element-wise buffers multiplication instruction implementation.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Represents an element-wise multiplication operation that multiplies multiple input buffers.
pub struct ElemWiseBuffersMulInstruction {
    input_ptrs: Vec<usize>,
    output_ptr: usize,
    data_size: usize,
}

impl ElemWiseBuffersMulInstruction {
    pub fn new(input_ptrs: Vec<usize>, output_ptr: usize, data_size: usize) -> Self {
        Self {
            input_ptrs,
            output_ptr,
            data_size,
        }
    }
}

impl Instruction for ElemWiseBuffersMulInstruction {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        self.data_size
    }

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        // Initialize output buffer with ones for multiplication
        for i in 0..self.data_size {
            unified_computation_buffer[self.output_ptr + i] = 1.0;
        }

        // Multiply all input buffers to the output buffer
        for &input_ptr in &self.input_ptrs {
            for i in 0..self.data_size {
                unified_computation_buffer[self.output_ptr + i] *=
                    unified_computation_buffer[input_ptr + i];
            }
        }

        Ok(())
    }
}
