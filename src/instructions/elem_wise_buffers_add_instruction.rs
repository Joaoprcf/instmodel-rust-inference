//! Element-wise buffers addition instruction implementation.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Represents an element-wise addition operation that sums multiple input buffers.
pub struct ElemWiseBuffersAddInstruction {
    input_ptrs: Vec<usize>,
    output_ptr: usize,
    data_size: usize,
}

impl ElemWiseBuffersAddInstruction {
    pub fn new(input_ptrs: Vec<usize>, output_ptr: usize, data_size: usize) -> Self {
        Self {
            input_ptrs,
            output_ptr,
            data_size,
        }
    }
}

impl Instruction for ElemWiseBuffersAddInstruction {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        self.data_size
    }

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        // Initialize output buffer with zeros
        for i in 0..self.data_size {
            unified_computation_buffer[self.output_ptr + i] = 0.0;
        }

        // Add all input buffers to the output buffer
        for &input_ptr in &self.input_ptrs {
            for i in 0..self.data_size {
                unified_computation_buffer[self.output_ptr + i] +=
                    unified_computation_buffer[input_ptr + i];
            }
        }

        Ok(())
    }
}
