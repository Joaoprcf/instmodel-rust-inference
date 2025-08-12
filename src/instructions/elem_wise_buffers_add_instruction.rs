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
        if self.input_ptrs.len() < 2 {
            return Err(InstructionModelError::InsufficientInputBuffers);
        }

        // Optimized implementation for exactly 2 input buffers (most common case)
        if self.input_ptrs.len() == 2 {
            let input1_start = self.input_ptrs[0];
            let input2_start = self.input_ptrs[1];
            let output_start = self.output_ptr;

            // Simple direct loop - matches manual implementation pattern
            for i in 0..self.data_size {
                unified_computation_buffer[output_start + i] = unified_computation_buffer
                    [input1_start + i]
                    + unified_computation_buffer[input2_start + i];
            }
        } else {
            // Fallback for more than 2 input buffers
            for i in 0..self.data_size {
                let mut sum = unified_computation_buffer[self.input_ptrs[0] + i];
                for &input_ptr in &self.input_ptrs[1..] {
                    sum += unified_computation_buffer[input_ptr + i];
                }
                unified_computation_buffer[self.output_ptr + i] = sum;
            }
        }

        Ok(())
    }
}
