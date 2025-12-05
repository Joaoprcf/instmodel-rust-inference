//! Reduce sum instruction implementation.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Represents a reduce sum operation that sums all values in an input buffer.
///
/// This instruction computes the sum of all elements in the input buffer
/// and stores the result as a single value in the output buffer.
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

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        let sum: f32 = unified_computation_buffer[self.input_ptr..self.input_ptr + self.input_size]
            .iter()
            .sum();
        unified_computation_buffer[self.output_ptr] = sum;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_sum_basic() {
        let mut buffer = vec![1.0, 2.0, 3.0, 4.0, 0.0];
        let instruction = ReduceSumInstruction::new(0, 4, 4);
        instruction.apply(&mut buffer).unwrap();
        assert!((buffer[4] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_sum_partial() {
        let mut buffer = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0];
        let instruction = ReduceSumInstruction::new(1, 5, 3);
        instruction.apply(&mut buffer).unwrap();
        assert!((buffer[5] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_sum_single_element() {
        let mut buffer = vec![42.0, 0.0];
        let instruction = ReduceSumInstruction::new(0, 1, 1);
        instruction.apply(&mut buffer).unwrap();
        assert!((buffer[1] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_sum_negative_values() {
        let mut buffer = vec![-1.0, 2.0, -3.0, 4.0, 0.0];
        let instruction = ReduceSumInstruction::new(0, 4, 4);
        instruction.apply(&mut buffer).unwrap();
        assert!((buffer[4] - 2.0).abs() < 1e-6);
    }
}
