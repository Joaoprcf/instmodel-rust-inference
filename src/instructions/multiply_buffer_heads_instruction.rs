//! Multiply buffer heads instruction implementation.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Represents a head-wise multiplication operation.
///
/// Each head value from the heads buffer is multiplied across its corresponding
/// segment of the data buffer. The head dimension is precomputed at construction
/// time for maximum efficiency.
pub struct MultiplyBufferHeadsInstruction {
    data_ptr: usize,
    heads_ptr: usize,
    output_ptr: usize,
    data_size: usize,
    num_heads: usize,
    head_dim: usize,
}

impl MultiplyBufferHeadsInstruction {
    pub fn new(
        data_ptr: usize,
        heads_ptr: usize,
        output_ptr: usize,
        data_size: usize,
        heads_size: usize,
    ) -> Result<Self, InstructionModelError> {
        if !data_size.is_multiple_of(heads_size) {
            return Err(InstructionModelError::InvalidBufferHeadsSize {
                data_size,
                heads_size,
            });
        }

        let head_dim = data_size / heads_size;

        Ok(Self {
            data_ptr,
            heads_ptr,
            output_ptr,
            data_size,
            num_heads: heads_size,
            head_dim,
        })
    }
}

impl Instruction for MultiplyBufferHeadsInstruction {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        self.data_size
    }

    fn apply(&self, buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        let mut idx = 0;
        for head in 0..self.num_heads {
            let head_value = buffer[self.heads_ptr + head];
            for _ in 0..self.head_dim {
                buffer[self.output_ptr + idx] = buffer[self.data_ptr + idx] * head_value;
                idx += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiply_buffer_heads_basic() {
        // Data: 8 elements, Heads: 2 elements, head_dim = 4
        // Buffer layout: [data(8), heads(2), output(8)]
        let mut buffer = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // data
            2.0, 3.0, // heads
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // output
        ];

        let instruction = MultiplyBufferHeadsInstruction::new(0, 8, 10, 8, 2).unwrap();
        instruction.apply(&mut buffer).unwrap();

        // First 4 elements multiplied by 2.0, next 4 by 3.0
        let expected = vec![2.0, 4.0, 6.0, 8.0, 15.0, 18.0, 21.0, 24.0];
        assert_eq!(&buffer[10..18], &expected[..]);
    }

    #[test]
    fn test_multiply_buffer_heads_four_heads() {
        // Data: 20 elements, Heads: 4 elements, head_dim = 5
        // Buffer layout: [data(20), heads(4), output(20)]
        let data: Vec<f32> = (1..=20).map(|x| x as f32).collect();
        let heads = vec![2.0, 3.0, 4.0, 5.0];
        let output = vec![0.0; 20];

        let mut buffer = Vec::new();
        buffer.extend(&data);
        buffer.extend(&heads);
        buffer.extend(&output);

        let instruction = MultiplyBufferHeadsInstruction::new(0, 20, 24, 20, 4).unwrap();
        instruction.apply(&mut buffer).unwrap();

        // First 5 * 2, next 5 * 3, next 5 * 4, last 5 * 5
        let expected: Vec<f32> = vec![
            2.0, 4.0, 6.0, 8.0, 10.0, // * 2
            18.0, 21.0, 24.0, 27.0, 30.0, // * 3
            44.0, 48.0, 52.0, 56.0, 60.0, // * 4
            80.0, 85.0, 90.0, 95.0, 100.0, // * 5
        ];
        assert_eq!(&buffer[24..44], &expected[..]);
    }

    #[test]
    fn test_multiply_buffer_heads_invalid_size() {
        // Data size 7 is not divisible by heads size 3
        let result = MultiplyBufferHeadsInstruction::new(0, 7, 10, 7, 3);
        assert!(result.is_err());
        match result {
            Err(InstructionModelError::InvalidBufferHeadsSize {
                data_size,
                heads_size,
            }) => {
                assert_eq!(data_size, 7);
                assert_eq!(heads_size, 3);
            }
            _ => panic!("Expected InvalidBufferHeadsSize error"),
        }
    }

    #[test]
    fn test_multiply_buffer_heads_single_head() {
        // Data: 4 elements, Heads: 1 element (broadcasts to all)
        let mut buffer = vec![
            1.0, 2.0, 3.0, 4.0, // data
            5.0, // head (single)
            0.0, 0.0, 0.0, 0.0, // output
        ];

        let instruction = MultiplyBufferHeadsInstruction::new(0, 4, 5, 4, 1).unwrap();
        instruction.apply(&mut buffer).unwrap();

        let expected = vec![5.0, 10.0, 15.0, 20.0];
        assert_eq!(&buffer[5..9], &expected[..]);
    }
}
