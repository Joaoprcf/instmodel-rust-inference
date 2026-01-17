//! Add buffer heads instruction implementation.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Represents a head-wise addition operation.
///
/// Each head value from the heads buffer is added to its corresponding
/// segment of the data buffer. The head dimension is precomputed at construction
/// time for maximum efficiency.
pub struct AddBufferHeadsInstruction {
    data_ptr: usize,
    heads_ptr: usize,
    output_ptr: usize,
    data_size: usize,
    num_heads: usize,
    head_dim: usize,
}

impl AddBufferHeadsInstruction {
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

impl Instruction for AddBufferHeadsInstruction {
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
                buffer[self.output_ptr + idx] = buffer[self.data_ptr + idx] + head_value;
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
    fn test_add_buffer_heads_basic() {
        // Data: 8 elements, Heads: 2 elements, head_dim = 4
        // Buffer layout: [data(8), heads(2), output(8)]
        let mut buffer = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // data
            10.0, 20.0, // heads
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // output
        ];

        let instruction = AddBufferHeadsInstruction::new(0, 8, 10, 8, 2).unwrap();
        instruction.apply(&mut buffer).unwrap();

        // First 4 elements + 10.0, next 4 + 20.0
        let expected = vec![11.0, 12.0, 13.0, 14.0, 25.0, 26.0, 27.0, 28.0];
        assert_eq!(&buffer[10..18], &expected[..]);
    }

    #[test]
    fn test_add_buffer_heads_four_heads() {
        // Data: 20 elements, Heads: 4 elements, head_dim = 5
        // Buffer layout: [data(20), heads(4), output(20)]
        let data: Vec<f32> = (1..=20).map(|x| x as f32).collect();
        let heads = vec![100.0, 200.0, 300.0, 400.0];
        let output = vec![0.0; 20];

        let mut buffer = Vec::new();
        buffer.extend(&data);
        buffer.extend(&heads);
        buffer.extend(&output);

        let instruction = AddBufferHeadsInstruction::new(0, 20, 24, 20, 4).unwrap();
        instruction.apply(&mut buffer).unwrap();

        // First 5 + 100, next 5 + 200, next 5 + 300, last 5 + 400
        let expected: Vec<f32> = vec![
            101.0, 102.0, 103.0, 104.0, 105.0, // + 100
            206.0, 207.0, 208.0, 209.0, 210.0, // + 200
            311.0, 312.0, 313.0, 314.0, 315.0, // + 300
            416.0, 417.0, 418.0, 419.0, 420.0, // + 400
        ];
        assert_eq!(&buffer[24..44], &expected[..]);
    }

    #[test]
    fn test_add_buffer_heads_invalid_size() {
        // Data size 7 is not divisible by heads size 3
        let result = AddBufferHeadsInstruction::new(0, 7, 10, 7, 3);
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
    fn test_add_buffer_heads_single_head() {
        // Data: 4 elements, Heads: 1 element (broadcasts to all)
        let mut buffer = vec![
            1.0, 2.0, 3.0, 4.0,  // data
            10.0, // head (single)
            0.0, 0.0, 0.0, 0.0, // output
        ];

        let instruction = AddBufferHeadsInstruction::new(0, 4, 5, 4, 1).unwrap();
        instruction.apply(&mut buffer).unwrap();

        let expected = vec![11.0, 12.0, 13.0, 14.0];
        assert_eq!(&buffer[5..9], &expected[..]);
    }

    #[test]
    fn test_add_buffer_heads_negative_values() {
        // Data: 6 elements, Heads: 2 elements, head_dim = 3
        let mut buffer = vec![
            -1.0, -2.0, -3.0, 4.0, 5.0, 6.0, // data
            5.0, -10.0, // heads
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // output
        ];

        let instruction = AddBufferHeadsInstruction::new(0, 6, 8, 6, 2).unwrap();
        instruction.apply(&mut buffer).unwrap();

        let expected = vec![4.0, 3.0, 2.0, -6.0, -5.0, -4.0];
        assert_eq!(&buffer[8..14], &expected[..]);
    }
}
