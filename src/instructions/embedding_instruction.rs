//! EMBEDDING.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;
use crate::utils::dot::Dense;

/// Table lookup: writes row `round(input)` of a `[rows][width]` weight slot to the output.
///
/// It keeps the slot's bias as well, unused here, so that θ reads and writes through this
/// instruction stay complete when a tied readout DOT shares the slot.
#[derive(Clone)]
pub struct EmbeddingInstruction {
    id_ptr: usize,
    output_ptr: usize,
    data_size: usize,
    table: Dense,
}

impl EmbeddingInstruction {
    /// `id_ptr` is the absolute buffer position of the id; `data_size` is the row width.
    pub fn new(
        id_ptr: usize,
        output_ptr: usize,
        data_size: usize,
        weights: &[Vec<f32>],
        bias: &[f32],
    ) -> Self {
        Self {
            id_ptr,
            output_ptr,
            data_size,
            table: Dense::new(weights, bias),
        }
    }
}

impl Instruction for EmbeddingInstruction {
    super::instruction_common!();

    fn apply(&self, buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        let value = buffer[self.id_ptr];
        let rows = self.table.rows();
        // Ties round to even, as NumPy's `rint` does; NaN fails the range test.
        let id = value.round_ties_even();
        let row = if id >= 0.0 && id < rows as f32 {
            self.table.row(id as usize)
        } else {
            None
        };
        let row = row.ok_or(InstructionModelError::EmbeddingIdOutOfRange { value, rows })?;
        buffer[self.output_ptr..self.output_ptr + self.data_size].copy_from_slice(row);
        Ok(())
    }

    fn set_dense_params(
        &mut self,
        weights: &[f32],
        bias: &[f32],
    ) -> Result<(), InstructionModelError> {
        self.table.set(weights, bias)
    }

    fn read_dense_params(
        &self,
        weights: &mut [f32],
        bias: &mut [f32],
    ) -> Result<(), InstructionModelError> {
        self.table.read(weights, bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> EmbeddingInstruction {
        let weights = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        EmbeddingInstruction::new(0, 1, 2, &weights, &[0.0; 3])
    }

    #[test]
    fn looks_up_rounded_rows() {
        // (id, expected row): exact ids, rounding, and ties to even.
        for (id, expected) in [
            (0.0, [1.0, 2.0]),
            (2.0, [5.0, 6.0]),
            (1.4, [3.0, 4.0]),
            (0.5, [1.0, 2.0]),
            (1.5, [5.0, 6.0]),
            (-0.4, [1.0, 2.0]),
        ] {
            let mut buffer = [id, 0.0, 0.0];
            table().apply(&mut buffer).unwrap();
            assert_eq!(buffer[1..], expected, "id {id}");
        }
    }

    #[test]
    fn rejects_ids_outside_the_table() {
        for id in [-1.0, 3.0, 2.6, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut buffer = [id, 7.0, 7.0];
            assert!(
                matches!(
                    table().apply(&mut buffer),
                    Err(InstructionModelError::EmbeddingIdOutOfRange { rows: 3, .. })
                ),
                "id {id}"
            );
            assert_eq!(buffer[1..], [7.0, 7.0]);
        }
    }

    #[test]
    fn dense_params_round_trip_including_bias() {
        let mut instruction = table();
        let weights: Vec<f32> = (0..6).map(|v| v as f32).collect();
        instruction
            .set_dense_params(&weights, &[9.0, 8.0, 7.0])
            .unwrap();
        let (mut read_weights, mut read_bias) = (vec![0.0; 6], vec![0.0; 3]);
        instruction
            .read_dense_params(&mut read_weights, &mut read_bias)
            .unwrap();
        assert_eq!((read_weights, read_bias), (weights, vec![9.0, 8.0, 7.0]));
        let mut buffer = [1.0, 0.0, 0.0];
        instruction.apply(&mut buffer).unwrap();
        assert_eq!(buffer[1..], [2.0, 3.0]);
    }
}
