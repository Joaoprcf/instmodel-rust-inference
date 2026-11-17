//! LAYER_NORM.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// `(x - mean) / sqrt(var + epsilon) * gain + bias` over one buffer (population variance).
#[derive(Clone)]
pub struct LayerNormInstruction {
    input_ptr: usize,
    output_ptr: usize,
    data_size: usize,
    gain: Vec<f32>,
    bias: Vec<f32>,
    epsilon: f32,
}

impl LayerNormInstruction {
    /// `gain` and `bias` hold `data_size` values each. Input and output are the same
    /// buffer or disjoint ones.
    pub fn new(
        input_ptr: usize,
        output_ptr: usize,
        data_size: usize,
        gain: &[f32],
        bias: &[f32],
        epsilon: f32,
    ) -> Self {
        Self {
            input_ptr,
            output_ptr,
            data_size,
            gain: gain.to_vec(),
            bias: bias.to_vec(),
            epsilon,
        }
    }
}

impl Instruction for LayerNormInstruction {
    super::instruction_common!();

    fn apply(&self, buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        let n = self.data_size as f32;
        let input = &buffer[self.input_ptr..self.input_ptr + self.data_size];
        let mean = input.iter().sum::<f32>() / n;
        let variance = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
        let scale = (variance + self.epsilon).sqrt();
        // Element i is read before it is written, so an in-place norm needs no copy.
        for (i, (&gain, &bias)) in self.gain.iter().zip(&self.bias).enumerate() {
            let centered = buffer[self.input_ptr + i] - mean;
            buffer[self.output_ptr + i] = centered / scale * gain + bias;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_with_gain_and_bias() {
        // x = [1, 2, 3, 6]: mean 3, variance (4 + 1 + 0 + 9) / 4 = 3.5.
        let scale = (3.5f32 + 0.5).sqrt(); // = 2 with epsilon 0.5
        let instruction =
            LayerNormInstruction::new(0, 4, 4, &[1.0, 2.0, 1.0, -1.0], &[0.0, 0.0, 1.0, 0.5], 0.5);
        let mut buffer = [1.0, 2.0, 3.0, 6.0, 0.0, 0.0, 0.0, 0.0];
        instruction.apply(&mut buffer).unwrap();
        assert_eq!(scale, 2.0);
        assert_eq!(buffer[4..], [-1.0, -1.0, 1.0, -1.0]);

        let mut in_place = [1.0, 2.0, 3.0, 6.0];
        LayerNormInstruction::new(0, 0, 4, &[1.0; 4], &[0.0; 4], 0.5)
            .apply(&mut in_place)
            .unwrap();
        assert_eq!(in_place, [-1.0, -0.5, 0.0, 1.5]);
    }

    #[test]
    fn constant_input_maps_to_bias() {
        let mut buffer = [5.0, 5.0, 0.0, 0.0];
        LayerNormInstruction::new(0, 2, 2, &[3.0, 3.0], &[0.25, -0.25], 1e-5)
            .apply(&mut buffer)
            .unwrap();
        assert_eq!(buffer[2..], [0.25, -0.25]);
    }
}
