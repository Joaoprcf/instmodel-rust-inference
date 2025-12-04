//! Attention instruction implementation.
//!
//! Implements a softmax attention mechanism that computes:
//! 1. Linear transform on key buffer: key @ weights.T + bias
//! 2. Softmax normalization (numerically stable)
//! 3. Element-wise multiplication with the query/value buffer

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Instruction that performs a softmax attention operation.
///
/// This instruction takes a query buffer and a key buffer, applies a linear
/// transformation to the key, normalizes with softmax, and multiplies
/// element-wise with the query buffer.
pub struct AttentionInstruction {
    query_ptr: usize,
    key_ptr: usize,
    output_ptr: usize,
    data_size: usize,
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

impl AttentionInstruction {
    pub fn new(
        query_ptr: usize,
        key_ptr: usize,
        output_ptr: usize,
        data_size: usize,
        weights: &[Vec<f32>],
        bias: &[f32],
    ) -> Self {
        Self {
            query_ptr,
            key_ptr,
            output_ptr,
            data_size,
            weights: weights.to_vec(),
            bias: bias.to_vec(),
        }
    }

    #[inline(always)]
    fn apply_linear_transform(&self, buffer: &mut [f32]) {
        let key_start = self.key_ptr;
        let output_start = self.output_ptr;

        for (row_index, (weights_row, &bias_value)) in
            self.weights.iter().zip(self.bias.iter()).enumerate()
        {
            let mut sum = bias_value;
            for (col_index, &weight) in weights_row.iter().enumerate() {
                sum += weight * buffer[key_start + col_index];
            }
            buffer[output_start + row_index] = sum;
        }
    }

    #[inline(always)]
    fn apply_softmax(&self, buffer: &mut [f32]) {
        let output_start = self.output_ptr;
        let output_slice = &mut buffer[output_start..output_start + self.data_size];

        let max_val = output_slice
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut sum = 0.0f32;
        for val in output_slice.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }

        for val in output_slice.iter_mut() {
            *val /= sum;
        }
    }

    #[inline(always)]
    fn apply_elementwise_multiply(&self, buffer: &mut [f32]) {
        let query_start = self.query_ptr;
        let output_start = self.output_ptr;

        for i in 0..self.data_size {
            buffer[output_start + i] *= buffer[query_start + i];
        }
    }
}

impl Instruction for AttentionInstruction {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        self.data_size
    }

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        self.apply_linear_transform(unified_computation_buffer);
        self.apply_softmax(unified_computation_buffer);
        self.apply_elementwise_multiply(unified_computation_buffer);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DELTA: f32 = 1e-5;

    #[test]
    fn test_attention_basic() {
        // Simple 2-element attention
        // Query: [1.0, 2.0], Key: [0.5, 0.5]
        // Weights: identity matrix, Bias: [0.0, 0.0]
        // Linear: [0.5, 0.5] -> Softmax: [0.5, 0.5] -> Multiply: [0.5, 1.0]
        let mut buffer = vec![
            1.0, 2.0, // query (input) at index 0
            0.5, 0.5, // key at index 2
            0.0, 0.0, // output at index 4
        ];

        let weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let bias = vec![0.0, 0.0];

        let instruction = AttentionInstruction::new(0, 2, 4, 2, &weights, &bias);
        instruction.apply(&mut buffer).unwrap();

        // After linear: [0.5, 0.5]
        // After softmax: [0.5, 0.5] (equal values -> equal probabilities)
        // After multiply: [0.5 * 1.0, 0.5 * 2.0] = [0.5, 1.0]
        assert!((buffer[4] - 0.5).abs() < DELTA);
        assert!((buffer[5] - 1.0).abs() < DELTA);
    }

    #[test]
    fn test_attention_with_bias() {
        let mut buffer = vec![
            1.0, 1.0, // query
            1.0, 1.0, // key
            0.0, 0.0, // output
        ];

        let weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let bias = vec![1.0, -1.0]; // bias shifts the values

        let instruction = AttentionInstruction::new(0, 2, 4, 2, &weights, &bias);
        instruction.apply(&mut buffer).unwrap();

        // After linear: [1.0 + 1.0, 1.0 - 1.0] = [2.0, 0.0]
        // Softmax([2.0, 0.0]): exp(2-2)/sum, exp(0-2)/sum = exp(0)/(exp(0)+exp(-2)), exp(-2)/(exp(0)+exp(-2))
        let exp_0 = 1.0f32;
        let exp_neg2 = (-2.0f32).exp();
        let sum = exp_0 + exp_neg2;
        let softmax_0 = exp_0 / sum;
        let softmax_1 = exp_neg2 / sum;

        // After multiply: [softmax_0 * 1.0, softmax_1 * 1.0]
        assert!((buffer[4] - softmax_0).abs() < DELTA);
        assert!((buffer[5] - softmax_1).abs() < DELTA);
    }

    #[test]
    fn test_attention_softmax_numerical_stability() {
        // Test with large values that could cause overflow without numerical stability
        let mut buffer = vec![
            1.0, 1.0, // query
            100.0, 100.0, // key with large values
            0.0, 0.0, // output
        ];

        let weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let bias = vec![0.0, 0.0];

        let instruction = AttentionInstruction::new(0, 2, 4, 2, &weights, &bias);
        instruction.apply(&mut buffer).unwrap();

        // Equal inputs -> softmax should give [0.5, 0.5]
        assert!((buffer[4] - 0.5).abs() < DELTA);
        assert!((buffer[5] - 0.5).abs() < DELTA);
    }
}
