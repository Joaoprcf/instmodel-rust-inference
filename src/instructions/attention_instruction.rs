use crate::errors::InstructionModelError;
use crate::instructions::Instruction;
use crate::utils::dot::Dense;

/// `output = softmax(key @ W.T + b) * query`.
#[derive(Clone)]
pub struct AttentionInstruction {
    query_ptr: usize,
    key_ptr: usize,
    output_ptr: usize,
    data_size: usize,
    dense: Dense,
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
            dense: Dense::new(weights, bias),
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

    fn apply(&self, buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        let out = self.dense.forward(buffer, self.key_ptr, self.output_ptr);
        // Subtracting the max keeps exp() from overflowing.
        let max_val = out.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0f32;
        for val in out.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }
        for val in out.iter_mut() {
            *val /= sum;
        }
        for i in 0..self.data_size {
            buffer[self.output_ptr + i] *= buffer[self.query_ptr + i];
        }
        Ok(())
    }

    fn set_dense_params(
        &mut self,
        weights: &[f32],
        bias: &[f32],
    ) -> Result<(), InstructionModelError> {
        self.dense.set(weights, bias)
    }

    fn read_dense_params(
        &self,
        weights: &mut [f32],
        bias: &mut [f32],
    ) -> Result<(), InstructionModelError> {
        self.dense.read(weights, bias)
    }

    fn clone_box(&self) -> Option<Box<dyn Instruction>> {
        Some(Box::new(self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn softmax(x: &[f32]) -> Vec<f32> {
        let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let e: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = e.iter().sum();
        e.iter().map(|v| v / sum).collect()
    }

    #[test]
    fn computes_softmax_attention() {
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        // (query, key, bias): plain, biased, and large logits that would overflow exp() unshifted.
        for (query, key, bias) in [
            ([1.0, 2.0], [0.5, 0.5], [0.0, 0.0]),
            ([1.0, 1.0], [1.0, 1.0], [1.0, -1.0]),
            ([1.0, 1.0], [100.0, 100.0], [0.0, 0.0]),
        ] {
            let mut buffer = [query[0], query[1], key[0], key[1], 0.0, 0.0];
            AttentionInstruction::new(0, 2, 4, 2, &identity, &bias)
                .apply(&mut buffer)
                .unwrap();
            let p = softmax(&[key[0] + bias[0], key[1] + bias[1]]);
            for i in 0..2 {
                assert!((buffer[4 + i] - p[i] * query[i]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn key_aliasing_output_reads_original_key() {
        let swap = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let mut buffer = [1.0, 1.0, 3.0, 1.0];
        AttentionInstruction::new(0, 2, 2, 2, &swap, &[0.0, 0.0])
            .apply(&mut buffer)
            .unwrap();
        let p = softmax(&[1.0, 3.0]);
        assert!((buffer[2] - p[0]).abs() < 1e-6 && (buffer[3] - p[1]).abs() < 1e-6);
    }
}
