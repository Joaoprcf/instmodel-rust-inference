//! CAUSAL_ATTENTION.

use crate::errors::InstructionModelError;
use crate::instructions::{Instruction, split_src_dst};
use crate::sequence::StepState;
use crate::utils::dot::{DotKernel, dot};

/// Causal self-attention for one token over a key/value cache slot of the sequence state.
///
/// The input buffer is `[q | k | v]` with `heads`, `kv_heads` and `kv_heads` heads of
/// `head_dim` values. The token's k and v are stored at the current position, then query
/// head `h` attends over every cached position of key/value head `h / (heads / kv_heads)`:
/// `softmax(q_h · k_t / sqrt(head_dim))`, weighting the cached v.
#[derive(Clone)]
pub struct CausalAttentionInstruction {
    input_ptr: usize,
    output_ptr: usize,
    data_size: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    slot: usize,
    scale: f32,
    kernel: DotKernel,
}

impl CausalAttentionInstruction {
    /// `kv_heads` must divide `heads`; input and output buffers must not overlap.
    pub fn new(
        input_ptr: usize,
        output_ptr: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        slot: usize,
    ) -> Self {
        Self {
            input_ptr,
            output_ptr,
            data_size: heads * head_dim,
            heads,
            kv_heads,
            head_dim,
            slot,
            scale: (head_dim as f64).powf(-0.5) as f32,
            kernel: DotKernel::detect(),
        }
    }
}

impl Instruction for CausalAttentionInstruction {
    super::instruction_common!();

    fn apply(&self, _buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        Err(InstructionModelError::MissingSequenceState {
            instruction_type: "CAUSAL_ATTENTION".to_string(),
        })
    }

    fn apply_with_state(
        &self,
        buffer: &mut [f32],
        state: &mut StepState,
    ) -> Result<(), InstructionModelError> {
        let (head_dim, row) = (self.head_dim, self.kv_heads * self.head_dim);
        let StepState {
            position,
            slots,
            scores,
            ..
        } = state;
        let cache = slots
            .get_mut(self.slot)
            .filter(|cache| cache.row == row)
            .ok_or(InstructionModelError::KvSlotIndexOutOfBounds { index: self.slot })?;
        let start = *position * row;
        let length = *position + 1;
        if start + row > cache.keys.len() || length > scores.len() {
            return Err(InstructionModelError::SequenceFull {
                max_positions: scores.len(),
            });
        }
        let input_size = self.data_size + 2 * row;
        let (input, output) = split_src_dst(
            buffer,
            self.input_ptr,
            input_size,
            self.output_ptr,
            self.data_size,
        )
        .ok_or_else(|| InstructionModelError::SameInputOutputIndexes {
            instruction_type: "CAUSAL_ATTENTION".to_string(),
        })?;
        let (queries, new_kv) = input.split_at(self.data_size);
        let (key, value) = new_kv.split_at(row);
        cache.keys[start..start + row].copy_from_slice(key);
        cache.values[start..start + row].copy_from_slice(value);

        let keys = &cache.keys[..length * row];
        let values = &cache.values[..length * row];
        let scores = &mut scores[..length];
        let group = self.heads / self.kv_heads;
        for (h, (query, out)) in queries
            .chunks_exact(head_dim)
            .zip(output.chunks_exact_mut(head_dim))
            .enumerate()
        {
            let offset = (h / group) * head_dim;
            let mut max = f32::NEG_INFINITY;
            for (score, key) in scores.iter_mut().zip(keys.chunks_exact(row)) {
                *score = dot(self.kernel, query, &key[offset..offset + head_dim]) * self.scale;
                max = max.max(*score);
            }
            // Subtracting the max keeps exp() from overflowing.
            let mut sum = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max).exp();
                sum += *score;
            }
            out.fill(0.0);
            for (&score, value) in scores.iter().zip(values.chunks_exact(row)) {
                let weight = score / sum;
                for (o, &v) in out.iter_mut().zip(&value[offset..offset + head_dim]) {
                    *o += weight * v;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One head, head_dim 2: scale 1/sqrt(2).
    fn step(
        instruction: &CausalAttentionInstruction,
        state: &mut StepState,
        qkv: &[f32],
    ) -> Vec<f32> {
        let mut buffer = qkv.to_vec();
        buffer.resize(qkv.len() + instruction.data_size, 0.0);
        instruction.apply_with_state(&mut buffer, state).unwrap();
        state.position += 1;
        buffer[qkv.len()..].to_vec()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-5, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn attends_over_the_cached_prefix() {
        let instruction = CausalAttentionInstruction::new(0, 6, 1, 1, 2, 0);
        let mut state = StepState::for_test_slots(3, &[(1, 2)]);
        // The first token sees only itself: output = v0.
        assert_close(
            &step(&instruction, &mut state, &[1.0, 0.0, 1.0, 0.0, 3.0, 4.0]),
            &[3.0, 4.0],
        );
        // q = [sqrt(2) ln 3, 0], k0 = [1, 0], k1 = [0, 1]: scores ln 3 and 0, so
        // p = [3/4, 1/4] over v0 = [3, 4] and v1 = [-1, 8].
        let q = std::f32::consts::SQRT_2 * 3.0f32.ln();
        assert_close(
            &step(&instruction, &mut state, &[q, 0.0, 0.0, 1.0, -1.0, 8.0]),
            &[2.0, 5.0],
        );
        assert_eq!(state.slots[0].keys[..4], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(state.slots[0].values[..4], [3.0, 4.0, -1.0, 8.0]);
    }

    #[test]
    fn query_heads_share_key_value_heads() {
        // heads 4, kv_heads 2, head_dim 1: heads 0-1 read kv head 0, heads 2-3 kv head 1.
        let instruction = CausalAttentionInstruction::new(0, 8, 4, 2, 1, 0);
        let mut state = StepState::for_test_slots(2, &[(2, 1)]);
        let first = step(
            &instruction,
            &mut state,
            &[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 10.0, 20.0],
        );
        assert_close(&first, &[10.0, 10.0, 20.0, 20.0]);
        // Keys: head 0 = [0, ln 3], head 1 = [0, 0]; queries 1, 0, 1, 0 (head_dim 1, scale 1).
        let ln3 = 3.0f32.ln();
        let second = step(
            &instruction,
            &mut state,
            &[1.0, 0.0, 1.0, 0.0, ln3, 0.0, 30.0, 40.0],
        );
        assert_close(&second, &[25.0, 20.0, 30.0, 30.0]);
    }

    #[test]
    fn large_scores_stay_finite() {
        let instruction = CausalAttentionInstruction::new(0, 3, 1, 1, 1, 0);
        let mut state = StepState::for_test_slots(2, &[(1, 1)]);
        step(&instruction, &mut state, &[1000.0, 1000.0, 1.0]);
        let out = step(&instruction, &mut state, &[1000.0, -1000.0, 2.0]);
        assert_close(&out, &[1.0]);
    }

    #[test]
    fn errors_without_state_or_room() {
        let instruction = CausalAttentionInstruction::new(0, 3, 1, 1, 1, 0);
        assert!(matches!(
            instruction.apply(&mut [0.0; 4]),
            Err(InstructionModelError::MissingSequenceState { .. })
        ));
        let mut state = StepState::for_test_slots(1, &[(1, 1)]);
        state.position = 1;
        assert!(matches!(
            instruction.apply_with_state(&mut [0.0; 4], &mut state),
            Err(InstructionModelError::SequenceFull { max_positions: 1 })
        ));
        let mut wrong_shape = StepState::for_test_slots(1, &[(2, 1)]);
        assert!(matches!(
            instruction.apply_with_state(&mut [0.0; 4], &mut wrong_shape),
            Err(InstructionModelError::KvSlotIndexOutOfBounds { index: 0 })
        ));
    }
}
