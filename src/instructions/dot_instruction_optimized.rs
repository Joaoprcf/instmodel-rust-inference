//! Optimized dot product instruction implementation.
//!
//! This version addresses performance bottlenecks identified in benchmarking.

use crate::activation::Activation;
use crate::errors::InstructionModelError;
use crate::instructions::{Instruction, activation_instruction::ActivationInstruction};

/// Optimized version of DotInstruction that minimizes memory access overhead.
pub struct DotInstructionOptimized {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    input_ptr: usize,
    output_ptr: usize,
    data_size: usize,
    activation_instruction: Option<ActivationInstruction>,
}

impl DotInstructionOptimized {
    /// Creates a new optimized DotInstruction.
    pub fn new(
        input_ptr: usize,
        output_ptr: usize,
        data_size: usize,
        weights: &[Vec<f32>],
        bias: &[f32],
        activation: Option<Activation>,
    ) -> Result<Self, InstructionModelError> {
        let activation_instruction =
            activation.map(|act| ActivationInstruction::new(act, output_ptr, data_size));

        Ok(Self {
            weights: weights.to_vec(),
            bias: bias.to_vec(),
            input_ptr,
            output_ptr,
            data_size,
            activation_instruction,
        })
    }

    /// Optimized forward pass computation using local accumulation.
    fn apply_forward_pass(&self, unified_computation_buffer: &mut [f32]) {
        // Get slices to avoid repeated pointer arithmetic
        let input_slice =
            &unified_computation_buffer[self.input_ptr..self.input_ptr + self.weights[0].len()];
        let output_slice =
            &mut unified_computation_buffer[self.output_ptr..self.output_ptr + self.weights.len()];

        // Perform matrix-vector multiplication with local accumulation (like manual implementation)
        for (i, (weights_row, &bias_value)) in self.weights.iter().zip(self.bias.iter()).enumerate()
        {
            let mut sum = bias_value; // Start with bias

            // Accumulate dot product locally to minimize memory writes
            for (j, &weight) in weights_row.iter().enumerate() {
                sum += weight * input_slice[j];
            }

            // Single write to output
            output_slice[i] = sum;
        }
    }

    /// Ultra-optimized version using unsafe code for maximum performance.
    #[allow(unsafe_code)]
    fn apply_forward_pass_unsafe(&self, unified_computation_buffer: &mut [f32]) {
        unsafe {
            let input_ptr = unified_computation_buffer.as_ptr().add(self.input_ptr);
            let output_ptr = unified_computation_buffer.as_mut_ptr().add(self.output_ptr);

            for (i, (weights_row, &bias_value)) in
                self.weights.iter().zip(self.bias.iter()).enumerate()
            {
                let mut sum = bias_value;

                for (j, &weight) in weights_row.iter().enumerate() {
                    sum += weight * *input_ptr.add(j);
                }

                *output_ptr.add(i) = sum;
            }
        }
    }
}

impl Instruction for DotInstructionOptimized {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        self.data_size
    }

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        // Use safe optimized version by default
        self.apply_forward_pass(unified_computation_buffer);

        // For maximum performance, could switch to:
        // self.apply_forward_pass_unsafe(unified_computation_buffer);

        if let Some(ref activation_instruction) = self.activation_instruction {
            activation_instruction.apply(unified_computation_buffer)?;
        }

        Ok(())
    }
}

/// Additional optimizations that could be implemented:

/// 1. SIMD vectorization for parallel computation
#[cfg(target_arch = "x86_64")]
pub fn dot_product_simd(weights: &[f32], input: &[f32], bias: f32) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = bias;
    let chunks = weights.chunks_exact(8).zip(input.chunks_exact(8));

    unsafe {
        let mut acc = _mm256_setzero_ps();

        for (w_chunk, i_chunk) in chunks {
            let w_vec = _mm256_loadu_ps(w_chunk.as_ptr());
            let i_vec = _mm256_loadu_ps(i_chunk.as_ptr());
            acc = _mm256_fmadd_ps(w_vec, i_vec, acc);
        }

        // Horizontal sum
        let acc_low = _mm256_extractf128_ps(acc, 0);
        let acc_high = _mm256_extractf128_ps(acc, 1);
        let acc_sum = _mm_add_ps(acc_low, acc_high);
        let acc_hadd = _mm_hadd_ps(acc_sum, acc_sum);
        let acc_hadd2 = _mm_hadd_ps(acc_hadd, acc_hadd);
        sum += _mm_cvtss_f32(acc_hadd2);
    }

    // Handle remaining elements
    let remainder = weights.len() % 8;
    if remainder > 0 {
        for i in weights.len() - remainder..weights.len() {
            sum += weights[i] * input[i];
        }
    }

    sum
}

/// 2. Cache-friendly matrix layout (row-major to column-major conversion)
pub fn transpose_weights(weights: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if weights.is_empty() {
        return Vec::new();
    }

    let rows = weights.len();
    let cols = weights[0].len();
    let mut transposed = vec![vec![0.0; rows]; cols];

    for (i, row) in weights.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            transposed[j][i] = value;
        }
    }

    transposed
}

/// 3. Block-based multiplication for better cache utilization
pub fn block_matrix_multiply(
    weights: &[Vec<f32>],
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
    block_size: usize,
) {
    let rows = weights.len();
    let cols = weights[0].len();

    for i_block in (0..rows).step_by(block_size) {
        for j_block in (0..cols).step_by(block_size) {
            for i in i_block..std::cmp::min(i_block + block_size, rows) {
                if i_block == 0 && j_block == 0 {
                    output[i] = bias[i]; // Initialize with bias only once
                }

                for j in j_block..std::cmp::min(j_block + block_size, cols) {
                    output[i] += weights[i][j] * input[j];
                }
            }
        }
    }
}
