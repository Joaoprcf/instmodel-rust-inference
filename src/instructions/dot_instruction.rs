//! Dot product instruction implementation.
//!
//! Represents an instruction that performs a complete dot product operation,
//! similar to the Dense layer in common deep learning frameworks.

use crate::activation::Activation;
use crate::errors::InstructionModelError;
use crate::instructions::Instruction;
use crate::utils::dot::{DotKernel, dot};

/// Instruction that performs a dense (matrix-vector) operation followed by bias and activation.
///
/// This implementation flattens the weight matrix and uses a runtime-selected SIMD kernel
/// (AVX512/AVX2 + FMA when available) to minimize overhead in the hot loop.
pub struct DotInstruction {
    weights: Vec<f32>,
    bias: Vec<f32>,
    input_ptr: usize,
    output_ptr: usize,
    data_size: usize,
    input_size: usize,
    kernel: DotKernel,
    activation: Option<Activation>,
}

impl DotInstruction {
    /// Creates a new DotInstruction.
    pub fn new(
        input_ptr: usize,
        output_ptr: usize,
        data_size: usize,
        weights: &[Vec<f32>],
        bias: &[f32],
        activation: Option<Activation>,
    ) -> Result<Self, InstructionModelError> {
        let input_size = weights.first().map_or(0, |row| row.len());
        let mut flattened_weights = Vec::with_capacity(weights.len() * input_size);
        for row in weights {
            flattened_weights.extend_from_slice(row);
        }

        Ok(Self {
            weights: flattened_weights,
            bias: bias.to_vec(),
            input_ptr,
            output_ptr,
            data_size,
            input_size,
            kernel: DotKernel::detect(),
            activation,
        })
    }

    #[inline(always)]
    fn apply_forward_pass(&self, unified_computation_buffer: &mut [f32]) {
        let row_stride = self.input_size;
        let kernel = self.kernel;

        let input_end = self.input_ptr + self.input_size;
        let output_end = self.output_ptr + self.data_size;

        let (input_slice, output_slice) = if self.input_ptr < self.output_ptr {
            debug_assert!(input_end <= self.output_ptr);
            let (before_output, output_and_after) =
                unified_computation_buffer.split_at_mut(self.output_ptr);
            (
                &before_output[self.input_ptr..input_end],
                &mut output_and_after[..self.data_size],
            )
        } else {
            debug_assert!(output_end <= self.input_ptr);
            let (before_input, input_and_after) =
                unified_computation_buffer.split_at_mut(self.input_ptr);
            (
                &input_and_after[..self.input_size],
                &mut before_input[self.output_ptr..output_end],
            )
        };

        debug_assert_eq!(input_slice.len(), self.input_size);
        debug_assert_eq!(output_slice.len(), self.data_size);

        let input_ptr = input_slice.as_ptr();
        let weights_rows = self.weights.chunks_exact(row_stride);
        debug_assert!(weights_rows.remainder().is_empty());

        for ((out, &bias_value), weights_row) in output_slice
            .iter_mut()
            .zip(self.bias.iter())
            .zip(weights_rows)
        {
            *out = dot(kernel, weights_row.as_ptr(), input_ptr, row_stride) + bias_value;
        }

        if let Some(activation) = self.activation {
            activation.apply_in_place(output_slice);
        }
    }
}

impl Instruction for DotInstruction {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        self.data_size
    }

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        self.apply_forward_pass(unified_computation_buffer);
        Ok(())
    }
}
