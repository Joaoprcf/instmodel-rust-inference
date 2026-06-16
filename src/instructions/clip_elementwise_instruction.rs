//! Element-wise clip (clamp) instruction implementation.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Represents an instruction that clamps buffer values element-wise to optional
/// lower/upper bounds supplied as parameter vectors.
///
/// Applies `max(x, min[i])` when a lower bound is present and `min(x, max[i])`
/// when an upper bound is present, in that order, matching the reference
/// `CLIP_ELEMENTWISE` (`np.maximum` then `np.minimum`).
pub struct ClipElementwiseInstruction {
    output_ptr: usize,
    data_size: usize,
    parameters_min: Option<Vec<f32>>,
    parameters_max: Option<Vec<f32>>,
}

impl ClipElementwiseInstruction {
    pub fn new(
        output_ptr: usize,
        data_size: usize,
        parameters_min: Option<Vec<f32>>,
        parameters_max: Option<Vec<f32>>,
    ) -> Self {
        Self {
            output_ptr,
            data_size,
            parameters_min,
            parameters_max,
        }
    }
}

impl Instruction for ClipElementwiseInstruction {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        self.data_size
    }

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        debug_assert!(
            self.parameters_min
                .as_ref()
                .is_none_or(|p| p.len() == self.data_size)
        );
        debug_assert!(
            self.parameters_max
                .as_ref()
                .is_none_or(|p| p.len() == self.data_size)
        );

        let output_start = self.output_ptr;

        for i in 0..self.data_size {
            let mut value = unified_computation_buffer[output_start + i];
            if let Some(min) = &self.parameters_min {
                value = value.max(min[i]);
            }
            if let Some(max) = &self.parameters_max {
                value = value.min(max[i]);
            }
            unified_computation_buffer[output_start + i] = value;
        }

        Ok(())
    }
}
