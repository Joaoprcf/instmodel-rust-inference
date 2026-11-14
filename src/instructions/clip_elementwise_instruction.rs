//! CLIP_ELEMENTWISE.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// In-place `min(max(x, lower), upper)` with optional per-element bounds.
#[derive(Clone)]
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
    super::instruction_common!();

    fn apply(&self, buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        let values = &mut buffer[self.output_ptr..self.output_ptr + self.data_size];
        if let Some(min) = &self.parameters_min {
            values.iter_mut().zip(min).for_each(|(v, &m)| *v = v.max(m));
        }
        if let Some(max) = &self.parameters_max {
            values.iter_mut().zip(max).for_each(|(v, &m)| *v = v.min(m));
        }
        Ok(())
    }
}
