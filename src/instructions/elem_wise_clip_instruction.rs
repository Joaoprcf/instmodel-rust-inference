//! Element-wise clipping instruction implementation.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;

/// Represents an instruction that performs element-wise clipping with optional min/max parameters.
pub struct ElemWiseClipInstruction {
    output_ptr: usize,
    data_size: usize,
    parameters_min: Option<Vec<f32>>,
    parameters_max: Option<Vec<f32>>,
}

impl ElemWiseClipInstruction {
    pub fn new(
        output_ptr: usize,
        data_size: usize,
        parameters_min: Option<&[f32]>,
        parameters_max: Option<&[f32]>,
    ) -> Self {
        Self {
            output_ptr,
            data_size,
            parameters_min: parameters_min.map(|p| p.to_vec()),
            parameters_max: parameters_max.map(|p| p.to_vec()),
        }
    }
}

impl Instruction for ElemWiseClipInstruction {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        self.data_size
    }

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        let output_start = self.output_ptr;

        if let Some(min_params) = &self.parameters_min {
            debug_assert_eq!(min_params.len(), self.data_size);
            for i in 0..self.data_size {
                let val = unified_computation_buffer[output_start + i];
                if val < min_params[i] {
                    unified_computation_buffer[output_start + i] = min_params[i];
                }
            }
        }
        if let Some(max_params) = &self.parameters_max {
            debug_assert_eq!(max_params.len(), self.data_size);
            for i in 0..self.data_size {
                let val = unified_computation_buffer[output_start + i];
                if val > max_params[i] {
                    unified_computation_buffer[output_start + i] = max_params[i];
                }
            }
        }

        Ok(())
    }
}
