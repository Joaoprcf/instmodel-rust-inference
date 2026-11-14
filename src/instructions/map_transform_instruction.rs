//! MAP_TRANSFORM.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;
use std::collections::HashMap;

/// Embedding lookup: writes the vector mapped to `round(input)` (or the default) to the output.
#[derive(Clone)]
pub struct MapTransformInstruction {
    input_ptr: usize,
    output_ptr: usize,
    data_size: usize,
    default_value: Vec<f32>,
    map: HashMap<i32, Vec<f32>>,
}

impl MapTransformInstruction {
    /// Keys that do not parse as `i32` are ignored.
    pub fn new(
        input_ptr: usize,
        output_ptr: usize,
        data_size: usize,
        map: &HashMap<String, Vec<f32>>,
        default_value: &[f32],
    ) -> Self {
        Self {
            input_ptr,
            output_ptr,
            data_size,
            default_value: default_value.to_vec(),
            map: map
                .iter()
                .filter_map(|(k, v)| k.parse::<i32>().ok().map(|key| (key, v.clone())))
                .collect(),
        }
    }
}

impl Instruction for MapTransformInstruction {
    super::instruction_common!();

    fn apply(&self, buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        let key = buffer[self.input_ptr].round() as i32;
        let values = self.map.get(&key).unwrap_or(&self.default_value);
        for (i, &value) in values.iter().enumerate().take(self.data_size) {
            buffer[self.output_ptr + i] = value;
        }
        Ok(())
    }
}
