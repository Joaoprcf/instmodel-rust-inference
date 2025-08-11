//! Map transform instruction implementation.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;
use std::collections::HashMap;

/// Represents an instruction that maps a feature to a vector of values using a hashtable.
/// This instruction mimics an embedding layer in a neural network.
/// It is used to map categorical features such as IDs or types to a vector of values.
pub struct MapTransformInstruction {
    input_ptr: usize,
    output_ptr: usize,
    data_size: usize,
    default_value: Vec<f32>,
    map: HashMap<i32, Vec<f32>>,
}

impl MapTransformInstruction {
    pub fn new(
        input_ptr: usize,
        output_ptr: usize,
        data_size: usize,
        map: &HashMap<String, Vec<f32>>,
        default_value: &[f32],
    ) -> Self {
        // Convert string keys to integer keys
        let int_map: HashMap<i32, Vec<f32>> = map
            .iter()
            .filter_map(|(k, v)| k.parse::<i32>().ok().map(|key| (key, v.clone())))
            .collect();

        Self {
            input_ptr,
            output_ptr,
            data_size,
            default_value: default_value.to_vec(),
            map: int_map,
        }
    }
}

impl Instruction for MapTransformInstruction {
    fn output_ptr(&self) -> usize {
        self.output_ptr
    }

    fn data_size(&self) -> usize {
        self.data_size
    }

    fn apply(&self, unified_computation_buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        let input_value = unified_computation_buffer[self.input_ptr];
        let key = input_value.round() as i32;

        let values = self.map.get(&key).unwrap_or(&self.default_value);

        for (i, &value) in values.iter().enumerate().take(self.data_size) {
            unified_computation_buffer[self.output_ptr + i] = value;
        }

        Ok(())
    }
}
