use crate::activation::Activation;
use crate::errors::InstructionModelError;
use crate::instructions::Instruction;
use crate::utils::dot::Dense;

/// Dense layer: matrix-vector product plus bias, then an optional activation.
#[derive(Clone)]
pub struct DotInstruction {
    dense: Dense,
    input_ptr: usize,
    output_ptr: usize,
    data_size: usize,
    activation: Option<Activation>,
}

impl DotInstruction {
    pub fn new(
        input_ptr: usize,
        output_ptr: usize,
        data_size: usize,
        weights: &[Vec<f32>],
        bias: &[f32],
        activation: Option<Activation>,
    ) -> Result<Self, InstructionModelError> {
        Ok(Self {
            dense: Dense::new(weights, bias),
            input_ptr,
            output_ptr,
            data_size,
            activation,
        })
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
        let out = self
            .dense
            .forward(unified_computation_buffer, self.input_ptr, self.output_ptr);
        if let Some(activation) = self.activation {
            activation.apply_in_place(out);
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

    fn instruction() -> DotInstruction {
        let weights = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let bias = vec![0.5, -0.5];
        DotInstruction::new(0, 2, 2, &weights, &bias, None).unwrap()
    }

    #[test]
    fn set_and_read_dense_params_round_trip() {
        let mut instruction = instruction();
        instruction
            .set_dense_params(&[5.0, 6.0, 7.0, 8.0], &[1.0, 2.0])
            .unwrap();

        let mut weights = vec![0.0; 4];
        let mut bias = vec![0.0; 2];
        instruction
            .read_dense_params(&mut weights, &mut bias)
            .unwrap();
        assert_eq!(weights, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(bias, vec![1.0, 2.0]);

        let mut buffer = vec![1.0, 1.0, 0.0, 0.0];
        instruction.apply(&mut buffer).unwrap();
        assert_eq!(&buffer[2..], &[12.0, 17.0]);
    }

    #[test]
    fn set_dense_params_rejects_wrong_lengths() {
        let mut instruction = instruction();
        let result = instruction.set_dense_params(&[1.0, 2.0, 3.0], &[1.0, 2.0]);
        assert!(matches!(
            result,
            Err(InstructionModelError::ThetaLengthMismatch {
                expected: 6,
                got: 5
            })
        ));
    }

    #[test]
    fn clone_box_yields_independent_copy() {
        let mut original = instruction();
        let cloned = original.clone_box().unwrap();
        original
            .set_dense_params(&[9.0, 9.0, 9.0, 9.0], &[9.0, 9.0])
            .unwrap();

        let mut buffer = vec![1.0, 1.0, 0.0, 0.0];
        cloned.apply(&mut buffer).unwrap();
        assert_eq!(&buffer[2..], &[3.5, 6.5]);
    }
}
