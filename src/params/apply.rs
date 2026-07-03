//! In-place flat-parameter mutation for [`InstructionModel`].
//!
//! A model built by [`InstructionModel::new`] carries a [`ThetaBinding`]: the
//! [`ParamLayout`] of its definition plus, per weight slot, the indices of the
//! instructions that own a private copy of that slot (shared slots are
//! duplicated into every referencing instruction at build time, so all owners
//! must be patched). [`InstructionModel::apply_theta`] is then a handful of
//! `copy_from_slice`s — no allocation, no revalidation, no rebuild.

use crate::errors::InstructionModelError;
use crate::instruction_model::InstructionModel;
use crate::params::layout::ParamLayout;

/// Precomputed link between a model's flat θ and its instructions.
#[derive(Debug, Clone)]
pub(crate) struct ThetaBinding {
    pub(crate) layout: ParamLayout,
    /// Instruction indices owning each weight slot, in slot order.
    pub(crate) slot_owners: Vec<Vec<usize>>,
}

fn theta_unsupported() -> InstructionModelError {
    InstructionModelError::ThetaUnsupported {
        reason: "model was built without a parameter binding (e.g. via new_for_test)".to_string(),
    }
}

impl InstructionModel {
    /// Flat parameter count of this model, or 0 when the model carries no
    /// parameter binding (models built via [`InstructionModel::new_for_test`]).
    pub fn theta_len(&self) -> usize {
        self.theta_binding
            .as_ref()
            .map_or(0, |binding| binding.layout.total)
    }

    /// The model's flat parameter layout, when it carries one.
    pub fn param_layout(&self) -> Option<&ParamLayout> {
        self.theta_binding.as_ref().map(|binding| &binding.layout)
    }

    /// Overwrites every dense parameter (weights + bias) in place from a flat θ
    /// in canonical order — see [`ParamLayout`] for the order contract.
    ///
    /// This mutates the existing instructions without rebuilding or revalidating
    /// the model, so it runs at memcpy speed regardless of model depth.
    pub fn apply_theta(&mut self, theta: &[f32]) -> Result<(), InstructionModelError> {
        let binding = self.theta_binding.as_ref().ok_or_else(theta_unsupported)?;
        if theta.len() != binding.layout.total {
            return Err(InstructionModelError::ThetaLengthMismatch {
                expected: binding.layout.total,
                got: theta.len(),
            });
        }

        for (slot, owners) in binding.slot_owners.iter().enumerate() {
            let weights = &theta[binding.layout.slot_weight_range(slot)];
            let bias = &theta[binding.layout.slot_bias_range(slot)];
            for &owner in owners {
                self.instructions[owner].set_dense_params(weights, bias)?;
            }
        }
        Ok(())
    }

    /// Reads the model's current dense parameters into `theta` (canonical order).
    /// `theta.len()` must equal [`InstructionModel::theta_len`].
    pub fn read_theta(&self, theta: &mut [f32]) -> Result<(), InstructionModelError> {
        let binding = self.theta_binding.as_ref().ok_or_else(theta_unsupported)?;
        if theta.len() != binding.layout.total {
            return Err(InstructionModelError::ThetaLengthMismatch {
                expected: binding.layout.total,
                got: theta.len(),
            });
        }

        for (slot, owners) in binding.slot_owners.iter().enumerate() {
            let weight_range = binding.layout.slot_weight_range(slot);
            let bias_range = binding.layout.slot_bias_range(slot);
            // The slot's weight and bias runs are adjacent in θ; split one
            // mutable slice so both can be filled by a single owner read.
            let slot_slice = &mut theta[weight_range.start..bias_range.end];
            let (weights, bias) = slot_slice.split_at_mut(weight_range.len());
            if let Some(&owner) = owners.first() {
                self.instructions[owner].read_dense_params(weights, bias)?;
            }
        }
        Ok(())
    }

    /// Returns an independent copy of this model, sharing no mutable state, so
    /// each worker thread can [`InstructionModel::apply_theta`] its own copy.
    ///
    /// Fallible rather than `impl Clone`: models assembled from external
    /// [`crate::instructions::Instruction`] implementations may not support
    /// cloning, in which case the offending instruction index is reported.
    pub fn try_clone(&self) -> Result<Self, InstructionModelError> {
        let mut instructions = Vec::with_capacity(self.instructions.len());
        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            let cloned = instruction
                .clone_box()
                .ok_or(InstructionModelError::CloneUnsupported { instruction_index })?;
            instructions.push(cloned);
        }

        Ok(InstructionModel {
            instructions,
            feature_size: self.feature_size,
            computation_buffer_sizes: self.computation_buffer_sizes.clone(),
            computation_buffer_indexes: self.computation_buffer_indexes.clone(),
            output_index_start: self.output_index_start,
            output_index_end: self.output_index_end,
            parallel_graph: self.parallel_graph.clone(),
            theta_binding: self.theta_binding.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction_model_info::{
        DotInstructionInfo, InstructionInfo, InstructionModelInfo,
    };

    fn two_layer_info(weights: Vec<Vec<Vec<f32>>>, bias: Vec<Vec<f32>>) -> InstructionModelInfo {
        InstructionModelInfo {
            features: None,
            feature_size: Some(3),
            computation_buffer_sizes: vec![3, 2, 1],
            instructions: vec![
                InstructionInfo::Dot(DotInstructionInfo {
                    input: 0,
                    output: 1,
                    weights: 0,
                    activation: None,
                }),
                InstructionInfo::Dot(DotInstructionInfo {
                    input: 1,
                    output: 2,
                    weights: 1,
                    activation: None,
                }),
            ],
            weights,
            bias,
            parameters: None,
            maps: None,
            validation_data: None,
        }
    }

    fn zeroed_two_layer_model() -> InstructionModel {
        let info = two_layer_info(
            vec![vec![vec![0.0; 3], vec![0.0; 3]], vec![vec![0.0; 2]]],
            vec![vec![0.0; 2], vec![0.0]],
        );
        InstructionModel::new(info).unwrap()
    }

    #[test]
    fn apply_theta_matches_freshly_built_model() {
        let theta: Vec<f32> = (1..=11).map(|v| v as f32 * 0.25).collect();

        let mut mutated = zeroed_two_layer_model();
        assert_eq!(mutated.theta_len(), 11);
        mutated.apply_theta(&theta).unwrap();

        let layout = mutated.param_layout().unwrap().clone();
        let jagged = layout.scatter_to_jagged(&theta).unwrap();
        let rebuilt = InstructionModel::new(two_layer_info(jagged.weights, jagged.bias)).unwrap();

        let input = [0.5, -1.0, 2.0];
        assert_eq!(
            mutated.predict(&input).unwrap(),
            rebuilt.predict(&input).unwrap()
        );
    }

    #[test]
    fn read_theta_round_trips() {
        let theta: Vec<f32> = (1..=11).map(|v| v as f32).collect();
        let mut model = zeroed_two_layer_model();
        model.apply_theta(&theta).unwrap();

        let mut read_back = vec![0.0f32; model.theta_len()];
        model.read_theta(&mut read_back).unwrap();
        assert_eq!(read_back, theta);
    }

    #[test]
    fn apply_theta_rejects_wrong_length() {
        let mut model = zeroed_two_layer_model();
        let result = model.apply_theta(&[0.0; 3]);
        assert!(matches!(
            result,
            Err(InstructionModelError::ThetaLengthMismatch {
                expected: 11,
                got: 3
            })
        ));
    }

    #[test]
    fn test_model_reports_theta_unsupported() {
        let mut model = InstructionModel::new_for_test(vec![2, 2], vec![], 2).unwrap();
        assert_eq!(model.theta_len(), 0);
        assert!(model.param_layout().is_none());
        assert!(matches!(
            model.apply_theta(&[]),
            Err(InstructionModelError::ThetaUnsupported { .. })
        ));
        assert!(matches!(
            model.read_theta(&mut []),
            Err(InstructionModelError::ThetaUnsupported { .. })
        ));
    }

    #[test]
    fn try_clone_is_independent() {
        let theta_a: Vec<f32> = (1..=11).map(|v| v as f32).collect();
        let theta_b: Vec<f32> = (1..=11).map(|v| -(v as f32)).collect();

        let mut original = zeroed_two_layer_model();
        original.apply_theta(&theta_a).unwrap();

        let mut cloned = original.try_clone().unwrap();
        cloned.apply_theta(&theta_b).unwrap();

        let mut original_theta = vec![0.0f32; 11];
        original.read_theta(&mut original_theta).unwrap();
        assert_eq!(original_theta, theta_a);

        let mut cloned_theta = vec![0.0f32; 11];
        cloned.read_theta(&mut cloned_theta).unwrap();
        assert_eq!(cloned_theta, theta_b);
    }
}
