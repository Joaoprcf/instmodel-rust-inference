//! Flat parameter layout: the offset table mapping a flat `theta: &[f32]` onto
//! the model's weight/bias storage.
//!
//! The canonical θ (source) order is: for each weight slot, in slot order, its
//! row-major `[out × in]` weight run followed by its `[out]` bias run. Shared
//! slots appear exactly once. This order is also how [`crate::gpu::GpuModel`]
//! serializes its weights region, so a flat θ doubles as a GPU weights payload
//! with no repacking.

use crate::errors::InstructionModelError;
use crate::instruction_model_info::InstructionModelInfo;

/// Which flat destination buffer a [`ParamCopy`] targets.
///
/// Marked `#[non_exhaustive]`: a future `Parameters` kind (trainable clip
/// bounds, normalization vectors) can be added without a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParamKind {
    /// The flat weights-buffer (reshaped via [`ParamLayout::weight_shapes`]).
    Weights,
    /// The flat bias-buffer (reshaped via [`ParamLayout::bias_lens`]).
    Bias,
}

/// One contiguous copy from the flat θ into a flat destination buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamCopy {
    pub kind: ParamKind,
    /// Offset into the flat destination buffer of `kind`.
    pub dest_offset: u32,
    /// Offset into the flat source `theta`.
    pub src_offset: u32,
    /// Number of scalars to copy.
    pub size: u32,
}

/// Jagged weight/bias storage in the shape [`InstructionModelInfo`] expects.
#[derive(Debug, Clone, PartialEq)]
pub struct JaggedParams {
    /// One `[out][in]` matrix per weight slot.
    pub weights: Vec<Vec<Vec<f32>>>,
    /// One `[out]` vector per weight slot.
    pub bias: Vec<Vec<f32>>,
}

/// The complete offset table for a model's dense parameters, plus the shapes
/// needed to reshape the flat buffers back into the served jagged format.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamLayout {
    /// One weight run + one bias run per weight slot, in slot order.
    pub copies: Vec<ParamCopy>,
    /// `[out, in]` of each weight slot (index = slot = DOT/ATTENTION `weights` index).
    pub weight_shapes: Vec<[usize; 2]>,
    /// `out` (bias length) of each weight slot.
    pub bias_lens: Vec<usize>,
    /// Total scalars across all weight matrices.
    pub weights_len: usize,
    /// Total scalars across all bias vectors.
    pub bias_len: usize,
    /// Total θ length (`weights_len + bias_len`).
    pub total: usize,
}

impl ParamLayout {
    /// Builds the layout from `[out, in]` shapes, one per weight slot in slot order.
    pub fn from_shapes(shapes: &[[usize; 2]]) -> Self {
        let mut copies = Vec::with_capacity(shapes.len() * 2);
        let mut weight_shapes = Vec::with_capacity(shapes.len());
        let mut bias_lens = Vec::with_capacity(shapes.len());
        let mut weights_len = 0usize;
        let mut bias_len = 0usize;
        let mut total = 0usize;

        for &[out, inp] in shapes {
            let weight_size = out * inp;
            copies.push(ParamCopy {
                kind: ParamKind::Weights,
                dest_offset: weights_len as u32,
                src_offset: total as u32,
                size: weight_size as u32,
            });
            total += weight_size;
            copies.push(ParamCopy {
                kind: ParamKind::Bias,
                dest_offset: bias_len as u32,
                src_offset: total as u32,
                size: out as u32,
            });
            total += out;
            weights_len += weight_size;
            bias_len += out;
            weight_shapes.push([out, inp]);
            bias_lens.push(out);
        }

        ParamLayout {
            copies,
            weight_shapes,
            bias_lens,
            weights_len,
            bias_len,
            total,
        }
    }

    /// Derives the layout from a model definition's jagged weights/bias.
    ///
    /// Works for any source of [`InstructionModelInfo`] — hand-built, JSON-loaded,
    /// or graph-compiled. Every weight matrix must be rectangular and match its
    /// bias length; definitions accepted by [`crate::InstructionModel::new`]
    /// always satisfy both.
    pub fn from_info(info: &InstructionModelInfo) -> Result<Self, InstructionModelError> {
        if info.bias.len() != info.weights.len() {
            return Err(InstructionModelError::BiasWeightsMismatch);
        }

        let mut shapes = Vec::with_capacity(info.weights.len());
        for (slot, (matrix, bias)) in info.weights.iter().zip(info.bias.iter()).enumerate() {
            let columns = matrix.first().map_or(0, |row| row.len());
            for (row_index, row) in matrix.iter().enumerate() {
                if row.len() != columns {
                    return Err(InstructionModelError::JaggedWeightsMatrix {
                        slot,
                        row: row_index,
                        expected: columns,
                        got: row.len(),
                    });
                }
            }
            if bias.len() != matrix.len() {
                return Err(InstructionModelError::BiasWeightsSizeMismatch {
                    index: slot,
                    bias_size: bias.len(),
                    weights_size: matrix.len(),
                });
            }
            shapes.push([matrix.len(), columns]);
        }

        Ok(Self::from_shapes(&shapes))
    }

    /// Number of weight slots.
    pub fn slot_count(&self) -> usize {
        self.weight_shapes.len()
    }

    /// Range of `slot`'s row-major weight run within the flat θ.
    ///
    /// Panics if `slot >= slot_count()`.
    pub fn slot_weight_range(&self, slot: usize) -> std::ops::Range<usize> {
        let copy = &self.copies[2 * slot];
        copy.src_offset as usize..(copy.src_offset + copy.size) as usize
    }

    /// Range of `slot`'s bias run within the flat θ.
    ///
    /// Panics if `slot >= slot_count()`.
    pub fn slot_bias_range(&self, slot: usize) -> std::ops::Range<usize> {
        let copy = &self.copies[2 * slot + 1];
        copy.src_offset as usize..(copy.src_offset + copy.size) as usize
    }

    /// Reshapes a flat θ into the jagged weights/bias format of
    /// [`InstructionModelInfo`].
    pub fn scatter_to_jagged(&self, theta: &[f32]) -> Result<JaggedParams, InstructionModelError> {
        if theta.len() != self.total {
            return Err(InstructionModelError::ThetaLengthMismatch {
                expected: self.total,
                got: theta.len(),
            });
        }

        let mut weights = Vec::with_capacity(self.weight_shapes.len());
        let mut bias = Vec::with_capacity(self.weight_shapes.len());
        for (slot, &[out, inp]) in self.weight_shapes.iter().enumerate() {
            let weight_run = &theta[self.slot_weight_range(slot)];
            let matrix: Vec<Vec<f32>> = (0..out)
                .map(|row| weight_run[row * inp..(row + 1) * inp].to_vec())
                .collect();
            weights.push(matrix);
            bias.push(theta[self.slot_bias_range(slot)].to_vec());
        }

        Ok(JaggedParams { weights, bias })
    }

    /// Flattens a model definition's jagged weights/bias into canonical θ order.
    pub fn flatten_from_info(info: &InstructionModelInfo) -> Vec<f32> {
        let mut theta = Vec::new();
        for (matrix, bias) in info.weights.iter().zip(info.bias.iter()) {
            for row in matrix {
                theta.extend_from_slice(row);
            }
            theta.extend_from_slice(bias);
        }
        theta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_slot_info() -> InstructionModelInfo {
        InstructionModelInfo {
            features: None,
            feature_size: Some(3),
            computation_buffer_sizes: vec![3, 2, 1],
            instructions: vec![],
            weights: vec![
                vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
                vec![vec![7.0, 8.0]],
            ],
            bias: vec![vec![0.1, 0.2], vec![0.3]],
            parameters: None,
            maps: None,
            validation_data: None,
        }
    }

    #[test]
    fn from_shapes_offsets_and_totals() {
        let layout = ParamLayout::from_shapes(&[[2, 3], [1, 2]]);

        assert_eq!(layout.total, 6 + 2 + 2 + 1);
        assert_eq!(layout.weights_len, 8);
        assert_eq!(layout.bias_len, 3);
        assert_eq!(layout.weight_shapes, vec![[2, 3], [1, 2]]);
        assert_eq!(layout.bias_lens, vec![2, 1]);

        assert_eq!(layout.slot_weight_range(0), 0..6);
        assert_eq!(layout.slot_bias_range(0), 6..8);
        assert_eq!(layout.slot_weight_range(1), 8..10);
        assert_eq!(layout.slot_bias_range(1), 10..11);

        assert_eq!(layout.copies[0].kind, ParamKind::Weights);
        assert_eq!(layout.copies[0].dest_offset, 0);
        assert_eq!(layout.copies[1].kind, ParamKind::Bias);
        assert_eq!(layout.copies[1].dest_offset, 0);
        assert_eq!(layout.copies[2].dest_offset, 6);
        assert_eq!(layout.copies[3].dest_offset, 2);
    }

    #[test]
    fn from_info_matches_from_shapes() {
        let info = two_slot_info();
        let layout = ParamLayout::from_info(&info).unwrap();
        assert_eq!(layout, ParamLayout::from_shapes(&[[2, 3], [1, 2]]));
    }

    #[test]
    fn from_info_rejects_jagged_matrix() {
        let mut info = two_slot_info();
        info.weights[0][1] = vec![4.0, 5.0];

        let result = ParamLayout::from_info(&info);
        assert!(matches!(
            result,
            Err(InstructionModelError::JaggedWeightsMatrix {
                slot: 0,
                row: 1,
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn from_info_rejects_bias_length_mismatch() {
        let mut info = two_slot_info();
        info.bias[1] = vec![0.3, 0.4];

        let result = ParamLayout::from_info(&info);
        assert!(matches!(
            result,
            Err(InstructionModelError::BiasWeightsSizeMismatch { index: 1, .. })
        ));
    }

    #[test]
    fn flatten_then_scatter_round_trips() {
        let info = two_slot_info();
        let layout = ParamLayout::from_info(&info).unwrap();

        let theta = ParamLayout::flatten_from_info(&info);
        assert_eq!(
            theta,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 7.0, 8.0, 0.3]
        );

        let jagged = layout.scatter_to_jagged(&theta).unwrap();
        assert_eq!(jagged.weights, info.weights);
        assert_eq!(jagged.bias, info.bias);
    }

    #[test]
    fn scatter_rejects_wrong_length() {
        let layout = ParamLayout::from_shapes(&[[2, 3]]);
        let result = layout.scatter_to_jagged(&[0.0; 4]);
        assert!(matches!(
            result,
            Err(InstructionModelError::ThetaLengthMismatch {
                expected: 8,
                got: 4
            })
        ));
    }
}
