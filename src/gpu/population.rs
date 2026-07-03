//! Zero-repack population packing for GPU evolution workloads.
//!
//! A [`PopulationPack`] tiles one serialized [`GpuModel`] blob `n` times into
//! a single contiguous host buffer — one full model per candidate, each at
//! `candidate * model_stride()` f32s. Every candidate shares the header,
//! instructions, and constant parameters; only its weights region differs.
//!
//! Because the weights region is byte-identical to the canonical flat-θ
//! order (verified at construction — see
//! [`GpuModelError::PackedThetaOrderViolation`]), writing a candidate is a
//! pure memcpy of an optimizer's parameter vector: no per-write
//! re-serialization, no repacking. On the GPU side, thread `c` simply calls
//! `predict(c * model_stride, ...)` against the uploaded blob.

use crate::gpu::errors::{GpuModelError, PopulationError};
use crate::gpu::gpu_model::GpuModel;
use crate::graph::ModelGraph;
use crate::instruction_model_info::InstructionModelInfo;
use crate::params::ParamLayout;
use std::ops::Range;

/// A host-side buffer holding `n` copies of one GPU model, with per-candidate
/// mutable weights regions in canonical flat-θ order.
#[derive(Debug, Clone)]
pub struct PopulationPack {
    data: Vec<f32>,
    model_stride: usize,
    n_candidates: usize,
    theta_len: usize,
    weights_offset: usize,
    feature_size: usize,
    output_size: usize,
    output_start: usize,
    compute_buffer_size: usize,
}

impl PopulationPack {
    /// Builds a pack of `n_candidates` copies of the model described by
    /// `info`.
    ///
    /// Construction verifies the zero-repack invariant — the template's
    /// weights region must be byte-identical to
    /// [`ParamLayout::flatten_from_info`] — so a later
    /// [`write_candidate`](Self::write_candidate) is guaranteed to be a plain
    /// memcpy of a canonical θ vector.
    pub fn new(info: &InstructionModelInfo, n_candidates: usize) -> Result<Self, GpuModelError> {
        if n_candidates == 0 {
            return Err(GpuModelError::EmptyPopulation);
        }
        let template = GpuModel::from_info(info)?;
        Self::verify_canonical_theta_order(info, &template)?;

        let template_data = template.as_f32_slice();
        Ok(Self {
            data: template_data.repeat(n_candidates),
            model_stride: template.full_size(),
            n_candidates,
            theta_len: template.theta_len(),
            weights_offset: template.weights_offset(),
            feature_size: template.feature_size(),
            output_size: template.output_size(),
            output_start: template.output_start(),
            compute_buffer_size: template.compute_buffer_size(),
        })
    }

    /// Compiles `graph` with zeroed parameters and builds a pack from it —
    /// the natural entry point for evolution loops that overwrite every
    /// candidate before the first dispatch.
    pub fn from_graph(graph: &ModelGraph, n_candidates: usize) -> Result<Self, PopulationError> {
        let info = graph.compile_zeroed()?;
        Ok(Self::new(&info, n_candidates)?)
    }

    fn verify_canonical_theta_order(
        info: &InstructionModelInfo,
        template: &GpuModel,
    ) -> Result<(), GpuModelError> {
        let layout = ParamLayout::from_info(info).map_err(|source| {
            GpuModelError::PackedThetaOrderViolation {
                message: format!("canonical parameter layout unavailable: {source}"),
            }
        })?;
        if template.theta_len() != layout.total {
            return Err(GpuModelError::PackedThetaOrderViolation {
                message: format!(
                    "GPU weights region holds {} f32s but the canonical layout expects {}",
                    template.theta_len(),
                    layout.total
                ),
            });
        }
        let canonical = ParamLayout::flatten_from_info(info);
        let start = template.weights_offset();
        let region = &template.as_f32_slice()[start..start + template.theta_len()];
        let matches = region
            .iter()
            .zip(canonical.iter())
            .all(|(packed, flat)| packed.to_bits() == flat.to_bits());
        if !matches {
            return Err(GpuModelError::PackedThetaOrderViolation {
                message: "GPU weights region is not byte-identical to canonical flat theta"
                    .to_string(),
            });
        }
        Ok(())
    }

    fn candidate_weights_range(&self, candidate: usize) -> Result<Range<usize>, GpuModelError> {
        if candidate >= self.n_candidates {
            return Err(GpuModelError::CandidateOutOfBounds {
                index: candidate,
                count: self.n_candidates,
            });
        }
        let start = candidate * self.model_stride + self.weights_offset;
        Ok(start..start + self.theta_len)
    }

    fn check_theta_len(&self, got: usize) -> Result<(), GpuModelError> {
        if got != self.theta_len {
            return Err(GpuModelError::ThetaLengthMismatch {
                expected: self.theta_len,
                got,
            });
        }
        Ok(())
    }

    /// Overwrites `candidate`'s weights region with `theta` (canonical flat
    /// order). A pure memcpy.
    pub fn write_candidate(
        &mut self,
        candidate: usize,
        theta: &[f32],
    ) -> Result<(), GpuModelError> {
        self.check_theta_len(theta.len())?;
        let range = self.candidate_weights_range(candidate)?;
        self.data[range].copy_from_slice(theta);
        Ok(())
    }

    /// Like [`write_candidate`](Self::write_candidate) for f64 optimizer
    /// state; each value is narrowed to f32 during the copy.
    pub fn write_candidate_f64(
        &mut self,
        candidate: usize,
        theta: &[f64],
    ) -> Result<(), GpuModelError> {
        self.check_theta_len(theta.len())?;
        let range = self.candidate_weights_range(candidate)?;
        for (slot, &value) in self.data[range].iter_mut().zip(theta.iter()) {
            *slot = value as f32;
        }
        Ok(())
    }

    /// Reads back `candidate`'s weights region (canonical flat order).
    pub fn candidate_theta(&self, candidate: usize) -> Result<&[f32], GpuModelError> {
        let range = self.candidate_weights_range(candidate)?;
        Ok(&self.data[range])
    }

    /// The full packed buffer: `n_candidates` model blobs back to back,
    /// ready to upload as one storage buffer.
    pub fn host_blob(&self) -> &[f32] {
        &self.data
    }

    /// The packed buffer as raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }

    /// Size of one model blob in f32s; candidate `c` starts at
    /// `c * model_stride()`.
    pub fn model_stride(&self) -> usize {
        self.model_stride
    }

    /// Flat parameter count per candidate.
    pub fn theta_len(&self) -> usize {
        self.theta_len
    }

    /// Number of candidates in the pack.
    pub fn n_candidates(&self) -> usize {
        self.n_candidates
    }

    /// Model input size.
    pub fn feature_size(&self) -> usize {
        self.feature_size
    }

    /// Model output size.
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Output start index within a candidate's compute buffer.
    pub fn output_start(&self) -> usize {
        self.output_start
    }

    /// Required compute buffer size per prediction.
    pub fn compute_buffer_size(&self) -> usize {
        self.compute_buffer_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction_model_info::{DotInstructionInfo, InstructionInfo};

    fn dot_info() -> InstructionModelInfo {
        InstructionModelInfo {
            features: None,
            feature_size: Some(2),
            computation_buffer_sizes: vec![2, 2],
            instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
                input: 0,
                output: 1,
                weights: 0,
                activation: None,
            })],
            weights: vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]],
            bias: vec![vec![0.5, -0.5]],
            parameters: None,
            maps: None,
            validation_data: None,
        }
    }

    #[test]
    fn tiles_template_per_candidate() {
        let info = dot_info();
        let template = GpuModel::from_info(&info).unwrap();
        let pack = PopulationPack::new(&info, 3).unwrap();

        assert_eq!(pack.n_candidates(), 3);
        assert_eq!(pack.model_stride(), template.full_size());
        assert_eq!(pack.host_blob().len(), 3 * template.full_size());
        for candidate in 0..3 {
            let start = candidate * pack.model_stride();
            assert_eq!(
                &pack.host_blob()[start..start + pack.model_stride()],
                template.as_f32_slice()
            );
        }
    }

    #[test]
    fn zero_candidates_rejected() {
        let result = PopulationPack::new(&dot_info(), 0);
        assert!(matches!(result, Err(GpuModelError::EmptyPopulation)));
    }

    #[test]
    fn metadata_mirrors_template() {
        let info = dot_info();
        let template = GpuModel::from_info(&info).unwrap();
        let pack = PopulationPack::new(&info, 2).unwrap();

        assert_eq!(pack.theta_len(), template.theta_len());
        assert_eq!(pack.feature_size(), template.feature_size());
        assert_eq!(pack.output_size(), template.output_size());
        assert_eq!(pack.output_start(), template.output_start());
        assert_eq!(pack.compute_buffer_size(), template.compute_buffer_size());
    }
}
