//! [`GpuModel`]: an [`InstructionModelInfo`] packed into one f32 blob for WGSL execution.

use crate::errors::InstructionModelError;
use crate::gpu::errors::{GpuModelError, GpuModelResult};
use crate::gpu::gpu_instruction::GpuInstruction;
use crate::instruction_model_info::{InstructionInfo, InstructionModelInfo};
use crate::instruction_model_validation as validation;

fn model_error(e: InstructionModelError) -> GpuModelError {
    GpuModelError::InvalidModelBinary {
        message: e.to_string(),
    }
}

/// Magic number for GPU model binary format ("GPMI" in little-endian).
pub const GPU_MODEL_MAGIC: u32 = 0x494D5047;

/// Blob format version. Pair blobs with WGSL from the same crate version: older
/// interpreters silently skip unknown opcodes.
pub const GPU_MODEL_VERSION: u32 = 2;

/// Header length in f32s (bitcast u32 fields).
pub const HEADER_SIZE_F32S: usize = 13;

pub const MAX_GPU_COMPUTE_BUFFER: usize = 65536;

/// A model serialized as `[header | instructions | weights (θ order) | parameters]`.
#[derive(Debug, Clone)]
pub struct GpuModel {
    data: Vec<f32>,
    feature_size: usize,
    output_size: usize,
    compute_buffer_size: usize,
    output_start: usize,
    weights_offset: usize,
    weights_len: usize,
}

impl GpuModel {
    /// Validates and serializes `info` (ATTENTION and MAP_TRANSFORM are unsupported).
    pub fn from_info(info: &InstructionModelInfo) -> GpuModelResult<Self> {
        let mut builder = GpuModelBuilder::new(info)?;
        builder.build()
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }

    /// Blob length in f32s.
    pub fn full_size(&self) -> usize {
        self.data.len()
    }

    pub fn feature_size(&self) -> usize {
        self.feature_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn compute_buffer_size(&self) -> usize {
        self.compute_buffer_size
    }

    pub fn output_start(&self) -> usize {
        self.output_start
    }

    pub fn weights_offset(&self) -> usize {
        self.weights_offset
    }

    /// Length of the weights region, i.e. the model's θ length.
    pub fn theta_len(&self) -> usize {
        self.weights_len
    }

    /// Overwrites the weights region with `theta` (a memcpy; nothing else changes).
    pub fn write_theta(&mut self, theta: &[f32]) -> Result<(), GpuModelError> {
        if theta.len() != self.weights_len {
            return Err(GpuModelError::ThetaLengthMismatch {
                expected: self.weights_len,
                got: theta.len(),
            });
        }
        self.data[self.weights_offset..self.weights_offset + self.weights_len]
            .copy_from_slice(theta);
        Ok(())
    }

    /// Copies the weights region into `theta`.
    pub fn read_theta(&self, theta: &mut [f32]) -> Result<(), GpuModelError> {
        if theta.len() != self.weights_len {
            return Err(GpuModelError::ThetaLengthMismatch {
                expected: self.weights_len,
                got: theta.len(),
            });
        }
        theta.copy_from_slice(
            &self.data[self.weights_offset..self.weights_offset + self.weights_len],
        );
        Ok(())
    }
}

struct GpuModelBuilder<'a> {
    info: &'a InstructionModelInfo,
    computation_buffer_indexes: Vec<usize>,
    computation_buffer_sizes: Vec<usize>,
    weights_data: Vec<f32>,
    parameters_data: Vec<f32>,
    instructions: Vec<GpuInstruction>,
    weights_offsets: Vec<u32>,
    params_offsets: Vec<u32>,
}

impl<'a> GpuModelBuilder<'a> {
    fn new(info: &'a InstructionModelInfo) -> GpuModelResult<Self> {
        let computation_buffer_sizes = info.computation_buffer_sizes.clone();
        let Some(&first) = computation_buffer_sizes.first() else {
            return Err(model_error(InstructionModelError::NoLayersProvided));
        };
        let mut computation_buffer_indexes = vec![0usize];
        let mut current_idx = first;
        for &size in computation_buffer_sizes.iter().skip(1) {
            computation_buffer_indexes.push(current_idx);
            current_idx = current_idx.saturating_add(size);
        }

        if current_idx > MAX_GPU_COMPUTE_BUFFER {
            return Err(GpuModelError::ComputeBufferTooLarge {
                required: current_idx,
                max_size: MAX_GPU_COMPUTE_BUFFER,
            });
        }

        Ok(Self {
            info,
            computation_buffer_indexes,
            computation_buffer_sizes,
            weights_data: Vec::new(),
            parameters_data: Vec::new(),
            instructions: Vec::new(),
            weights_offsets: Vec::new(),
            params_offsets: Vec::new(),
        })
    }

    fn build(&mut self) -> GpuModelResult<GpuModel> {
        self.collect_weights_and_params()?;

        self.build_instructions()?;

        let compute_buffer_size = self.computation_buffer_indexes.last().unwrap()
            + self.computation_buffer_sizes.last().unwrap();
        let output_start = *self.computation_buffer_indexes.last().unwrap();
        let output_size = *self.computation_buffer_sizes.last().unwrap();
        let feature_size = self.calculate_feature_size()?;

        let instructions_offset = HEADER_SIZE_F32S;
        let weights_offset =
            instructions_offset + self.instructions.len() * GpuInstruction::SIZE_U32S;
        let params_offset = weights_offset + self.weights_data.len();

        let mut data = vec![
            f32::from_bits(GPU_MODEL_MAGIC),
            f32::from_bits(GPU_MODEL_VERSION),
            f32::from_bits(feature_size as u32),
            f32::from_bits(output_size as u32),
            f32::from_bits(compute_buffer_size as u32),
            f32::from_bits(self.instructions.len() as u32),
            f32::from_bits(instructions_offset as u32),
            f32::from_bits(weights_offset as u32),
            f32::from_bits(self.weights_data.len() as u32),
            f32::from_bits(params_offset as u32),
            f32::from_bits(self.parameters_data.len() as u32),
            f32::from_bits(output_start as u32),
        ];

        let full_model_size_index = data.len();
        data.push(0.0); // Placeholder

        for inst in &self.instructions {
            data.extend_from_slice(&inst.to_f32_array());
        }

        data.extend_from_slice(&self.weights_data);

        data.extend_from_slice(&self.parameters_data);

        data[full_model_size_index] = f32::from_bits(data.len() as u32);

        Ok(GpuModel {
            data,
            feature_size,
            output_size,
            compute_buffer_size,
            output_start,
            weights_offset,
            weights_len: self.weights_data.len(),
        })
    }

    fn calculate_feature_size(&self) -> GpuModelResult<usize> {
        let feature_size = if self.info.features.is_none() && self.info.feature_size.is_none() {
            self.computation_buffer_sizes[0]
        } else {
            validation::calculate_feature_size(self.info).map_err(model_error)?
        };
        validation::validate_feature_size(feature_size, &self.computation_buffer_sizes)
            .map_err(|e| model_error(e.into()))?;
        Ok(feature_size)
    }

    fn collect_weights_and_params(&mut self) -> GpuModelResult<()> {
        for (weights_matrix, bias_vec) in self.info.weights.iter().zip(self.info.bias.iter()) {
            let offset = self.weights_data.len() as u32;
            self.weights_offsets.push(offset);

            for row in weights_matrix {
                self.weights_data.extend_from_slice(row);
            }
            self.weights_data.extend_from_slice(bias_vec);
        }

        if let Some(params) = &self.info.parameters {
            for param_vec in params {
                let offset = self.parameters_data.len() as u32;
                self.params_offsets.push(offset);
                self.parameters_data.extend_from_slice(param_vec);
            }
        }

        Ok(())
    }

    fn build_instructions(&mut self) -> GpuModelResult<()> {
        let info = self.info;
        let sizes = &info.computation_buffer_sizes;
        let parameters = info.parameters.as_deref().unwrap_or(&[]);
        let maps = info.maps.as_deref().unwrap_or(&[]);
        for (idx, inst_info) in info.instructions.iter().enumerate() {
            if let InstructionInfo::Attention(_) | InstructionInfo::MapTransform(_) = inst_info {
                return Err(GpuModelError::UnsupportedInstruction {
                    instruction_type: format!("{inst_info:?}"),
                });
            }
            validation::validate_instruction(
                inst_info,
                sizes,
                &info.weights,
                &info.bias,
                parameters,
                maps,
            )
            .map_err(|e| match e {
                InstructionModelError::WeightsIndexOutOfBounds { .. } => {
                    GpuModelError::MissingWeights {
                        instruction_index: idx,
                    }
                }
                InstructionModelError::ParametersIndexOutOfBounds { .. } => {
                    GpuModelError::MissingParameters {
                        instruction_index: idx,
                    }
                }
                e => GpuModelError::InvalidInstruction {
                    instruction_index: idx,
                    message: e.to_string(),
                },
            })?;
            let gpu_inst = self.convert_instruction(inst_info, idx)?;
            self.instructions.push(gpu_inst);
        }
        Ok(())
    }

    /// Appends bitcast-u32 compute-buffer pointers to the params region and
    /// returns the params-relative offset of the list. Bit-preserving copies
    /// only — the values are never reinterpreted as floats.
    fn push_pointer_payload(&mut self, pointers: &[u32]) -> u32 {
        let offset = self.parameters_data.len() as u32;
        self.parameters_data
            .extend(pointers.iter().map(|&pointer| f32::from_bits(pointer)));
        offset
    }

    /// Stores the input pointer list of an N-ary element-wise buffers op;
    /// returns `(pointers_offset, input_count)`.
    fn buffers_pointer_payload(&mut self, inputs: &[usize]) -> (u32, u32) {
        let pointers: Vec<u32> = inputs.iter().map(|&b| self.ptr(b)).collect();
        (self.push_pointer_payload(&pointers), pointers.len() as u32)
    }

    // Casts below are lossless: `new` bounds every offset by MAX_GPU_COMPUTE_BUFFER.
    fn ptr(&self, buffer: usize) -> u32 {
        self.computation_buffer_indexes[buffer] as u32
    }

    fn size(&self, buffer: usize) -> u32 {
        self.computation_buffer_sizes[buffer] as u32
    }

    /// Converts an instruction already checked by `validate_instruction`.
    fn convert_instruction(
        &mut self,
        info: &InstructionInfo,
        idx: usize,
    ) -> GpuModelResult<GpuInstruction> {
        Ok(match info {
            InstructionInfo::Dot(i) => GpuInstruction::dot(
                self.ptr(i.input),
                self.ptr(i.output),
                self.size(i.output),
                self.weights_offsets[i.weights],
                self.size(i.input),
                i.activation,
            ),
            InstructionInfo::Activation(i) => {
                GpuInstruction::activation(self.ptr(i.input), self.size(i.input), i.activation)
            }
            InstructionInfo::ElemWiseAdd(i) => GpuInstruction::elem_wise_add(
                self.ptr(i.input),
                self.size(i.input),
                self.params_offsets[i.parameters],
            ),
            InstructionInfo::ElemWiseMul(i) => GpuInstruction::elem_wise_mul(
                self.ptr(i.input),
                self.size(i.input),
                self.params_offsets[i.parameters],
            ),
            InstructionInfo::Copy(i) => GpuInstruction::copy(
                self.ptr(i.input),
                self.ptr(i.output) + i.internal_index as u32,
                self.size(i.input),
            ),
            InstructionInfo::CopyMasked(i) => {
                if i.indexes.is_empty() {
                    return Err(GpuModelError::InvalidInstruction {
                        instruction_index: idx,
                        message: "COPY_MASKED requires at least one index".to_string(),
                    });
                }
                let base = self.ptr(i.input);
                let pointers: Vec<u32> = i.indexes.iter().map(|&x| base + x as u32).collect();
                let offset = self.push_pointer_payload(&pointers);
                GpuInstruction::copy_masked(offset, pointers.len() as u32, self.ptr(i.output))
            }
            InstructionInfo::ClipElementwise(i) => GpuInstruction::clip_elementwise(
                self.ptr(i.input),
                self.size(i.input),
                i.parameters_min.map(|p| self.params_offsets[p]),
                i.parameters_max.map(|p| self.params_offsets[p]),
            ),
            InstructionInfo::ElemWiseBuffersAdd(i) => {
                let (offset, count) = self.buffers_pointer_payload(&i.input);
                GpuInstruction::elem_wise_buffers_add(
                    self.ptr(i.output),
                    self.size(i.output),
                    offset,
                    count,
                )
            }
            InstructionInfo::ElemWiseBuffersMul(i) => {
                let (offset, count) = self.buffers_pointer_payload(&i.input);
                GpuInstruction::elem_wise_buffers_mul(
                    self.ptr(i.output),
                    self.size(i.output),
                    offset,
                    count,
                )
            }
            InstructionInfo::MultiplyBufferHeads(i) => GpuInstruction::multiply_buffer_heads(
                self.ptr(i.input[0]),
                self.ptr(i.output),
                self.size(i.input[0]),
                self.ptr(i.input[1]),
                self.size(i.input[1]),
            ),
            InstructionInfo::AddBufferHeads(i) => GpuInstruction::add_buffer_heads(
                self.ptr(i.input[0]),
                self.ptr(i.output),
                self.size(i.input[0]),
                self.ptr(i.input[1]),
                self.size(i.input[1]),
            ),
            InstructionInfo::ReduceSum(i) => GpuInstruction::reduce_sum(
                self.ptr(i.input),
                self.ptr(i.output),
                self.size(i.input),
            ),
            InstructionInfo::Attention(_) | InstructionInfo::MapTransform(_) => {
                unreachable!("rejected by build_instructions")
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::gpu_instruction::{CLIP_BOUND_NONE, opcodes};
    use serde_json::{Value, json};

    fn info(v: Value) -> InstructionModelInfo {
        serde_json::from_value(v).unwrap()
    }

    fn dense_info() -> InstructionModelInfo {
        info(json!({"feature_size": 3, "buffer_sizes": [3, 2, 1],
            "instructions": [
                {"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "RELU"},
                {"type": "DOT", "input": 1, "output": 2, "weights": 1}],
            "weights": [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[1.0, 2.0]]],
            "bias": [[0.1, 0.2], [0.5]]}))
    }

    fn weightless(
        buffer_sizes: Value,
        instruction: Value,
        parameters: Value,
    ) -> InstructionModelInfo {
        info(
            json!({"feature_size": buffer_sizes[0], "buffer_sizes": buffer_sizes,
            "instructions": [instruction], "weights": [], "bias": [], "parameters": parameters}),
        )
    }

    fn instruction_fields(gpu_model: &GpuModel, index: usize) -> Vec<u32> {
        let base = HEADER_SIZE_F32S + index * GpuInstruction::SIZE_U32S;
        gpu_model.as_f32_slice()[base..base + GpuInstruction::SIZE_U32S]
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    #[test]
    fn header_and_theta_region() {
        let mut gpu_model = GpuModel::from_info(&dense_info()).unwrap();
        assert_eq!((gpu_model.feature_size(), gpu_model.output_size()), (3, 1));
        assert_eq!(
            (gpu_model.compute_buffer_size(), gpu_model.output_start()),
            (6, 5)
        );
        let header: Vec<u32> = gpu_model.as_f32_slice()[..HEADER_SIZE_F32S]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(&header[..4], &[GPU_MODEL_MAGIC, GPU_MODEL_VERSION, 3, 1]);
        assert_eq!(header[7] as usize, gpu_model.weights_offset());
        assert_eq!(header[8] as usize, gpu_model.theta_len());
        assert_eq!(header[12] as usize, gpu_model.full_size());

        // Canonical order: per slot, row-major weights then bias.
        let mut theta = vec![0.0f32; 11];
        gpu_model.read_theta(&mut theta).unwrap();
        assert_eq!(
            theta,
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 1.0, 2.0, 0.5]
        );
        let new_theta: Vec<f32> = (0..11).map(|v| v as f32).collect();
        gpu_model.write_theta(&new_theta).unwrap();
        let offset = gpu_model.weights_offset();
        assert_eq!(
            &gpu_model.as_f32_slice()[offset..offset + 11],
            &new_theta[..]
        );
        assert!(matches!(
            gpu_model.write_theta(&[1.0, 2.0]),
            Err(GpuModelError::ThetaLengthMismatch {
                expected: 11,
                got: 2
            })
        ));
    }

    #[test]
    fn copy_masked_payload_holds_absolute_pointers() {
        let gpu_model = GpuModel::from_info(&weightless(
            json!([4, 2]),
            json!({"type": "COPY_MASKED", "input": 0, "output": 1, "indexes": [3, 1]}),
            json!(null),
        ))
        .unwrap();
        let data = gpu_model.as_f32_slice();
        let params_offset = data[9].to_bits() as usize;
        assert_eq!(data[10].to_bits(), 2);
        assert_eq!(data[params_offset].to_bits(), 3);
        assert_eq!(data[params_offset + 1].to_bits(), 1);
        assert_eq!(
            instruction_fields(&gpu_model, 0)[..5],
            [opcodes::COPY_MASKED, 0, 4, 2, 0]
        );
    }

    #[test]
    fn clip_missing_bound_uses_sentinel() {
        let gpu_model = GpuModel::from_info(&weightless(
            json!([2]),
            json!({"type": "CLIP_ELEMENTWISE", "input": 0, "parameters_max": 0}),
            json!([[1.0, 1.0]]),
        ))
        .unwrap();
        let fields = instruction_fields(&gpu_model, 0);
        assert_eq!(
            (fields[0], fields[4], fields[5]),
            (opcodes::CLIP_ELEMENTWISE, CLIP_BOUND_NONE, 0)
        );
    }

    #[test]
    fn rejects_malformed_models() {
        use GpuModelError as E;
        type Case = (InstructionModelInfo, fn(&E) -> bool);
        let cases: Vec<Case> = vec![
            (
                weightless(
                    json!([3, 3]),
                    json!({"type": "ADD_ELEMENTWISE_BUFFERS", "input": [0], "output": 1}),
                    json!(null),
                ),
                |e| {
                    matches!(
                        e,
                        E::InvalidInstruction {
                            instruction_index: 0,
                            ..
                        }
                    )
                },
            ),
            (
                weightless(
                    json!([3, 2, 3]),
                    json!({"type": "MULTIPLY_BUFFER_HEADS", "input": [0, 1], "output": 2}),
                    json!(null),
                ),
                |e| {
                    matches!(
                        e,
                        E::InvalidInstruction {
                            instruction_index: 0,
                            ..
                        }
                    )
                },
            ),
            (
                weightless(
                    json!([2, 3]),
                    json!({"type": "COPY", "input": 0, "output": 1, "internal_index": 2}),
                    json!(null),
                ),
                |e| {
                    matches!(
                        e,
                        E::InvalidInstruction {
                            instruction_index: 0,
                            ..
                        }
                    )
                },
            ),
            (
                weightless(
                    json!([2, 3]),
                    json!({"type": "COPY", "input": 0, "output": 1, "internal_index": u64::MAX}),
                    json!(null),
                ),
                |e| {
                    matches!(
                        e,
                        E::InvalidInstruction {
                            instruction_index: 0,
                            ..
                        }
                    )
                },
            ),
            (
                weightless(
                    json!([2, 2]),
                    json!({"type": "COPY", "input": 0, "output": 5, "internal_index": 0}),
                    json!(null),
                ),
                |e| {
                    matches!(
                        e,
                        E::InvalidInstruction {
                            instruction_index: 0,
                            ..
                        }
                    )
                },
            ),
            (
                weightless(
                    json!([2]),
                    json!({"type": "ADD_ELEMENTWISE", "input": 0, "parameters": 3}),
                    json!(null),
                ),
                |e| {
                    matches!(
                        e,
                        E::MissingParameters {
                            instruction_index: 0
                        }
                    )
                },
            ),
            (
                weightless(
                    json!([2, 1]),
                    json!({"type": "DOT", "input": 0, "output": 1, "weights": 0}),
                    json!(null),
                ),
                |e| {
                    matches!(
                        e,
                        E::MissingWeights {
                            instruction_index: 0
                        }
                    )
                },
            ),
            (
                weightless(
                    json!([2, 2]),
                    json!({"type": "ATTENTION", "input": 0, "key": 0, "output": 1, "weights": 0}),
                    json!(null),
                ),
                |e| matches!(e, E::UnsupportedInstruction { .. }),
            ),
            (
                weightless(
                    json!([2, u64::MAX, 2]),
                    json!({"type": "COPY", "input": 0, "output": 2, "internal_index": 0}),
                    json!(null),
                ),
                |e| matches!(e, E::ComputeBufferTooLarge { .. }),
            ),
            (
                info(
                    json!({"feature_size": 1, "buffer_sizes": [], "instructions": [], "weights": [], "bias": []}),
                ),
                |e| matches!(e, E::InvalidModelBinary { .. }),
            ),
        ];
        for (model_info, check) in cases {
            let err = GpuModel::from_info(&model_info).unwrap_err();
            assert!(check(&err), "{:?}: {err}", model_info.instructions);
        }
    }
}
