//! GPU model serialization and management.

use crate::gpu::errors::{GpuModelError, GpuModelResult};
use crate::gpu::gpu_instruction::GpuInstruction;
use crate::instruction_model_info::{InstructionInfo, InstructionModelInfo};

/// Magic number for GPU model binary format ("GPMI" in little-endian).
pub const GPU_MODEL_MAGIC: u32 = 0x494D5047;

/// Format version.
pub const GPU_MODEL_VERSION: u32 = 1;

/// Header size in f32s (13 fields).
pub const HEADER_SIZE_F32S: usize = 13;

/// Maximum compute buffer size for GPU (in f32s).
pub const MAX_GPU_COMPUTE_BUFFER: usize = 65536;

/// Represents a model serialized for GPU execution.
///
/// All data is packed into a single contiguous f32 array for simple memory access.
#[derive(Debug, Clone)]
pub struct GpuModel {
    /// Single contiguous buffer containing header, instructions, weights, and parameters.
    data: Vec<f32>,
    /// Feature size (input size).
    feature_size: usize,
    /// Output size.
    output_size: usize,
    /// Required compute buffer size.
    compute_buffer_size: usize,
    /// Output start index in compute buffer.
    output_start: usize,
}

impl GpuModel {
    /// Create a GPU model from an InstructionModelInfo.
    pub fn from_info(info: &InstructionModelInfo) -> GpuModelResult<Self> {
        let mut builder = GpuModelBuilder::new(info)?;
        builder.build()
    }

    /// Get the packed f32 data.
    pub fn as_f32_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get the packed f32 data as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }

    /// Get the total model size in f32s (for multi-model indexing).
    pub fn full_size(&self) -> usize {
        self.data.len()
    }

    /// Get feature size (input size).
    pub fn feature_size(&self) -> usize {
        self.feature_size
    }

    /// Get output size.
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Get required compute buffer size.
    pub fn compute_buffer_size(&self) -> usize {
        self.compute_buffer_size
    }

    /// Get output start index in compute buffer.
    pub fn output_start(&self) -> usize {
        self.output_start
    }
}

/// Builder for constructing GPU models.
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
        let mut computation_buffer_indexes = vec![0usize];
        let mut current_idx = computation_buffer_sizes[0];

        for &size in computation_buffer_sizes.iter().skip(1) {
            computation_buffer_indexes.push(current_idx);
            current_idx += size;
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
        // First pass: collect weights and parameters with offsets
        self.collect_weights_and_params()?;

        // Second pass: build instructions
        self.build_instructions()?;

        // Calculate sizes
        let compute_buffer_size = self.computation_buffer_indexes.last().unwrap()
            + self.computation_buffer_sizes.last().unwrap();
        let output_start = *self.computation_buffer_indexes.last().unwrap();
        let output_size = *self.computation_buffer_sizes.last().unwrap();
        let feature_size = self.calculate_feature_size()?;

        // Calculate offsets (in f32 units)
        let instructions_offset = HEADER_SIZE_F32S;
        let weights_offset =
            instructions_offset + self.instructions.len() * GpuInstruction::SIZE_U32S;
        let params_offset = weights_offset + self.weights_data.len();

        // Build header (13 f32s, stored as bitcast u32)
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

        // full_model_size will be set after we know the total size
        let full_model_size_index = data.len();
        data.push(0.0); // Placeholder

        // Add instructions
        for inst in &self.instructions {
            data.extend_from_slice(&inst.to_f32_array());
        }

        // Add weights
        data.extend_from_slice(&self.weights_data);

        // Add parameters
        data.extend_from_slice(&self.parameters_data);

        // Update full_model_size
        data[full_model_size_index] = f32::from_bits(data.len() as u32);

        Ok(GpuModel {
            data,
            feature_size,
            output_size,
            compute_buffer_size,
            output_start,
        })
    }

    fn calculate_feature_size(&self) -> GpuModelResult<usize> {
        if let Some(size) = self.info.feature_size {
            return Ok(size);
        }
        if let Some(features) = &self.info.features {
            let mut total = 0;
            for feature in features {
                if let Some(open) = feature.find('[') {
                    if feature.ends_with(']') {
                        let num_str = &feature[open + 1..feature.len() - 1];
                        total += num_str.parse::<usize>().unwrap_or(1);
                    } else {
                        total += 1;
                    }
                } else {
                    total += 1;
                }
            }
            return Ok(total);
        }
        Ok(self.computation_buffer_sizes[0])
    }

    fn collect_weights_and_params(&mut self) -> GpuModelResult<()> {
        // Collect weights: flatten each weight matrix and append bias
        for (weights_matrix, bias_vec) in self.info.weights.iter().zip(self.info.bias.iter()) {
            let offset = self.weights_data.len() as u32;
            self.weights_offsets.push(offset);

            // Flatten weights row-major
            for row in weights_matrix {
                self.weights_data.extend_from_slice(row);
            }
            // Append bias
            self.weights_data.extend_from_slice(bias_vec);
        }

        // Collect parameters
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
        for (idx, inst_info) in self.info.instructions.iter().enumerate() {
            let gpu_inst = self.convert_instruction(inst_info, idx)?;
            self.instructions.push(gpu_inst);
        }
        Ok(())
    }

    fn convert_instruction(
        &self,
        info: &InstructionInfo,
        idx: usize,
    ) -> GpuModelResult<GpuInstruction> {
        match info {
            InstructionInfo::Dot(dot_info) => {
                let input_ptr = self.computation_buffer_indexes[dot_info.input] as u32;
                let output_ptr = self.computation_buffer_indexes[dot_info.output] as u32;
                let output_size = self.computation_buffer_sizes[dot_info.output] as u32;
                let input_size = self.computation_buffer_sizes[dot_info.input] as u32;

                let weights_offset = self.weights_offsets.get(dot_info.weights).copied().ok_or(
                    GpuModelError::MissingWeights {
                        instruction_index: idx,
                    },
                )?;

                Ok(GpuInstruction::dot(
                    input_ptr,
                    output_ptr,
                    output_size,
                    weights_offset,
                    input_size,
                    dot_info.activation,
                ))
            }
            InstructionInfo::Activation(act_info) => {
                let ptr = self.computation_buffer_indexes[act_info.input] as u32;
                let size = self.computation_buffer_sizes[act_info.input] as u32;

                Ok(GpuInstruction::activation(ptr, size, act_info.activation))
            }
            InstructionInfo::ElemWiseAdd(add_info) => {
                let ptr = self.computation_buffer_indexes[add_info.input] as u32;
                let size = self.computation_buffer_sizes[add_info.input] as u32;

                let params_offset = self
                    .params_offsets
                    .get(add_info.parameters)
                    .copied()
                    .ok_or(GpuModelError::MissingParameters {
                        instruction_index: idx,
                    })?;

                Ok(GpuInstruction::elem_wise_add(ptr, size, params_offset))
            }
            InstructionInfo::ElemWiseMul(mul_info) => {
                let ptr = self.computation_buffer_indexes[mul_info.input] as u32;
                let size = self.computation_buffer_sizes[mul_info.input] as u32;

                let params_offset = self
                    .params_offsets
                    .get(mul_info.parameters)
                    .copied()
                    .ok_or(GpuModelError::MissingParameters {
                        instruction_index: idx,
                    })?;

                Ok(GpuInstruction::elem_wise_mul(ptr, size, params_offset))
            }
            InstructionInfo::Copy(copy_info) => {
                let src_ptr = self.computation_buffer_indexes[copy_info.input] as u32;
                let dst_ptr = self.computation_buffer_indexes[copy_info.output] as u32
                    + copy_info.internal_index as u32;
                let size = self.computation_buffer_sizes[copy_info.input] as u32;

                Ok(GpuInstruction::copy(src_ptr, dst_ptr, size))
            }
            _ => Err(GpuModelError::UnsupportedInstruction {
                instruction_type: format!("{:?}", info),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction_model_info::DotInstructionInfo;

    #[test]
    fn test_simple_model_serialization() {
        let info = InstructionModelInfo {
            features: Some(vec!["f1".to_string(), "f2".to_string()]),
            feature_size: Some(2),
            computation_buffer_sizes: vec![2, 1],
            instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
                input: 0,
                output: 1,
                weights: 0,
                activation: None,
            })],
            weights: vec![vec![vec![1.0, 2.0]]],
            bias: vec![vec![0.5]],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let gpu_model = GpuModel::from_info(&info).expect("Failed to create GPU model");

        assert_eq!(gpu_model.feature_size(), 2);
        assert_eq!(gpu_model.output_size(), 1);
        assert_eq!(gpu_model.compute_buffer_size(), 3);
        assert_eq!(gpu_model.output_start(), 2);

        // Verify header
        let data = gpu_model.as_f32_slice();
        assert_eq!(data[0].to_bits(), GPU_MODEL_MAGIC);
        assert_eq!(data[1].to_bits(), GPU_MODEL_VERSION);
        assert_eq!(data[2].to_bits(), 2); // feature_size
        assert_eq!(data[3].to_bits(), 1); // output_size
    }

    #[test]
    fn test_multi_layer_model() {
        use crate::activation::Activation;

        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(3),
            computation_buffer_sizes: vec![3, 4, 2],
            instructions: vec![
                InstructionInfo::Dot(DotInstructionInfo {
                    input: 0,
                    output: 1,
                    weights: 0,
                    activation: Some(Activation::Relu),
                }),
                InstructionInfo::Dot(DotInstructionInfo {
                    input: 1,
                    output: 2,
                    weights: 1,
                    activation: Some(Activation::Sigmoid),
                }),
            ],
            weights: vec![
                vec![
                    vec![0.1, 0.2, 0.3],
                    vec![0.4, 0.5, 0.6],
                    vec![0.7, 0.8, 0.9],
                    vec![1.0, 1.1, 1.2],
                ],
                vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.5, 0.6, 0.7, 0.8]],
            ],
            bias: vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.01, 0.02]],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let gpu_model = GpuModel::from_info(&info).expect("Failed to create GPU model");

        assert_eq!(gpu_model.feature_size(), 3);
        assert_eq!(gpu_model.output_size(), 2);
        assert_eq!(gpu_model.compute_buffer_size(), 9); // 3 + 4 + 2
    }

    #[test]
    fn test_full_model_size_in_header() {
        let info = InstructionModelInfo {
            features: None,
            feature_size: Some(2),
            computation_buffer_sizes: vec![2, 1],
            instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
                input: 0,
                output: 1,
                weights: 0,
                activation: None,
            })],
            weights: vec![vec![vec![1.0, 2.0]]],
            bias: vec![vec![0.5]],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let gpu_model = GpuModel::from_info(&info).expect("Failed to create GPU model");
        let data = gpu_model.as_f32_slice();

        // Header field 12 should contain full_model_size
        let full_model_size_from_header = data[12].to_bits() as usize;
        assert_eq!(full_model_size_from_header, gpu_model.full_size());
    }
}
