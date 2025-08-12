use crate::InstructionModel;
use crate::benchmarks::benchmark_errors::{BenchmarkError, BenchmarkResult};
use crate::benchmarks::benchmark_types::ElementWiseOpsConfig;
use crate::instruction_model_info::{
    ElemWiseBuffersAddInstructionInfo, ElemWiseBuffersMulInstructionInfo, InstructionInfo,
    InstructionModelInfo,
};

pub struct FrameworkElementWiseOps {
    model: InstructionModel,
    buffer_sizes: Vec<usize>,
    input_size: usize,
}

impl FrameworkElementWiseOps {
    pub fn new(config: &ElementWiseOpsConfig) -> BenchmarkResult<Self> {
        let instructions = vec![
            InstructionInfo::ElemWiseBuffersAdd(ElemWiseBuffersAddInstructionInfo {
                input: vec![0, 1],
                output: 2,
            }),
            InstructionInfo::ElemWiseBuffersMul(ElemWiseBuffersMulInstructionInfo {
                input: vec![0, 1],
                output: 3,
            }),
        ];

        let model_info = InstructionModelInfo {
            features: None,
            feature_size: Some(config.input_size),
            computation_buffer_sizes: config.buffer_sizes.clone(),
            instructions,
            weights: vec![],
            bias: vec![],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let model = InstructionModel::new(model_info).map_err(|e| {
            BenchmarkError::BenchmarkExecutionError {
                benchmark_name: "element_wise_buffer_ops".to_string(),
                message: format!("Failed to create instruction model: {}", e),
            }
        })?;

        Ok(Self {
            model,
            buffer_sizes: config.buffer_sizes.clone(),
            input_size: config.input_size,
        })
    }

    pub fn required_memory(&self) -> usize {
        self.model.required_memory()
    }

    pub fn run_with_buffer<'a>(
        &'a self,
        input: &[f32],
        unified_buffer: &'a mut [f32],
    ) -> BenchmarkResult<(&'a [f32], &'a [f32])> {
        if input.len() != self.input_size {
            return Err(BenchmarkError::BenchmarkExecutionError {
                benchmark_name: "element_wise_buffer_ops".to_string(),
                message: format!(
                    "Input size mismatch: expected {}, got {}",
                    self.input_size,
                    input.len()
                ),
            });
        }

        if unified_buffer.len() < self.required_memory() {
            return Err(BenchmarkError::BenchmarkExecutionError {
                benchmark_name: "element_wise_buffer_ops".to_string(),
                message: format!(
                    "Buffer too small: required {}, got {}",
                    self.required_memory(),
                    unified_buffer.len()
                ),
            });
        }

        unified_buffer[..input.len()].copy_from_slice(input);

        self.model
            .predict_with_buffer(unified_buffer)
            .map_err(|e| BenchmarkError::BenchmarkExecutionError {
                benchmark_name: "element_wise_buffer_ops".to_string(),
                message: format!("Computation with buffer failed: {}", e),
            })?;

        let buffer0_size = self.buffer_sizes[0];
        let buffer1_size = self.buffer_sizes[1];
        let buffer2_size = self.buffer_sizes[2];
        let buffer3_size = self.buffer_sizes[3];

        let buffer0_start = 0;
        let buffer1_start = buffer0_start + buffer0_size;
        let buffer2_start = buffer1_start + buffer1_size;
        let buffer3_start = buffer2_start + buffer2_size;

        let add_slice = &unified_buffer[buffer2_start..buffer2_start + buffer2_size];
        let mul_slice = &unified_buffer[buffer3_start..buffer3_start + buffer3_size];

        Ok((add_slice, mul_slice))
    }
}
