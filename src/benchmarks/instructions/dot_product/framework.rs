use crate::benchmarks::benchmark_errors::{BenchmarkError, BenchmarkResult};
use crate::benchmarks::benchmark_types::DotProductConfig;
use crate::instruction_model_info::{DotInstructionInfo, InstructionInfo, InstructionModelInfo};
use crate::{Activation, InstructionModel};

pub struct FrameworkDotProduct {
    model: InstructionModel,
    input_size: usize,
}

impl FrameworkDotProduct {
    pub fn new(
        config: &DotProductConfig,
        weights_layer1: &[Vec<f32>],
        bias_layer1: &[f32],
        weights_layer2: &[Vec<f32>],
        bias_layer2: &[f32],
    ) -> BenchmarkResult<Self> {
        let computation_buffer_sizes = vec![
            config.network_config.input_size,
            config.network_config.hidden_size,
            config.network_config.output_size,
        ];

        let instructions = vec![
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
        ];

        let model_info = InstructionModelInfo {
            features: None,
            feature_size: Some(config.network_config.input_size),
            computation_buffer_sizes,
            instructions,
            weights: vec![weights_layer1.to_vec(), weights_layer2.to_vec()],
            bias: vec![bias_layer1.to_vec(), bias_layer2.to_vec()],
            parameters: None,
            maps: None,
            validation_data: None,
        };

        let model = InstructionModel::new(model_info).map_err(|e| {
            BenchmarkError::BenchmarkExecutionError {
                benchmark_name: "dot_product".to_string(),
                message: format!("Failed to create instruction model: {}", e),
            }
        })?;

        Ok(Self {
            model,
            input_size: config.network_config.input_size,
        })
    }

    pub fn required_memory(&self) -> usize {
        self.model.required_memory()
    }

    pub fn run_with_buffer<'a>(
        &'a self,
        input: &[f32],
        unified_buffer: &'a mut [f32],
    ) -> BenchmarkResult<&'a [f32]> {
        if input.len() != self.input_size {
            return Err(BenchmarkError::BenchmarkExecutionError {
                benchmark_name: "dot_product".to_string(),
                message: format!(
                    "Input size mismatch: expected {}, got {}",
                    self.input_size,
                    input.len()
                ),
            });
        }

        if unified_buffer.len() < self.required_memory() {
            return Err(BenchmarkError::BenchmarkExecutionError {
                benchmark_name: "dot_product".to_string(),
                message: format!(
                    "Buffer too small: required {}, got {}",
                    self.required_memory(),
                    unified_buffer.len()
                ),
            });
        }

        unified_buffer[..self.input_size].copy_from_slice(input);

        self.model
            .predict_with_buffer(unified_buffer)
            .map_err(|e| BenchmarkError::BenchmarkExecutionError {
                benchmark_name: "dot_product".to_string(),
                message: format!("Prediction with buffer failed: {}", e),
            })?;

        let output_start = self.model.get_output_index_start();
        let output_size = self.model.get_output_size();
        Ok(&unified_buffer[output_start..output_start + output_size])
    }
}
