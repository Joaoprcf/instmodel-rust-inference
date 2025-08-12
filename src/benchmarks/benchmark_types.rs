//! Benchmark type definitions and configuration structures.

use super::benchmark_errors::{BenchmarkError, BenchmarkResult};
use serde::{Deserialize, Serialize};

/// Configuration for dot product neural network benchmark
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DotProductConfig {
    pub name: String,
    pub description: String,
    pub network_config: NetworkConfig,
    pub num_executions: u32,
}

/// Neural network configuration for dot product benchmark
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

/// Configuration for element-wise operations benchmark
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ElementWiseOpsConfig {
    pub name: String,
    pub description: String,
    pub buffer_sizes: Vec<usize>,
    pub input_size: usize,
    pub operations: Vec<String>,
    pub num_executions: u32,
}

/// Enum representing all available benchmark types
#[derive(Debug, Clone)]
pub enum BenchmarkConfig {
    DotProduct(DotProductConfig),
    ElementWiseOps(ElementWiseOpsConfig),
}

impl DotProductConfig {
    /// Creates default configuration for dot product benchmark
    pub fn default() -> Self {
        Self {
            name: "dot_product".to_string(),
            description: "Neural network with dot product operations".to_string(),
            network_config: NetworkConfig {
                input_size: 2,
                hidden_size: 10000,
                output_size: 10,
            },
            num_executions: 1000,
        }
    }

    /// Validates the configuration
    pub fn validate(&self) -> BenchmarkResult<()> {
        if self.num_executions == 0 {
            return Err(BenchmarkError::InvalidNumExecutions {
                value: self.num_executions,
            });
        }

        if self.network_config.input_size == 0 {
            return Err(BenchmarkError::ConfigValidationError {
                field: "network_config.input_size".to_string(),
                message: "Input size must be greater than 0".to_string(),
            });
        }

        if self.network_config.hidden_size == 0 {
            return Err(BenchmarkError::ConfigValidationError {
                field: "network_config.hidden_size".to_string(),
                message: "Hidden size must be greater than 0".to_string(),
            });
        }

        if self.network_config.output_size == 0 {
            return Err(BenchmarkError::ConfigValidationError {
                field: "network_config.output_size".to_string(),
                message: "Output size must be greater than 0".to_string(),
            });
        }

        Ok(())
    }
}

impl ElementWiseOpsConfig {
    /// Creates default configuration for element-wise operations benchmark
    pub fn default() -> Self {
        Self {
            name: "element_wise_ops".to_string(),
            description: "Element-wise addition and multiplication operations".to_string(),
            buffer_sizes: vec![1000, 1000, 1000, 1000],
            input_size: 2000,
            operations: vec!["add".to_string(), "multiply".to_string()],
            num_executions: 1000,
        }
    }

    /// Validates the configuration
    pub fn validate(&self) -> BenchmarkResult<()> {
        if self.num_executions == 0 {
            return Err(BenchmarkError::InvalidNumExecutions {
                value: self.num_executions,
            });
        }

        if self.buffer_sizes.len() < 4 {
            return Err(BenchmarkError::ConfigValidationError {
                field: "buffer_sizes".to_string(),
                message: "Must have at least 4 buffers for element-wise operations".to_string(),
            });
        }

        for (_i, &size) in self.buffer_sizes.iter().enumerate() {
            if size == 0 {
                return Err(BenchmarkError::InvalidBufferSize { size });
            }
        }

        if self.input_size == 0 {
            return Err(BenchmarkError::ConfigValidationError {
                field: "input_size".to_string(),
                message: "Input size must be greater than 0".to_string(),
            });
        }

        // Validate that input size matches the first two buffer sizes
        let expected_input_size = self.buffer_sizes[0] + self.buffer_sizes[1];
        if self.input_size != expected_input_size {
            return Err(BenchmarkError::InvalidInputSize {
                provided: self.input_size,
                expected: expected_input_size,
            });
        }

        // Validate operations
        for operation in &self.operations {
            match operation.as_str() {
                "add" | "multiply" => {}
                _ => {
                    return Err(BenchmarkError::InvalidOperationType {
                        operation: operation.clone(),
                    });
                }
            }
        }

        // Validate that buffer[2] and buffer[3] have the same size for add/multiply operations
        if self.buffer_sizes[2] != self.buffer_sizes[0]
            || self.buffer_sizes[3] != self.buffer_sizes[0]
        {
            return Err(BenchmarkError::ConfigValidationError {
                field: "buffer_sizes".to_string(),
                message: "Buffers 0, 2, and 3 must have the same size for element-wise operations"
                    .to_string(),
            });
        }

        Ok(())
    }
}

impl BenchmarkConfig {
    /// Gets the benchmark name
    pub fn name(&self) -> &str {
        match self {
            BenchmarkConfig::DotProduct(config) => &config.name,
            BenchmarkConfig::ElementWiseOps(config) => &config.name,
        }
    }

    /// Gets the benchmark description
    pub fn description(&self) -> &str {
        match self {
            BenchmarkConfig::DotProduct(config) => &config.description,
            BenchmarkConfig::ElementWiseOps(config) => &config.description,
        }
    }

    /// Gets the number of executions
    pub fn num_executions(&self) -> u32 {
        match self {
            BenchmarkConfig::DotProduct(config) => config.num_executions,
            BenchmarkConfig::ElementWiseOps(config) => config.num_executions,
        }
    }

    /// Validates the configuration
    pub fn validate(&self) -> BenchmarkResult<()> {
        match self {
            BenchmarkConfig::DotProduct(config) => config.validate(),
            BenchmarkConfig::ElementWiseOps(config) => config.validate(),
        }
    }
}

/// Performance measurement structure
#[derive(Debug, Clone)]
pub struct PerformanceResults {
    pub method: String,
    pub total_time_ns: u128,
    pub average_time_ns: u128,
    pub average_time_ms: f64,
    pub num_executions: u32,
}

impl PerformanceResults {
    pub fn new(method: String, total_time_ns: u128, num_executions: u32) -> Self {
        let average_time_ns = total_time_ns / num_executions as u128;
        let average_time_ms = average_time_ns as f64 / 1_000_000.0;

        Self {
            method,
            total_time_ns,
            average_time_ns,
            average_time_ms,
            num_executions,
        }
    }

    pub fn overhead_ratio(&self, baseline: &PerformanceResults) -> f64 {
        self.average_time_ns as f64 / baseline.average_time_ns as f64
    }

    pub fn overhead_percentage(&self, baseline: &PerformanceResults) -> f64 {
        (self.overhead_ratio(baseline) - 1.0) * 100.0
    }
}
