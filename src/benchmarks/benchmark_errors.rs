//! Error types for benchmark operations.

use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum BenchmarkError {
    ConfigFileNotFound {
        path: String,
    },
    ConfigParseError {
        path: String,
        source: serde_json::Error,
    },
    ConfigValidationError {
        field: String,
        message: String,
    },
    IoError {
        source: std::io::Error,
    },
    InvalidBufferSize {
        size: usize,
    },
    InvalidNumExecutions {
        value: u32,
    },
    InvalidInputSize {
        provided: usize,
        expected: usize,
    },
    BenchmarkExecutionError {
        benchmark_name: String,
        message: String,
    },
    InvalidOperationType {
        operation: String,
    },
}

impl fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchmarkError::ConfigFileNotFound { path } => {
                write!(f, "Configuration file not found: {}", path)
            }
            BenchmarkError::ConfigParseError { path, source } => {
                write!(
                    f,
                    "Failed to parse configuration file '{}': {}",
                    path, source
                )
            }
            BenchmarkError::ConfigValidationError { field, message } => {
                write!(
                    f,
                    "Configuration validation error for field '{}': {}",
                    field, message
                )
            }
            BenchmarkError::IoError { source } => {
                write!(f, "IO error: {}", source)
            }
            BenchmarkError::InvalidBufferSize { size } => {
                write!(f, "Invalid buffer size: {}. Must be greater than 0", size)
            }
            BenchmarkError::InvalidNumExecutions { value } => {
                write!(
                    f,
                    "Invalid number of executions: {}. Must be greater than 0",
                    value
                )
            }
            BenchmarkError::InvalidInputSize { provided, expected } => {
                write!(
                    f,
                    "Invalid input size: provided {}, expected {}",
                    provided, expected
                )
            }
            BenchmarkError::BenchmarkExecutionError {
                benchmark_name,
                message,
            } => {
                write!(
                    f,
                    "Benchmark '{}' execution error: {}",
                    benchmark_name, message
                )
            }
            BenchmarkError::InvalidOperationType { operation } => {
                write!(
                    f,
                    "Invalid operation type: '{}'. Supported: add, multiply",
                    operation
                )
            }
        }
    }
}

impl Error for BenchmarkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            BenchmarkError::ConfigParseError { source, .. } => Some(source),
            BenchmarkError::IoError { source } => Some(source),
            _ => None,
        }
    }
}

impl From<std::io::Error> for BenchmarkError {
    fn from(error: std::io::Error) -> Self {
        BenchmarkError::IoError { source: error }
    }
}

pub type BenchmarkResult<T> = Result<T, BenchmarkError>;
