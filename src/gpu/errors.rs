//! GPU-specific error types for the neural inference library.

use thiserror::Error;

/// Errors specific to GPU model operations.
#[derive(Error, Debug)]
pub enum GpuModelError {
    #[error("Model too large for GPU: {model_size} f32s exceeds limit of {max_size} f32s")]
    ModelTooLarge { model_size: usize, max_size: usize },

    #[error("Compute buffer size {required} exceeds GPU maximum {max_size}")]
    ComputeBufferTooLarge { required: usize, max_size: usize },

    #[error("Unsupported instruction type for GPU: {instruction_type}")]
    UnsupportedInstruction { instruction_type: String },

    #[error("Invalid activation type: {activation_id}")]
    InvalidActivation { activation_id: u32 },

    #[error("Shader compilation failed: {message}")]
    ShaderCompilationFailed { message: String },

    #[error("GPU device not available: {message}")]
    DeviceNotAvailable { message: String },

    #[error("Buffer creation failed: {message}")]
    BufferCreationFailed { message: String },

    #[error("Invalid model binary: {message}")]
    InvalidModelBinary { message: String },

    #[error("Missing weights for instruction at index {instruction_index}")]
    MissingWeights { instruction_index: usize },

    #[error("Missing parameters for instruction at index {instruction_index}")]
    MissingParameters { instruction_index: usize },
}

pub type GpuModelResult<T> = std::result::Result<T, GpuModelError>;
