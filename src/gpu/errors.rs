//! GPU-specific error types for the neural inference library.

use crate::errors::GraphError;
use thiserror::Error;

/// Errors specific to GPU model operations.
///
/// Marked `#[non_exhaustive]` since 1.0.0 so future variants can be added without
/// a breaking change; match with a wildcard arm.
#[derive(Error, Debug)]
#[non_exhaustive]
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

    #[error("Invalid instruction at index {instruction_index}: {message}")]
    InvalidInstruction {
        instruction_index: usize,
        message: String,
    },

    #[error("Missing weights for instruction at index {instruction_index}")]
    MissingWeights { instruction_index: usize },

    #[error("Missing parameters for instruction at index {instruction_index}")]
    MissingParameters { instruction_index: usize },

    #[error("Flat parameter vector length mismatch: expected {expected} but got {got}")]
    ThetaLengthMismatch { expected: usize, got: usize },

    #[error("Serialized weights region is not in canonical flat-theta order: {message}")]
    PackedThetaOrderViolation { message: String },

    #[error("Population must contain at least one candidate")]
    EmptyPopulation,

    #[error("Candidate index {index} out of bounds for population of {count}")]
    CandidateOutOfBounds { index: usize, count: usize },
}

pub type GpuModelResult<T> = std::result::Result<T, GpuModelError>;

/// Errors from population workflows that start at a graph, which can fail in
/// either the graph-compilation or the GPU-serialization domain.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum PopulationError {
    #[error(transparent)]
    Graph(#[from] GraphError),

    #[error(transparent)]
    Gpu(#[from] GpuModelError),
}

/// Errors from the wgpu-backed evaluation host (`gpu-runtime` feature).
#[cfg(feature = "gpu-runtime")]
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum GpuRuntimeError {
    #[error("No GPU adapter available")]
    NoAdapter,

    #[error("Adapter '{name}' is a software/CPU device; set allow_software_adapter to accept it")]
    SoftwareAdapterRejected { name: String },

    #[error("GPU device request failed: {message}")]
    DeviceRequestFailed { message: String },

    #[error("Shader compilation failed: {message}")]
    ShaderCompilationFailed { message: String },

    #[error("Dispatch of {workgroups} workgroups exceeds the device limit of {max}")]
    DispatchTooLarge { workgroups: u32, max: u32 },

    #[error("Buffer of {required} bytes exceeds the device storage-binding limit of {max} bytes")]
    BufferLimitExceeded { required: u64, max: u64 },

    #[error("Batch size must be at least 1")]
    ZeroBatchSize,

    #[error("Input length mismatch: expected {expected} f32s but got {got}")]
    InputLengthMismatch { expected: usize, got: usize },

    #[error("Output length mismatch: expected {expected} f32s but got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("Population length mismatch: expected {expected} values but got {got}")]
    PopulationLengthMismatch { expected: usize, got: usize },

    #[error(transparent)]
    Model(#[from] GpuModelError),

    #[error(transparent)]
    Population(#[from] PopulationError),
}
