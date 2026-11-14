//! Error types.

mod evolution_error;
mod graph_error;
mod instruction_model_error;
mod parallel_predict_error;
mod validation_error;

pub use evolution_error::EvolutionError;
pub use graph_error::GraphError;
pub use instruction_model_error::{
    BufferIndexOutOfBoundsError, ComputationBufferSizeExceedsLimitError, FeatureSizeMismatchError,
    InstructionModelError, InvalidFeatureSizeError, UnusedComputationError,
    ValidationInputOutputMismatchError,
};
pub use parallel_predict_error::{ParallelPredictError, ParallelPredictResult};
pub use validation_error::ValidationError;

/// Result type alias for operations that may fail with neural inference errors.
pub type InstructionModelResult<T> = std::result::Result<T, InstructionModelError>;

/// Result type alias for validation operations.
pub type ValidationResult<T> = std::result::Result<T, ValidationError>;
