//! Error types for the neural inference library.
//!
//! This module contains specific error types used throughout the library,
//! avoiding generic error wrappers like `anyhow` or `Box<dyn Error>` for better
//! error handling and debugging.

mod instruction_model_error;
mod validation_error;

pub use instruction_model_error::{
    BufferIndexOutOfBoundsError, ComputationBufferSizeExceedsLimitError, FeatureSizeMismatchError,
    InstructionModelError, InvalidFeatureSizeError, UnusedComputationError,
    ValidationInputOutputMismatchError,
};
pub use validation_error::ValidationError;

/// Result type alias for operations that may fail with neural inference errors.
pub type InstructionModelResult<T> = std::result::Result<T, InstructionModelError>;

/// Result type alias for validation operations.
pub type ValidationResult<T> = std::result::Result<T, ValidationError>;
