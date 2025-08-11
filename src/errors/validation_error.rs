//! Error types for validation operations.

use thiserror::Error;

/// Errors that can occur during validation operations.
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Validation failed: {message}")]
    ValidationFailed { message: String },

    #[error("Invalid validation data: {reason}")]
    InvalidValidationData { reason: String },

    #[error(
        "Validation tolerance exceeded: expected {expected}, got {actual}, tolerance {tolerance}"
    )]
    ToleranceExceeded {
        expected: f32,
        actual: f32,
        tolerance: f32,
    },
}
