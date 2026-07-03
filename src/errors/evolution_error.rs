//! Errors for the evolution-strategies optimizer.

use thiserror::Error;

/// Errors raised by [`EsOptimizer`](crate::evolution::EsOptimizer) and its
/// configuration.
///
/// Marked `#[non_exhaustive]` since 1.0.0 so future variants can be added
/// without a breaking change; match with a wildcard arm.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum EvolutionError {
    #[error("Invalid ES configuration: {message}")]
    InvalidConfig { message: String },

    #[error("Fitness vector length mismatch: expected {expected} but got {got}")]
    FitnessLengthMismatch { expected: usize, got: usize },

    #[error("Out-of-order call: {message}")]
    CallOrder { message: String },

    #[error("Non-finite fitness at index {index}: {value}")]
    NonFiniteFitness { index: usize, value: f64 },

    #[error("Candidate index {index} out of bounds for population of {count}")]
    CandidateOutOfBounds { index: usize, count: usize },
}
