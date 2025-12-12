use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParallelPredictError {
    #[error(
        "Input buffer size mismatch: expected {expected} f32 values ({num_samples} samples * {feature_size} features), got {actual}"
    )]
    InputBufferSizeMismatch {
        expected: usize,
        actual: usize,
        num_samples: usize,
        feature_size: usize,
    },

    #[error("Invalid slice range: start ({start}) must be less than end ({end})")]
    InvalidSliceRange { start: usize, end: usize },

    #[error("Slice range out of bounds: range [{start}..{end}) exceeds buffer size {buffer_size}")]
    SliceRangeOutOfBounds {
        start: usize,
        end: usize,
        buffer_size: usize,
    },

    #[error("Thread count must be at least 1, got {count}")]
    InvalidThreadCount { count: usize },

    #[error("Result index {index} is out of bounds for {num_samples} samples")]
    ResultIndexOutOfBounds { index: usize, num_samples: usize },

    #[error("Destination buffer size mismatch: expected {expected}, got {actual}")]
    DestinationBufferSizeMismatch { expected: usize, actual: usize },

    #[error("Model prediction failed for sample {sample_index}: {message}")]
    PredictionFailed {
        sample_index: usize,
        message: String,
    },

    #[error("Thread panicked during parallel execution")]
    ThreadPanicked,
}

pub type ParallelPredictResult<T> = std::result::Result<T, ParallelPredictError>;
