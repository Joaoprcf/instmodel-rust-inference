//! Errors produced while planning or compiling a [`crate::graph`] model graph.

use thiserror::Error;

use crate::errors::InstructionModelError;

/// Errors from planning or compiling a model graph.
///
/// Marked `#[non_exhaustive]` so future graph validations can add variants
/// without a breaking change.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum GraphError {
    /// The graph declared no inputs.
    #[error("graph declared no inputs")]
    NoInputs,

    /// A declared input has size 0.
    #[error("declared input has size 0")]
    ZeroSizedInput,

    /// The same input buffer was declared more than once in `model()`.
    #[error("the same input buffer was declared more than once")]
    DuplicateInput,

    /// A named input's feature-name count does not match its size.
    #[error("input of size {expected} declares {got} feature names")]
    FeatureNamesLengthMismatch { expected: usize, got: usize },

    /// Feature names must be plain scalar names; brackets denote expansion in
    /// the model format and would corrupt the declared feature size.
    #[error("feature name {name:?} must not contain '[' or ']'")]
    InvalidFeatureName { name: String },

    /// Traversal reached a producerless buffer that was not declared in `model()`.
    #[error("traversal reached an input buffer that was not declared in model()")]
    UndeclaredInput,

    /// A declared input is not reachable from the output.
    #[error("a declared input is not reachable from the output")]
    InputNotVisited,

    /// `theta.len()` did not match the layout's `total`.
    #[error("theta length {got} != expected {expected}")]
    ThetaLenMismatch { expected: usize, got: usize },

    /// One [`crate::graph::WeightId`] was reused with two different `[out, in]` shapes.
    #[error("shared weight {weight} used with shapes {first:?} and {again:?}")]
    SharedWeightShapeMismatch {
        weight: u32,
        first: [usize; 2],
        again: [usize; 2],
    },

    /// An op received fewer operands than it requires.
    #[error("{op} requires at least {minimum} operand(s), got {got}")]
    InsufficientOperands {
        op: &'static str,
        minimum: usize,
        got: usize,
    },

    /// Element-wise buffer ops require equally sized operands.
    #[error("{op} operands must have equal sizes: expected {expected}, got {got}")]
    OperandSizeMismatch {
        op: &'static str,
        expected: usize,
        got: usize,
    },

    /// `gather` was given an empty index list.
    #[error("gather requires at least one index")]
    EmptyGather,

    /// A `gather` index exceeds the input buffer size.
    #[error("gather index {index} is out of bounds for input of size {input_size}")]
    GatherIndexOutOfBounds { index: usize, input_size: usize },

    /// A per-element constant's length does not match the target buffer size.
    #[error("per-element constant has length {got}, target buffer has size {expected}")]
    ConstantLengthMismatch { expected: usize, got: usize },

    /// `clip` was given neither a lower nor an upper bound.
    #[error("clip requires at least one of min/max bounds")]
    ClipWithoutBounds,

    /// Head-wise ops require the data size to be a multiple of the heads size.
    #[error("data buffer size {data_size} is not divisible by heads buffer size {heads_size}")]
    HeadsNotDivisible { data_size: usize, heads_size: usize },

    /// A map value's length does not match the map's default value length.
    #[error("map value for key {key:?} has length {got}, expected {expected}")]
    MapValueLengthMismatch {
        key: String,
        expected: usize,
        got: usize,
    },

    /// An op would produce a zero-sized buffer.
    #[error("{op} would produce a zero-sized buffer")]
    ZeroSizedBuffer { op: &'static str },

    /// A compiler invariant was violated (a bug in the graph compiler).
    #[error("graph compiler invariant violated: {message}")]
    Internal { message: String },

    /// The assembled model was rejected by [`crate::InstructionModel`].
    #[error(transparent)]
    Model(#[from] InstructionModelError),
}
