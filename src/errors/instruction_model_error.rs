//! Error types for instruction model operations.

use thiserror::Error;

/// Errors that can occur during instruction model creation, validation, or execution.
#[derive(Error, Debug)]
pub enum InstructionModelError {
    #[error("Invalid feature format: {feature}")]
    InvalidFeatureFormat { feature: String },

    #[error("Feature size mismatch: expected {expected} but got {actual} from features")]
    FeatureSizeMismatch { expected: usize, actual: usize },

    #[error(
        "Invalid feature size: expected cumulative capacity {expected} but got {actual}: {capacities:?}"
    )]
    InvalidFeatureSize {
        expected: usize,
        actual: usize,
        capacities: Vec<usize>,
    },

    #[error("Features or feature size must be provided")]
    MissingFeatures,

    #[error("At least one layer is required")]
    NoLayersProvided,

    #[error("At least one instruction is required")]
    NoInstructionsProvided,

    #[error("The numbers of bias and weights must be the same")]
    BiasWeightsMismatch,

    #[error("The number of weights/bias must not exceed the number instructions")]
    TooManyWeightsForInstructions,

    #[error("The size of the weights exceeds the maximum allowed size: {actual} > {max}")]
    WeightSizeExceedsLimit { actual: usize, max: usize },

    #[error(
        "The computation buffer array size exceeds the maximum allowed elements: {actual} > {max}"
    )]
    ComputationBufferSizeExceedsLimit { actual: usize, max: usize },

    #[error("Feature size cannot be negative: {size}")]
    NegativeFeatureSize { size: isize },

    #[error("The size of the layer must be greater than 0")]
    InvalidLayerSize,

    #[error("The size of the unified computation buffer must be greater than 0")]
    InvalidUnifiedBufferSize,

    #[error("The size of the input layer must be greater than or equal to 0")]
    InvalidInputLayerSize,

    #[error("Computation buffer cannot be null")]
    NullComputationBuffer,

    #[error(
        "Computation buffer is not large enough to compute the result: {buffer_size} < {required_size}"
    )]
    ComputationBufferTooSmall {
        buffer_size: usize,
        required_size: usize,
    },

    #[error(
        "The feature size {feature_size} must be less than or equal to the computation buffer size {buffer_size}"
    )]
    FeatureSizeExceedsBufferSize {
        feature_size: usize,
        buffer_size: usize,
    },

    #[error("The {label} {index} must be within the layer sizes")]
    BufferIndexOutOfBounds { label: String, index: usize },

    #[error("The weights {index} must be within the number of weights")]
    WeightsIndexOutOfBounds { index: usize },

    #[error(
        "The number of rows of the weights must match the output layer size: {weights_rows} != {output_size}"
    )]
    WeightsRowSizeMismatch {
        weights_rows: usize,
        output_size: usize,
    },

    #[error(
        "The size of the bias in index {bias_index} must match the output layer size at index {output_index}: {bias_size} != {output_size}"
    )]
    BiasOutputSizeMismatch {
        bias_index: usize,
        output_index: usize,
        bias_size: usize,
        output_size: usize,
    },

    #[error(
        "The size of the columns of weights must match the input layer size at index {input_index}: {weights_columns} != {input_size}"
    )]
    WeightsColumnSizeMismatch {
        input_index: usize,
        weights_columns: usize,
        input_size: usize,
    },

    #[error("The input and output indexes must be different on a {instruction_type} instruction")]
    SameInputOutputIndexes { instruction_type: String },

    #[error("The internal index must be greater than or equal to 0")]
    NegativeInternalIndex,

    #[error(
        "The internal index {internal_index} plus the data size {data_size} must be less than or equal to the output size {output_size}"
    )]
    InternalIndexOutOfBounds {
        internal_index: usize,
        data_size: usize,
        output_size: usize,
    },

    #[error("The internal indexes must not be empty")]
    EmptyInternalIndexes,

    #[error(
        "The number of internal indexes must be less than or equal to the output size: {indexes_size} > {output_size}"
    )]
    TooManyInternalIndexes {
        indexes_size: usize,
        output_size: usize,
    },

    #[error(
        "All internal indexes must be within the input size {input_size}, {invalid_index} is not"
    )]
    InternalIndexOutOfInputBounds {
        input_size: usize,
        invalid_index: usize,
    },

    #[error("The internal indexes on a masked copy instruction must be unique")]
    DuplicateInternalIndexes,

    #[error("The parameters index {index} must be within the number of parameters")]
    ParametersIndexOutOfBounds { index: usize },

    #[error(
        "The size of the parameters must match the target layer size at index {index}: {params_size} != {layer_size}"
    )]
    ParametersSizeMismatch {
        index: usize,
        params_size: usize,
        layer_size: usize,
    },

    #[error("The number of input buffers for elementwise operations must be at least 2")]
    InsufficientInputBuffers,

    #[error(
        "Input buffer at index {index} has size {actual_size}, but the expected size is {expected_size}"
    )]
    InputBufferSizeMismatch {
        index: usize,
        actual_size: usize,
        expected_size: usize,
    },

    #[error("The map index {index} must be within the number of maps")]
    MapIndexOutOfBounds { index: usize },

    #[error("The internal input index {index} must be within the input size")]
    InternalInputIndexOutOfBounds { index: usize },

    #[error(
        "The internal output index {internal_output_index} plus the data size {data_size} must be less or equal than the output size {output_size}"
    )]
    InternalOutputIndexOutOfBounds {
        internal_output_index: usize,
        data_size: usize,
        output_size: usize,
    },

    #[error(
        "The size of the default values must match the data size: {default_size} != {data_size}"
    )]
    DefaultValuesSizeMismatch {
        default_size: usize,
        data_size: usize,
    },

    #[error("The size of the map values must match the data size: {map_value_size} != {data_size}")]
    MapValuesSizeMismatch {
        map_value_size: usize,
        data_size: usize,
    },

    #[error("The weights at index {index} are not used")]
    UnusedWeights { index: usize },

    #[error("The parameters at index {index} are not used")]
    UnusedParameters { index: usize },

    #[error("The map at index {index} is not used")]
    UnusedMap { index: usize },

    #[error("The instruction is not supported: {instruction}")]
    UnsupportedInstruction { instruction: String },

    #[error("Unexpected instruction type: {instruction_type}")]
    UnexpectedInstructionType { instruction_type: String },

    #[error("Null pointer exception: {message}")]
    NullPointer { message: String },

    #[error(
        "The number of rows in weights does not match bias size in index {index}: {bias_size} != {weights_size}"
    )]
    BiasWeightsSizeMismatch {
        index: usize,
        bias_size: usize,
        weights_size: usize,
    },

    #[error("The number of inputs must match the number of outputs in the validation data")]
    ValidationInputOutputMismatch,

    #[error(
        "The number of inputs provided must match the feature size of the model: {provided} != {expected}"
    )]
    ValidationInputSizeMismatch { provided: usize, expected: usize },

    #[error(
        "The size of the outputs must match the output size at index {index}: {provided} != {expected}"
    )]
    ValidationOutputSizeMismatch {
        index: usize,
        provided: usize,
        expected: usize,
    },

    #[error(
        "In validation case number {case_number}, applying inference on the following inputs: {inputs:?}, the expected outputs were: {expected:?}, the computed outputs were: {computed:?}"
    )]
    ValidationMismatch {
        case_number: usize,
        inputs: Vec<f32>,
        expected: Vec<f32>,
        computed: Vec<f32>,
    },

    #[error("The number of inputs must match the number of outputs")]
    InputOutputCountMismatch,
}
