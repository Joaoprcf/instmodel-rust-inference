//! Data structures for instruction model configuration.
//!
//! This module contains the InstructionModelInfo structure and related types
//! that define how neural network inference should be executed through a sequence
//! of instructions operating on computation buffers.

use crate::activation::Activation;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Validation data required for model evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationData {
    /// List of input data where each item is a list of feature values.
    pub inputs: Vec<Vec<f32>>,
    /// List of expected outputs corresponding to the input data.
    #[serde(rename = "expected_outputs")]
    pub expected_outputs: Vec<Vec<f32>>,
}

/// Information required to configure a model instruction.
/// This is a polymorphic type that can represent different kinds of instructions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InstructionInfo {
    #[serde(rename = "DOT")]
    Dot(DotInstructionInfo),
    #[serde(rename = "COPY")]
    Copy(CopyInstructionInfo),
    #[serde(rename = "COPY_MASKED")]
    CopyMasked(CopyMaskedInstructionInfo),
    #[serde(rename = "ACTIVATION")]
    Activation(ActivationInstructionInfo),
    #[serde(rename = "ADD_ELEMENTWISE")]
    ElemWiseAdd(ElemWiseAddInstructionInfo),
    #[serde(rename = "MUL_ELEMENTWISE")]
    ElemWiseMul(ElemWiseMulInstructionInfo),
    #[serde(rename = "MAP_TRANSFORM")]
    MapTransform(MapTransformInstructionInfo),
    #[serde(rename = "ADD_ELEMENTWISE_BUFFERS")]
    ElemWiseBuffersAdd(ElemWiseBuffersAddInstructionInfo),
    #[serde(rename = "MULTIPLY_ELEMENTWISE_BUFFERS")]
    ElemWiseBuffersMul(ElemWiseBuffersMulInstructionInfo),
    #[serde(rename = "REDUCE_SUM")]
    ReduceSum(ReduceSumInstructionInfo),
    #[serde(rename = "ATTENTION")]
    Attention(AttentionInstructionInfo),
}

impl InstructionInfo {
    /// Returns the list of input buffer indices required for this instruction.
    pub fn get_inputs(&self) -> Vec<usize> {
        match self {
            InstructionInfo::Dot(info) => vec![info.input],
            InstructionInfo::Copy(info) => vec![info.input],
            InstructionInfo::CopyMasked(info) => vec![info.input],
            InstructionInfo::Activation(info) => vec![info.input],
            InstructionInfo::ElemWiseAdd(info) => vec![info.input],
            InstructionInfo::ElemWiseMul(info) => vec![info.input],
            InstructionInfo::MapTransform(info) => vec![info.input],
            InstructionInfo::ElemWiseBuffersAdd(info) => info.input.clone(),
            InstructionInfo::ElemWiseBuffersMul(info) => info.input.clone(),
            InstructionInfo::ReduceSum(info) => vec![info.input],
            InstructionInfo::Attention(info) => vec![info.input, info.key],
        }
    }

    /// Returns the output buffer index for this instruction.
    /// For instructions operating in place, the output index defaults to the input index.
    pub fn output(&self) -> usize {
        match self {
            InstructionInfo::Dot(info) => info.output,
            InstructionInfo::Copy(info) => info.output,
            InstructionInfo::CopyMasked(info) => info.output,
            InstructionInfo::Activation(info) => info.input, // In-place operation
            InstructionInfo::ElemWiseAdd(info) => info.input, // In-place operation
            InstructionInfo::ElemWiseMul(info) => info.input, // In-place operation
            InstructionInfo::MapTransform(info) => info.output,
            InstructionInfo::ElemWiseBuffersAdd(info) => info.output,
            InstructionInfo::ElemWiseBuffersMul(info) => info.output,
            InstructionInfo::ReduceSum(info) => info.output,
            InstructionInfo::Attention(info) => info.output,
        }
    }
}

/// Represents a dot product operation instruction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DotInstructionInfo {
    /// Input index of the target buffer.
    pub input: usize,
    /// Output index of the target buffer.
    pub output: usize,
    /// Weights index targeting the weights of a given layer.
    pub weights: usize,
    /// Activation function (may be null).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activation: Option<Activation>,
}

/// Represents a sliced data copy operation instruction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyInstructionInfo {
    /// Input index of the target buffer.
    pub input: usize,
    /// Output index of the target buffer.
    pub output: usize,
    /// Start index of the target output layer.
    pub internal_index: usize,
}

/// Represents an instruction to copy data from specific indexed positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyMaskedInstructionInfo {
    /// Input index of the target buffer.
    pub input: usize,
    /// Output index of the target buffer.
    pub output: usize,
    /// List of indexes to copy.
    pub indexes: Vec<usize>,
}

/// Represents a single activation function operation instruction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationInstructionInfo {
    /// Input index of the target buffer. Same as the output because it operates in place.
    pub input: usize,
    /// Activation function (should not be null).
    pub activation: Activation,
}

/// Represents an element-wise addition operation instruction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElemWiseAddInstructionInfo {
    /// Input index of the target buffer.
    pub input: usize,
    /// Number of parameters for the element-wise operation.
    pub parameters: usize,
}

/// Represents an element-wise multiplication operation instruction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElemWiseMulInstructionInfo {
    /// Input index of the target buffer.
    pub input: usize,
    /// Number of parameters for the element-wise operation.
    pub parameters: usize,
}

/// Represents a map transform operation instruction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapTransformInstructionInfo {
    /// Input index of the target buffer.
    pub input: usize,
    /// Output index of the target buffer.
    pub output: usize,
    /// Target internal index for the input buffer.
    pub internal_input_index: usize,
    /// Target internal index for the output buffer.
    pub internal_output_index: usize,
    /// Map index for the instruction.
    pub map: usize,
    /// Size parameter for the instruction.
    pub size: usize,
    /// Default value list when the map key is not found.
    #[serde(rename = "default")]
    pub default_value: Vec<f32>,
}

/// Represents an element-wise addition operation that sums multiple input DataBuffers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElemWiseBuffersAddInstructionInfo {
    /// List of input buffer indices.
    pub input: Vec<usize>,
    /// Output buffer index.
    pub output: usize,
}

/// Represents an element-wise multiplication operation that multiplies multiple input DataBuffers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElemWiseBuffersMulInstructionInfo {
    /// List of input buffer indices.
    pub input: Vec<usize>,
    /// Output buffer index.
    pub output: usize,
}

/// Represents a reduce sum operation that sums all values in an input buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReduceSumInstructionInfo {
    /// Input buffer index.
    pub input: usize,
    /// Output buffer index.
    pub output: usize,
}

/// Represents an attention operation instruction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionInstructionInfo {
    /// Input buffer index (query/value buffer).
    pub input: usize,
    /// Key buffer index.
    pub key: usize,
    /// Output buffer index.
    pub output: usize,
    /// Weights index targeting the weights of a given layer.
    pub weights: usize,
}

/// Instruction model information required to build the computation graph.
/// At least one of [features, feature_size] must be defined. When both are provided, their size must match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionModelInfo {
    /// List of input features.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<String>>,
    /// Size of the feature vector.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_size: Option<usize>,
    /// List of the computation buffer sizes.
    #[serde(rename = "buffer_sizes")]
    pub computation_buffer_sizes: Vec<usize>,
    /// List of instructions to compute the model output.
    pub instructions: Vec<InstructionInfo>,
    /// Weights as a list of matrices (each with shape: [output_size, input_size]).
    pub weights: Vec<Vec<Vec<f32>>>,
    /// Biases as a list of vectors (each with shape: [output_size]).
    pub bias: Vec<Vec<f32>>,
    /// Parameters as a list of vectors (each with shape: [output_size]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Vec<Vec<f32>>>,
    /// Mapping structures as a list of dictionaries (key: id, value: list of floats).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maps: Option<Vec<HashMap<String, Vec<f32>>>>,
    /// Data used to validate the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_data: Option<ValidationData>,
}

impl InstructionModelInfo {
    /// Creates a new builder for InstructionModelInfo.
    pub fn builder() -> InstructionModelInfoBuilder {
        InstructionModelInfoBuilder::new()
    }

    /// Creates instruction model information from the given logistic regression model data.
    ///
    /// The keys in the decision_function map (except "constant") represent feature weights,
    /// and the "constant" key represents the bias term.
    pub fn from_logistic_regression_model(
        decision_function: HashMap<String, f64>,
        feature_order: Option<Vec<String>>,
    ) -> Result<Self, crate::errors::InstructionModelError> {
        let mut sorted_input_features: Vec<String> = decision_function
            .keys()
            .filter(|k| *k != "constant")
            .cloned()
            .collect();
        sorted_input_features.sort();

        let bias = decision_function.get("constant").copied().unwrap_or(0.0) as f32;

        let model_feature_order = if let Some(order) = feature_order {
            let mut sorted_feature_order = order.clone();
            sorted_feature_order.sort();
            if sorted_feature_order != sorted_input_features {
                return Err(crate::errors::InstructionModelError::InvalidFeatureFormat {
                    feature: format!(
                        "Provided features do not match the expected features from the decision function. Expected: {:?}, but received: {:?}",
                        sorted_input_features, sorted_feature_order
                    ),
                });
            }
            order
        } else {
            sorted_input_features
        };

        let weights_row: Vec<f32> = model_feature_order
            .iter()
            .map(|feature| decision_function.get(feature).copied().unwrap_or(0.0) as f32)
            .collect();

        Ok(InstructionModelInfo {
            features: Some(model_feature_order.clone()),
            feature_size: Some(model_feature_order.len()),
            computation_buffer_sizes: vec![model_feature_order.len(), 1],
            instructions: vec![InstructionInfo::Dot(DotInstructionInfo {
                input: 0,
                output: 1,
                weights: 0,
                activation: Some(Activation::Sigmoid),
            })],
            weights: vec![vec![weights_row]],
            bias: vec![vec![bias]],
            parameters: None,
            maps: None,
            validation_data: None,
        })
    }
}

/// Builder for InstructionModelInfo.
pub struct InstructionModelInfoBuilder {
    features: Option<Vec<String>>,
    feature_size: Option<usize>,
    computation_buffer_sizes: Vec<usize>,
    instructions: Vec<InstructionInfo>,
    weights: Vec<Vec<Vec<f32>>>,
    bias: Vec<Vec<f32>>,
    parameters: Option<Vec<Vec<f32>>>,
    maps: Option<Vec<HashMap<String, Vec<f32>>>>,
    validation_data: Option<ValidationData>,
}

impl InstructionModelInfoBuilder {
    fn new() -> Self {
        Self {
            features: None,
            feature_size: None,
            computation_buffer_sizes: Vec::new(),
            instructions: Vec::new(),
            weights: Vec::new(),
            bias: Vec::new(),
            parameters: None,
            maps: None,
            validation_data: None,
        }
    }

    pub fn features(mut self, value: Vec<String>) -> Self {
        self.features = Some(value);
        self
    }

    pub fn feature_size(mut self, value: usize) -> Self {
        self.feature_size = Some(value);
        self
    }

    pub fn computation_buffer_sizes(mut self, value: Vec<usize>) -> Self {
        self.computation_buffer_sizes = value;
        self
    }

    pub fn instructions(mut self, value: Vec<InstructionInfo>) -> Self {
        self.instructions = value;
        self
    }

    pub fn weights(mut self, value: Vec<Vec<Vec<f32>>>) -> Self {
        self.weights = value;
        self
    }

    pub fn bias(mut self, value: Vec<Vec<f32>>) -> Self {
        self.bias = value;
        self
    }

    pub fn parameters(mut self, value: Vec<Vec<f32>>) -> Self {
        self.parameters = Some(value);
        self
    }

    pub fn maps(mut self, value: Vec<HashMap<String, Vec<f32>>>) -> Self {
        self.maps = Some(value);
        self
    }

    pub fn validation_data(mut self, value: ValidationData) -> Self {
        self.validation_data = Some(value);
        self
    }

    pub fn build(self) -> Result<InstructionModelInfo, crate::errors::InstructionModelError> {
        if self.features.is_none() && self.feature_size.is_none() {
            return Err(crate::errors::InstructionModelError::MissingFeatures);
        }

        Ok(InstructionModelInfo {
            features: self.features,
            feature_size: self.feature_size,
            computation_buffer_sizes: self.computation_buffer_sizes,
            instructions: self.instructions,
            weights: self.weights,
            bias: self.bias,
            parameters: self.parameters,
            maps: self.maps,
            validation_data: self.validation_data,
        })
    }
}
