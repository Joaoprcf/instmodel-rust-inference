//! Serde types for the instruction-model JSON format.

use crate::activation::Activation;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Input rows and expected outputs checked when the model is built.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationData {
    pub inputs: Vec<Vec<f32>>,
    #[serde(rename = "expected_outputs")]
    pub expected_outputs: Vec<Vec<f32>>,
}

/// One instruction, tagged by its JSON `"type"`.
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
    #[serde(rename = "CLIP_ELEMENTWISE")]
    ClipElementwise(ClipElementwiseInstructionInfo),
    #[serde(rename = "MAP_TRANSFORM")]
    MapTransform(MapTransformInstructionInfo),
    #[serde(rename = "ADD_ELEMENTWISE_BUFFERS")]
    ElemWiseBuffersAdd(ElemWiseBuffersAddInstructionInfo),
    #[serde(rename = "MULTIPLY_ELEMENTWISE_BUFFERS")]
    ElemWiseBuffersMul(ElemWiseBuffersMulInstructionInfo),
    #[serde(rename = "MULTIPLY_BUFFER_HEADS")]
    MultiplyBufferHeads(MultiplyBufferHeadsInstructionInfo),
    #[serde(rename = "ADD_BUFFER_HEADS")]
    AddBufferHeads(AddBufferHeadsInstructionInfo),
    #[serde(rename = "REDUCE_SUM")]
    ReduceSum(ReduceSumInstructionInfo),
    #[serde(rename = "ATTENTION")]
    Attention(AttentionInstructionInfo),
}

impl InstructionInfo {
    /// Buffer indices read by this instruction.
    pub fn get_inputs(&self) -> Vec<usize> {
        match self {
            InstructionInfo::Dot(info) => vec![info.input],
            InstructionInfo::Copy(info) => vec![info.input],
            InstructionInfo::CopyMasked(info) => vec![info.input],
            InstructionInfo::Activation(info) => vec![info.input],
            InstructionInfo::ElemWiseAdd(info) => vec![info.input],
            InstructionInfo::ElemWiseMul(info) => vec![info.input],
            InstructionInfo::ClipElementwise(info) => vec![info.input],
            InstructionInfo::MapTransform(info) => vec![info.input],
            InstructionInfo::ElemWiseBuffersAdd(info) => info.input.clone(),
            InstructionInfo::ElemWiseBuffersMul(info) => info.input.clone(),
            InstructionInfo::MultiplyBufferHeads(info) => info.input.clone(),
            InstructionInfo::AddBufferHeads(info) => info.input.clone(),
            InstructionInfo::ReduceSum(info) => vec![info.input],
            InstructionInfo::Attention(info) => vec![info.input, info.key],
        }
    }

    /// Buffer index written (the input buffer for in-place instructions).
    pub fn output(&self) -> usize {
        match self {
            InstructionInfo::Dot(info) => info.output,
            InstructionInfo::Copy(info) => info.output,
            InstructionInfo::CopyMasked(info) => info.output,
            InstructionInfo::Activation(info) => info.input,
            InstructionInfo::ElemWiseAdd(info) => info.input,
            InstructionInfo::ElemWiseMul(info) => info.input,
            InstructionInfo::ClipElementwise(info) => info.input,
            InstructionInfo::MapTransform(info) => info.output,
            InstructionInfo::ElemWiseBuffersAdd(info) => info.output,
            InstructionInfo::ElemWiseBuffersMul(info) => info.output,
            InstructionInfo::MultiplyBufferHeads(info) => info.output,
            InstructionInfo::AddBufferHeads(info) => info.output,
            InstructionInfo::ReduceSum(info) => info.output,
            InstructionInfo::Attention(info) => info.output,
        }
    }

    /// Whether the instruction may write only part of its output buffer.
    pub fn supports_partial_write(&self) -> bool {
        matches!(
            self,
            InstructionInfo::Copy(_)
                | InstructionInfo::CopyMasked(_)
                | InstructionInfo::MapTransform(_)
        )
    }
}

/// Dense layer: `output = activation(W @ input + b)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DotInstructionInfo {
    pub input: usize,
    pub output: usize,
    pub weights: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activation: Option<Activation>,
}

/// Copies the whole input buffer into `output[internal_index..]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyInstructionInfo {
    pub input: usize,
    pub output: usize,
    pub internal_index: usize,
}

/// Gathers `input[indexes[i]]` into `output[i]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyMaskedInstructionInfo {
    pub input: usize,
    pub output: usize,
    pub indexes: Vec<usize>,
}

/// In-place activation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationInstructionInfo {
    pub input: usize,
    pub activation: Activation,
}

/// In-place `input += parameters[parameters]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElemWiseAddInstructionInfo {
    pub input: usize,
    pub parameters: usize,
}

/// In-place `input *= parameters[parameters]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElemWiseMulInstructionInfo {
    pub input: usize,
    pub parameters: usize,
}

/// In-place clamp by optional per-element bounds (parameter vector indices); a missing bound is unconstrained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipElementwiseInstructionInfo {
    pub input: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters_min: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters_max: Option<usize>,
}

/// Writes `maps[map][round(input[internal_input_index])]` (or `default`) to `output[internal_output_index..+size]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapTransformInstructionInfo {
    pub input: usize,
    pub output: usize,
    pub internal_input_index: usize,
    pub internal_output_index: usize,
    pub map: usize,
    pub size: usize,
    #[serde(rename = "default")]
    pub default_value: Vec<f32>,
}

/// Element-wise sum of two or more equally sized buffers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElemWiseBuffersAddInstructionInfo {
    pub input: Vec<usize>,
    pub output: usize,
}

/// Element-wise product of two or more equally sized buffers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElemWiseBuffersMulInstructionInfo {
    pub input: Vec<usize>,
    pub output: usize,
}

/// `input = [data, heads]`; multiplies each `data.len() / heads.len()` segment by its head value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiplyBufferHeadsInstructionInfo {
    pub input: Vec<usize>,
    pub output: usize,
}

/// `input = [data, heads]`; adds each head value to its `data.len() / heads.len()` segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddBufferHeadsInstructionInfo {
    pub input: Vec<usize>,
    pub output: usize,
}

/// Sum of the input buffer into `output[0]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReduceSumInstructionInfo {
    pub input: usize,
    pub output: usize,
}

/// `output = softmax(key @ W.T + b) * input`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionInstructionInfo {
    pub input: usize,
    pub key: usize,
    pub output: usize,
    pub weights: usize,
}

/// A model in the JSON format. At least one of `features`/`feature_size` is required; if both, they must agree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionModelInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_size: Option<usize>,
    #[serde(rename = "buffer_sizes")]
    pub computation_buffer_sizes: Vec<usize>,
    pub instructions: Vec<InstructionInfo>,
    /// Matrices shaped `[output_size][input_size]`; `bias[i]` pairs with `weights[i]`.
    pub weights: Vec<Vec<Vec<f32>>>,
    pub bias: Vec<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Vec<Vec<f32>>>,
    /// MAP_TRANSFORM tables keyed by the integer id as a string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maps: Option<Vec<HashMap<String, Vec<f32>>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_data: Option<ValidationData>,
}

impl InstructionModelInfo {
    pub fn builder() -> InstructionModelInfoBuilder {
        InstructionModelInfoBuilder::new()
    }

    /// Sigmoid DOT model from `{feature: weight, "constant": bias}`; `feature_order`, if given,
    /// must be a permutation of the features (default: sorted).
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
