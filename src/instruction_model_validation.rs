//! Construction-time validation for [`crate::InstructionModel`] and the GPU serializer.

use std::collections::HashMap;

use crate::errors::{
    BufferIndexOutOfBoundsError, ComputationBufferSizeExceedsLimitError, FeatureSizeMismatchError,
    InstructionModelError, InstructionModelResult, InvalidFeatureSizeError,
    ValidationInputOutputMismatchError,
};
use crate::instruction_model_info::{InstructionInfo, InstructionModelInfo, ValidationData};

pub(crate) const MAX_COMPUTATION_BUFFER_SIZE: usize = 1_000_000;
pub(crate) const MAX_WEIGHT_SIZE: usize = 10_000_000;

/// Total feature count from `features` (`name` = 1, `name[n]` = n) or `feature_size`.
pub(crate) fn calculate_feature_size(
    instruction_model_info: &InstructionModelInfo,
) -> InstructionModelResult<usize> {
    let Some(features) = &instruction_model_info.features else {
        return instruction_model_info
            .feature_size
            .ok_or(InstructionModelError::MissingFeatures);
    };
    let mut total_size = 0usize;
    for feature in features {
        let invalid = || InstructionModelError::InvalidFeatureFormat {
            feature: feature.clone(),
        };
        let count = match feature.find('[') {
            None if feature.contains(']') => return Err(invalid()),
            None => 1,
            Some(open) => {
                let number = feature[open + 1..].strip_suffix(']').ok_or_else(invalid)?;
                if open == 0 || feature[..open].contains(']') || number.contains(['[', ']']) {
                    return Err(invalid());
                }
                match number.parse::<usize>() {
                    Ok(n) if n > 0 => n,
                    _ => return Err(invalid()),
                }
            }
        };
        total_size = total_size.checked_add(count).ok_or_else(invalid)?;
    }
    validate_declared_feature_size(instruction_model_info.feature_size, total_size)?;
    Ok(total_size)
}

/// Validates that the feature size exactly fills one or more leading buffers.
pub(crate) fn validate_feature_size(
    feature_size: usize,
    buffer_sizes: &[usize],
) -> std::result::Result<(), InvalidFeatureSizeError> {
    let mut accumulated = 0usize;
    let mut accumulated_capacities = Vec::new();

    for &capacity in buffer_sizes {
        accumulated = accumulated.saturating_add(capacity);
        accumulated_capacities.push(accumulated);
        match accumulated.cmp(&feature_size) {
            std::cmp::Ordering::Equal => return Ok(()),
            std::cmp::Ordering::Greater => break,
            std::cmp::Ordering::Less => {}
        }
    }

    Err(InvalidFeatureSizeError {
        expected: feature_size,
        actual: accumulated,
        capacities: accumulated_capacities,
    })
}

/// Model-level checks that do not depend on individual instructions; returns the feature size.
pub(crate) fn validate_inputs(
    instruction_model_info: &InstructionModelInfo,
) -> InstructionModelResult<usize> {
    let info = instruction_model_info;
    if info.features.is_none() && info.feature_size.is_none() {
        return Err(InstructionModelError::MissingFeatures);
    }
    if info.computation_buffer_sizes.is_empty() {
        return Err(InstructionModelError::NoLayersProvided);
    }
    if info.bias.len() != info.weights.len() {
        return Err(InstructionModelError::BiasWeightsMismatch);
    }
    if info.instructions.is_empty() {
        return Err(InstructionModelError::NoInstructionsProvided);
    }
    if info.bias.len() > info.instructions.len() {
        return Err(InstructionModelError::TooManyWeightsForInstructions);
    }

    let mut calculated_size = 0;
    for (i, (bias_vec, weights)) in info.bias.iter().zip(&info.weights).enumerate() {
        if bias_vec.len() != weights.len() {
            return Err(InstructionModelError::BiasWeightsSizeMismatch {
                index: i,
                bias_size: bias_vec.len(),
                weights_size: weights.len(),
            });
        }
        calculated_size += bias_vec.len() + weights.iter().map(Vec::len).sum::<usize>();
    }
    if calculated_size > MAX_WEIGHT_SIZE {
        return Err(InstructionModelError::WeightSizeExceedsLimit {
            actual: calculated_size,
            max: MAX_WEIGHT_SIZE,
        });
    }

    let feature_size = calculate_feature_size(info)?;
    validate_feature_size(feature_size, &info.computation_buffer_sizes)?;
    validate_validation_data_lengths(info.validation_data.as_ref())?;
    Ok(feature_size)
}

pub(crate) fn validate_required_memory(
    output_index_end: usize,
) -> std::result::Result<(), ComputationBufferSizeExceedsLimitError> {
    if output_index_end > MAX_COMPUTATION_BUFFER_SIZE {
        return Err(ComputationBufferSizeExceedsLimitError {
            actual: output_index_end,
            max: MAX_COMPUTATION_BUFFER_SIZE,
        });
    }
    Ok(())
}

pub(crate) fn validate_declared_feature_size(
    declared: Option<usize>,
    computed: usize,
) -> std::result::Result<(), FeatureSizeMismatchError> {
    if let Some(expected) = declared.filter(|&fs| fs != computed) {
        return Err(FeatureSizeMismatchError {
            expected,
            actual: computed,
        });
    }
    Ok(())
}

pub(crate) fn validate_validation_data_lengths(
    validation_data: Option<&ValidationData>,
) -> std::result::Result<(), ValidationInputOutputMismatchError> {
    if validation_data.is_some_and(|vd| vd.inputs.len() != vd.expected_outputs.len()) {
        return Err(ValidationInputOutputMismatchError);
    }
    Ok(())
}

/// Fills `computation_buffer_indexes` with each buffer's start offset and returns the
/// total size. Saturates on overflow so the caller's size limit rejects it.
pub(crate) fn calculate_computation_buffer_indexes(
    computation_buffer_sizes: &[usize],
    computation_buffer_indexes: &mut Vec<usize>,
) -> InstructionModelResult<usize> {
    computation_buffer_indexes.push(0);
    let mut index = computation_buffer_sizes[0];

    for &computation_buffer_size in computation_buffer_sizes.iter().skip(1) {
        if computation_buffer_size == 0 {
            return Err(InstructionModelError::InvalidLayerSize);
        }
        computation_buffer_indexes.push(index);
        index = index.saturating_add(computation_buffer_size);
    }

    if index == 0 {
        return Err(InstructionModelError::InvalidUnifiedBufferSize);
    }

    Ok(index)
}

pub(crate) fn validate_buffer_index(
    label: &str,
    buffer_index: usize,
    max_size: usize,
) -> std::result::Result<(), BufferIndexOutOfBoundsError> {
    if buffer_index >= max_size {
        return Err(BufferIndexOutOfBoundsError {
            label: label.to_string(),
            index: buffer_index,
        });
    }
    Ok(())
}

/// `offset + len <= size` without overflow.
fn fits(offset: usize, len: usize, size: usize) -> bool {
    offset.checked_add(len).is_some_and(|end| end <= size)
}

/// Checks every buffer, weight, parameter and map reference of one instruction, so that
/// execution stays inside the buffers it names (the CPU and GPU executors rely on this).
pub(crate) fn validate_instruction(
    instruction: &InstructionInfo,
    sizes: &[usize],
    weights: &[Vec<Vec<f32>>],
    bias: &[Vec<f32>],
    parameters: &[Vec<f32>],
    maps: &[HashMap<String, Vec<f32>>],
) -> InstructionModelResult<()> {
    for input in instruction.get_inputs() {
        validate_buffer_index("input", input, sizes.len())?;
    }
    validate_buffer_index("output", instruction.output(), sizes.len())?;

    let check_parameters = |index: usize, layer: usize| {
        let params = parameters
            .get(index)
            .ok_or(InstructionModelError::ParametersIndexOutOfBounds { index })?;
        if params.len() != sizes[layer] {
            return Err(InstructionModelError::ParametersSizeMismatch {
                index: layer,
                params_size: params.len(),
                layer_size: sizes[layer],
            });
        }
        Ok(())
    };

    match instruction {
        InstructionInfo::Dot(info) => {
            if info.input == info.output {
                return Err(InstructionModelError::SameInputOutputIndexes {
                    instruction_type: "DOT".to_string(),
                });
            }
            validate_dense(info.weights, info.input, info.output, sizes, weights, bias)
        }
        InstructionInfo::Attention(info) => {
            let (query_size, output_size) = (sizes[info.input], sizes[info.output]);
            if query_size != output_size {
                return Err(InstructionModelError::InputBufferSizeMismatch {
                    index: info.input,
                    actual_size: query_size,
                    expected_size: output_size,
                });
            }
            validate_dense(info.weights, info.key, info.output, sizes, weights, bias)
        }
        InstructionInfo::Copy(info) => {
            let (data_size, output_size) = (sizes[info.input], sizes[info.output]);
            if !fits(info.internal_index, data_size, output_size) {
                return Err(InstructionModelError::InternalIndexOutOfBounds {
                    internal_index: info.internal_index,
                    data_size,
                    output_size,
                });
            }
            Ok(())
        }
        InstructionInfo::CopyMasked(info) => {
            let (input_size, output_size) = (sizes[info.input], sizes[info.output]);
            if info.indexes.len() > output_size {
                return Err(InstructionModelError::TooManyInternalIndexes {
                    indexes_size: info.indexes.len(),
                    output_size,
                });
            }
            if let Some(&invalid_index) = info.indexes.iter().find(|&&i| i >= input_size) {
                return Err(InstructionModelError::InternalIndexOutOfInputBounds {
                    input_size,
                    invalid_index,
                });
            }
            Ok(())
        }
        InstructionInfo::ElemWiseAdd(info) => check_parameters(info.parameters, info.input),
        InstructionInfo::ElemWiseMul(info) => check_parameters(info.parameters, info.input),
        InstructionInfo::ClipElementwise(info) => [info.parameters_min, info.parameters_max]
            .into_iter()
            .flatten()
            .try_for_each(|index| check_parameters(index, info.input)),
        InstructionInfo::MapTransform(info) => {
            if info.map >= maps.len() {
                return Err(InstructionModelError::MapIndexOutOfBounds { index: info.map });
            }
            if info.internal_input_index >= sizes[info.input] {
                return Err(InstructionModelError::InternalInputIndexOutOfBounds {
                    index: info.internal_input_index,
                });
            }
            if !fits(info.internal_output_index, info.size, sizes[info.output]) {
                return Err(InstructionModelError::InternalOutputIndexOutOfBounds {
                    internal_output_index: info.internal_output_index,
                    data_size: info.size,
                    output_size: sizes[info.output],
                });
            }
            Ok(())
        }
        InstructionInfo::ElemWiseBuffersAdd(info) => {
            validate_same_sizes(&info.input, info.output, sizes)
        }
        InstructionInfo::ElemWiseBuffersMul(info) => {
            validate_same_sizes(&info.input, info.output, sizes)
        }
        InstructionInfo::MultiplyBufferHeads(info) => {
            validate_heads(&info.input, info.output, sizes)
        }
        InstructionInfo::AddBufferHeads(info) => validate_heads(&info.input, info.output, sizes),
        InstructionInfo::Activation(_) | InstructionInfo::ReduceSum(_) => Ok(()),
    }
}

fn validate_dense(
    slot: usize,
    input: usize,
    output: usize,
    sizes: &[usize],
    weights: &[Vec<Vec<f32>>],
    bias: &[Vec<f32>],
) -> InstructionModelResult<()> {
    let (Some(matrix), Some(bias_vector)) = (weights.get(slot), bias.get(slot)) else {
        return Err(InstructionModelError::WeightsIndexOutOfBounds { index: slot });
    };
    let (input_size, output_size) = (sizes[input], sizes[output]);
    if matrix.len() != output_size {
        return Err(InstructionModelError::WeightsRowSizeMismatch {
            weights_rows: matrix.len(),
            output_size,
        });
    }
    if bias_vector.len() != output_size {
        return Err(InstructionModelError::BiasOutputSizeMismatch {
            bias_index: slot,
            output_index: output,
            bias_size: bias_vector.len(),
            output_size,
        });
    }
    if let Some(row) = matrix.iter().find(|row| row.len() != input_size) {
        return Err(InstructionModelError::WeightsColumnSizeMismatch {
            input_index: input,
            weights_columns: row.len(),
            input_size,
        });
    }
    Ok(())
}

fn validate_same_sizes(
    inputs: &[usize],
    output: usize,
    sizes: &[usize],
) -> InstructionModelResult<()> {
    if inputs.len() < 2 {
        return Err(InstructionModelError::InsufficientInputBuffers);
    }
    match inputs.iter().find(|&&i| sizes[i] != sizes[output]) {
        Some(&index) => Err(InstructionModelError::InputBufferSizeMismatch {
            index,
            actual_size: sizes[index],
            expected_size: sizes[output],
        }),
        None => Ok(()),
    }
}

fn validate_heads(inputs: &[usize], output: usize, sizes: &[usize]) -> InstructionModelResult<()> {
    let &[data, heads] = inputs else {
        return Err(InstructionModelError::InsufficientInputBuffers);
    };
    let (data_size, heads_size) = (sizes[data], sizes[heads]);
    if heads_size == 0 || !data_size.is_multiple_of(heads_size) {
        return Err(InstructionModelError::InvalidBufferHeadsSize {
            data_size,
            heads_size,
        });
    }
    if sizes[output] != data_size {
        return Err(InstructionModelError::InputBufferSizeMismatch {
            index: data,
            actual_size: data_size,
            expected_size: sizes[output],
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InstructionModel;
    use serde_json::{Value, json};

    fn info(v: Value) -> InstructionModelInfo {
        serde_json::from_value(v).unwrap()
    }

    fn features(names: &[&str], feature_size: Option<usize>) -> InstructionModelInfo {
        let mut v = json!({"features": names, "buffer_sizes": [1], "instructions": [],
                           "weights": [], "bias": []});
        if let Some(fs) = feature_size {
            v["feature_size"] = json!(fs);
        }
        info(v)
    }

    #[test]
    fn feature_size_parsing() {
        assert_eq!(
            calculate_feature_size(&features(&["a", "b[3]", "c"], None)).unwrap(),
            5
        );
        assert_eq!(
            calculate_feature_size(&info(json!({"feature_size": 10, "buffer_sizes": [10],
            "instructions": [], "weights": [], "bias": []})))
            .unwrap(),
            10
        );
        let max = format!("a[{}]", usize::MAX);
        for bad in [
            "f[]",
            "f[[5]]",
            "f[5",
            "f[abc]",
            "f[0]",
            "[3]",
            "a]b[3]",
            "f]",
            max.as_str(),
        ] {
            let names = [bad, "x"];
            assert!(
                matches!(
                    calculate_feature_size(&features(&names, None)),
                    Err(InstructionModelError::InvalidFeatureFormat { .. })
                ),
                "{bad}"
            );
        }
        assert!(matches!(
            calculate_feature_size(&features(&["a", "b"], Some(5))),
            Err(InstructionModelError::FeatureSizeMismatch {
                expected: 5,
                actual: 2
            })
        ));
        assert!(matches!(
            validate_feature_size(5, &[2, 2]),
            Err(InvalidFeatureSizeError {
                expected: 5,
                actual: 4,
                ..
            })
        ));
    }

    #[test]
    fn buffer_layout_errors() {
        let mut idx = Vec::new();
        assert!(matches!(
            calculate_computation_buffer_indexes(&[1, 0], &mut idx),
            Err(InstructionModelError::InvalidLayerSize)
        ));
        assert!(matches!(
            calculate_computation_buffer_indexes(&[0], &mut Vec::new()),
            Err(InstructionModelError::InvalidUnifiedBufferSize)
        ));
        assert_eq!(
            calculate_computation_buffer_indexes(&[1, usize::MAX, 2], &mut Vec::new()).unwrap(),
            usize::MAX
        );
        assert!(matches!(
            validate_buffer_index("test", 5, 3),
            Err(BufferIndexOutOfBoundsError { label, index: 5 }) if label == "test"
        ));
    }

    fn model(extra: Value) -> InstructionModelResult<InstructionModel> {
        let mut v =
            json!({"feature_size": 4, "buffer_sizes": [4, 2, 4], "weights": [], "bias": []});
        v.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        InstructionModel::new(info(v))
    }

    #[test]
    fn model_level_errors() {
        use InstructionModelError as E;
        let copy = json!([{"type": "COPY", "input": 0, "output": 2, "internal_index": 0}]);
        let dot = json!([{"type": "DOT", "input": 0, "output": 1, "weights": 0}]);
        let cases: Vec<(Value, fn(&E) -> bool)> = vec![
            (json!({"buffer_sizes": [], "instructions": copy}), |e| {
                matches!(e, E::NoLayersProvided)
            }),
            (json!({"instructions": []}), |e| {
                matches!(e, E::NoInstructionsProvided)
            }),
            (json!({"instructions": copy, "weights": [[[1.0]]]}), |e| {
                matches!(e, E::BiasWeightsMismatch)
            }),
            (
                json!({"instructions": copy, "weights": [[[1.0]], [[1.0]]], "bias": [[1.0], [1.0]]}),
                |e| matches!(e, E::TooManyWeightsForInstructions),
            ),
            (
                json!({"instructions": dot, "weights": [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]], "bias": [[1.0]]}),
                |e| matches!(e, E::BiasWeightsSizeMismatch { index: 0, .. }),
            ),
            (
                json!({"instructions": copy, "validation_data": {"inputs": [[1.0, 1.0, 1.0, 1.0]], "expected_outputs": []}}),
                |e| matches!(e, E::ValidationInputOutputMismatch),
            ),
            (
                json!({"buffer_sizes": [4, usize::MAX, 4], "instructions": copy}),
                |e| matches!(e, E::ComputationBufferSizeExceedsLimit { .. }),
            ),
        ];
        for (extra, check) in cases {
            let err = model(extra.clone())
                .err()
                .unwrap_or_else(|| panic!("{extra} accepted"));
            assert!(check(&err), "{extra}: {err}");
        }
    }

    #[test]
    fn instruction_reference_errors() {
        use InstructionModelError as E;
        let cases: Vec<(Value, fn(&E) -> bool)> = vec![
            (
                json!({"type": "COPY", "input": 1, "output": 2, "internal_index": 3}),
                |e| {
                    matches!(
                        e,
                        E::InternalIndexOutOfBounds {
                            internal_index: 3,
                            data_size: 2,
                            output_size: 4
                        }
                    )
                },
            ),
            (
                json!({"type": "COPY", "input": 1, "output": 2, "internal_index": usize::MAX}),
                |e| matches!(e, E::InternalIndexOutOfBounds { .. }),
            ),
            (
                json!({"type": "COPY_MASKED", "input": 0, "output": 1, "indexes": [0, 1, 2]}),
                |e| {
                    matches!(
                        e,
                        E::TooManyInternalIndexes {
                            indexes_size: 3,
                            output_size: 2
                        }
                    )
                },
            ),
            (
                json!({"type": "COPY_MASKED", "input": 1, "output": 2, "indexes": [2]}),
                |e| {
                    matches!(
                        e,
                        E::InternalIndexOutOfInputBounds {
                            input_size: 2,
                            invalid_index: 2
                        }
                    )
                },
            ),
            (
                json!({"type": "ADD_ELEMENTWISE", "input": 1, "parameters": 1}),
                |e| matches!(e, E::ParametersIndexOutOfBounds { index: 1 }),
            ),
            (
                json!({"type": "MUL_ELEMENTWISE", "input": 0, "parameters": 0}),
                |e| {
                    matches!(
                        e,
                        E::ParametersSizeMismatch {
                            index: 0,
                            params_size: 2,
                            layer_size: 4
                        }
                    )
                },
            ),
            (
                json!({"type": "CLIP_ELEMENTWISE", "input": 1, "parameters_max": 7}),
                |e| matches!(e, E::ParametersIndexOutOfBounds { index: 7 }),
            ),
            (
                json!({"type": "MAP_TRANSFORM", "input": 0, "output": 1, "internal_input_index": 0,
                    "internal_output_index": 0, "map": 1, "size": 1, "default": [0.0]}),
                |e| matches!(e, E::MapIndexOutOfBounds { index: 1 }),
            ),
            (
                json!({"type": "MAP_TRANSFORM", "input": 0, "output": 1, "internal_input_index": 4,
                    "internal_output_index": 0, "map": 0, "size": 1, "default": [0.0]}),
                |e| matches!(e, E::InternalInputIndexOutOfBounds { index: 4 }),
            ),
            (
                json!({"type": "MAP_TRANSFORM", "input": 0, "output": 1, "internal_input_index": 0,
                    "internal_output_index": 1, "map": 0, "size": 2, "default": [0.0, 0.0]}),
                |e| matches!(e, E::InternalOutputIndexOutOfBounds { .. }),
            ),
            (
                json!({"type": "ADD_ELEMENTWISE_BUFFERS", "input": [0], "output": 2}),
                |e| matches!(e, E::InsufficientInputBuffers),
            ),
            (
                json!({"type": "MULTIPLY_ELEMENTWISE_BUFFERS", "input": [0, 1], "output": 2}),
                |e| {
                    matches!(
                        e,
                        E::InputBufferSizeMismatch {
                            index: 1,
                            actual_size: 2,
                            expected_size: 4
                        }
                    )
                },
            ),
            (
                json!({"type": "ADD_BUFFER_HEADS", "input": [0, 1], "output": 1}),
                |e| matches!(e, E::InputBufferSizeMismatch { index: 0, .. }),
            ),
            (
                json!({"type": "MULTIPLY_BUFFER_HEADS", "input": [0], "output": 2}),
                |e| matches!(e, E::InsufficientInputBuffers),
            ),
            (
                json!({"type": "ATTENTION", "input": 1, "key": 0, "output": 2, "weights": 0}),
                |e| matches!(e, E::InputBufferSizeMismatch { index: 1, .. }),
            ),
            (
                json!({"type": "DOT", "input": 0, "output": 0, "weights": 0}),
                |e| matches!(e, E::SameInputOutputIndexes { .. }),
            ),
            (
                json!({"type": "DOT", "input": 0, "output": 3, "weights": 0}),
                |e| matches!(e, E::BufferIndexOutOfBounds { .. }),
            ),
        ];
        for (instruction, check) in cases {
            let err = model(
                json!({"instructions": [instruction], "parameters": [[1.0, 1.0]],
                                   "maps": [{"1": [1.0]}]}),
            )
            .err()
            .unwrap_or_else(|| panic!("{instruction} accepted"));
            assert!(check(&err), "{instruction}: {err}");
        }
    }
}
