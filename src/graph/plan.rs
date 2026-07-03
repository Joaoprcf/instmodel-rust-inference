//! Weightless planning: DAG → ordered instruction list + buffer table +
//! [`ParamLayout`].
//!
//! Two iterative passes (explicit stacks, no recursion):
//!
//! 1. **Consumer-edge counting** from the output — an op consuming the same
//!    buffer twice counts two edges. The visited set doubles as the
//!    reachability set for the unused-input check.
//! 2. **Post-order emission** — declared inputs are pre-registered as buffers
//!    `0..n`; nodes are memoized on `Rc` address, so fan-out compiles once.
//!    In-place ops (activation / clip / add_const / mul_const) fuse onto their
//!    input's buffer iff that buffer has exactly one consumer edge, is not a
//!    declared input, and the node is not the graph output; otherwise a `COPY`
//!    into a fresh buffer precedes the in-place instruction. Post-order
//!    guarantees the output lands in the last computation buffer (the
//!    executor's output convention); a declared-input output gets a trailing
//!    `COPY`.

use std::collections::{HashMap, HashSet};

use crate::errors::GraphError;
use crate::instruction_model_info::{
    ActivationInstructionInfo, AddBufferHeadsInstructionInfo, AttentionInstructionInfo,
    ClipElementwiseInstructionInfo, CopyInstructionInfo, CopyMaskedInstructionInfo,
    DotInstructionInfo, ElemWiseAddInstructionInfo, ElemWiseBuffersAddInstructionInfo,
    ElemWiseBuffersMulInstructionInfo, ElemWiseMulInstructionInfo, InstructionInfo,
    MapTransformInstructionInfo, MultiplyBufferHeadsInstructionInfo, ReduceSumInstructionInfo,
};
use crate::params::ParamLayout;

use super::ModelGraph;
use super::buffer::{Buffer, BufferId};
use super::op::{Constant, Op};
use super::weights::WeightId;

/// Everything about the compiled model except the parameter values.
pub(crate) struct Plan {
    pub(crate) layout: ParamLayout,
    pub(crate) instructions: Vec<InstructionInfo>,
    pub(crate) buffer_sizes: Vec<usize>,
    pub(crate) parameters: Vec<Vec<f32>>,
    pub(crate) maps: Vec<HashMap<String, Vec<f32>>>,
    pub(crate) features: Option<Vec<String>>,
    pub(crate) feature_size: usize,
}

#[derive(Clone, Copy)]
enum VisitState {
    InProgress,
    Done(usize),
}

struct Planner {
    buffer_sizes: Vec<usize>,
    instructions: Vec<InstructionInfo>,
    parameters: Vec<Vec<f32>>,
    maps: Vec<HashMap<String, Vec<f32>>>,
    /// Weight sharing: handle → slot (== DOT/ATTENTION `weights` index).
    visited_weights: HashMap<WeightId, usize>,
    /// `[out, in]` per slot, in first-seen (emission) order.
    weight_shapes: Vec<[usize; 2]>,
}

/// Weightless planning pass: traversal + structural validation + layout.
pub(crate) fn plan(graph: &ModelGraph) -> Result<Plan, GraphError> {
    if graph.inputs.is_empty() {
        return Err(GraphError::NoInputs);
    }

    let mut states: HashMap<BufferId, VisitState> = HashMap::new();
    let mut planner = Planner {
        buffer_sizes: Vec::new(),
        instructions: Vec::new(),
        parameters: Vec::new(),
        maps: Vec::new(),
        visited_weights: HashMap::new(),
        weight_shapes: Vec::new(),
    };

    // Declared inputs occupy buffers 0..n and seed the visited map so the
    // traversal resolves them as leaves.
    let mut features: Vec<String> = Vec::new();
    let mut feature_size = 0usize;
    for input in &graph.inputs {
        if input.size() == 0 {
            return Err(GraphError::ZeroSizedInput);
        }
        if states.contains_key(&input.id()) {
            return Err(GraphError::DuplicateInput);
        }
        let names = input.features();
        if !names.is_empty() {
            if names.len() != input.size() {
                return Err(GraphError::FeatureNamesLengthMismatch {
                    expected: input.size(),
                    got: names.len(),
                });
            }
            for name in names {
                if name.contains('[') || name.contains(']') {
                    return Err(GraphError::InvalidFeatureName { name: name.clone() });
                }
            }
        }
        let index = planner.buffer_sizes.len();
        planner.buffer_sizes.push(input.size());
        states.insert(input.id(), VisitState::Done(index));
        features.extend(names.iter().cloned());
        feature_size += input.size();
    }

    let (consumer_edges, reached) = count_consumer_edges(graph);
    for input in &graph.inputs {
        if !reached.contains(&input.id()) {
            return Err(GraphError::InputNotVisited);
        }
    }

    // Post-order emission. A node is expanded once (marked in-progress, left
    // on the stack, operands pushed above it); when it resurfaces every
    // operand is done, so it emits. Duplicate frames from fan-out resolve to
    // done and pop silently.
    let output_id = graph.output.id();
    let mut stack: Vec<Buffer> = vec![graph.output.clone()];
    while let Some(top) = stack.last().cloned() {
        match states.get(&top.id()).copied() {
            Some(VisitState::Done(_)) => {
                stack.pop();
            }
            Some(VisitState::InProgress) => {
                let index = planner.emit(&top, &states, &consumer_edges, output_id)?;
                states.insert(top.id(), VisitState::Done(index));
                stack.pop();
            }
            None => {
                // Declared inputs are pre-registered, so an unvisited
                // producerless buffer was never declared.
                if top.producer().is_none() {
                    return Err(GraphError::UndeclaredInput);
                }
                states.insert(top.id(), VisitState::InProgress);
                // Operands are pushed in reverse so the leftmost one is
                // processed first: buffer indices and weight slots follow
                // declaration order (left-to-right post-order).
                for input in top.inputs().iter().rev() {
                    match states.get(&input.id()).copied() {
                        Some(VisitState::Done(_)) => {}
                        Some(VisitState::InProgress) => {
                            // An in-progress operand means the operand is an
                            // ancestor on the DFS stack — impossible for the
                            // acyclic Rc graphs the builder produces.
                            return Err(GraphError::Internal {
                                message: "cycle detected during graph traversal".to_string(),
                            });
                        }
                        None => stack.push(input.clone()),
                    }
                }
            }
        }
    }

    let mut output_index = match states.get(&output_id).copied() {
        Some(VisitState::Done(index)) => index,
        _ => {
            return Err(GraphError::Internal {
                message: "output was not compiled".to_string(),
            });
        }
    };

    // A declared input as the output must still be materialized into a fresh
    // (last) buffer to satisfy the executor's output convention.
    if graph.output.is_input() {
        let copied = planner.alloc(planner.buffer_sizes[output_index], "output")?;
        planner
            .instructions
            .push(InstructionInfo::Copy(CopyInstructionInfo {
                input: output_index,
                output: copied,
                internal_index: 0,
            }));
        output_index = copied;
    }

    if output_index + 1 != planner.buffer_sizes.len() {
        return Err(GraphError::Internal {
            message: format!(
                "output buffer {} is not the last computation buffer {}",
                output_index,
                planner.buffer_sizes.len() - 1
            ),
        });
    }

    // Only expose feature names when every input scalar is named; a partial
    // list would disagree with `feature_size`.
    let features = if features.len() == feature_size {
        Some(features)
    } else {
        None
    };

    Ok(Plan {
        layout: ParamLayout::from_shapes(&planner.weight_shapes),
        instructions: planner.instructions,
        buffer_sizes: planner.buffer_sizes,
        parameters: planner.parameters,
        maps: planner.maps,
        features,
        feature_size,
    })
}

/// Iterative DFS from the output counting consumer edges per node (an op
/// consuming the same buffer twice counts two edges). Returns the edge counts
/// and the set of nodes reachable from the output.
fn count_consumer_edges(graph: &ModelGraph) -> (HashMap<BufferId, usize>, HashSet<BufferId>) {
    let mut edges: HashMap<BufferId, usize> = HashMap::new();
    let mut visited: HashSet<BufferId> = HashSet::new();
    let mut stack: Vec<Buffer> = vec![graph.output.clone()];
    visited.insert(graph.output.id());
    while let Some(node) = stack.pop() {
        for input in node.inputs() {
            *edges.entry(input.id()).or_insert(0) += 1;
            if visited.insert(input.id()) {
                stack.push(input.clone());
            }
        }
    }
    (edges, visited)
}

impl Planner {
    fn alloc(&mut self, size: usize, op: &'static str) -> Result<usize, GraphError> {
        if size == 0 {
            return Err(GraphError::ZeroSizedBuffer { op });
        }
        let index = self.buffer_sizes.len();
        self.buffer_sizes.push(size);
        Ok(index)
    }

    /// Returns the weight slot for `weight`, allocating one on first sight and
    /// validating shape agreement on reuse (sharing).
    fn weight_slot(&mut self, weight: WeightId, shape: [usize; 2]) -> Result<usize, GraphError> {
        if let Some(&slot) = self.visited_weights.get(&weight) {
            if self.weight_shapes[slot] != shape {
                return Err(GraphError::SharedWeightShapeMismatch {
                    weight: weight.0,
                    first: self.weight_shapes[slot],
                    again: shape,
                });
            }
            return Ok(slot);
        }
        let slot = self.weight_shapes.len();
        self.weight_shapes.push(shape);
        self.visited_weights.insert(weight, slot);
        Ok(slot)
    }

    /// Resolves a [`Constant`] to a parameters slot of exactly `size` values.
    fn push_constant(&mut self, value: &Constant, size: usize) -> Result<usize, GraphError> {
        let resolved = match value {
            Constant::Scalar(v) => vec![*v; size],
            Constant::PerElement(values) => {
                if values.len() != size {
                    return Err(GraphError::ConstantLengthMismatch {
                        expected: size,
                        got: values.len(),
                    });
                }
                values.clone()
            }
        };
        let index = self.parameters.len();
        self.parameters.push(resolved);
        Ok(index)
    }

    /// Buffer an in-place op should mutate: the input's own buffer when fusion
    /// is safe, otherwise a fresh copy of it.
    fn in_place_target(
        &mut self,
        node: &Buffer,
        input_index: usize,
        consumer_edges: &HashMap<BufferId, usize>,
        output_id: BufferId,
        op: &'static str,
    ) -> Result<usize, GraphError> {
        let input = &node.inputs()[0];
        let single_consumer = consumer_edges.get(&input.id()).copied().unwrap_or(0) == 1;
        if single_consumer && !input.is_input() && node.id() != output_id {
            return Ok(input_index);
        }
        let copied = self.alloc(self.buffer_sizes[input_index], op)?;
        self.instructions
            .push(InstructionInfo::Copy(CopyInstructionInfo {
                input: input_index,
                output: copied,
                internal_index: 0,
            }));
        Ok(copied)
    }

    fn emit_buffers_op(
        &mut self,
        input_indexes: &[usize],
        op: &'static str,
        is_add: bool,
    ) -> Result<usize, GraphError> {
        if input_indexes.len() < 2 {
            return Err(GraphError::InsufficientOperands {
                op,
                minimum: 2,
                got: input_indexes.len(),
            });
        }
        let expected = self.buffer_sizes[input_indexes[0]];
        for &input in &input_indexes[1..] {
            if self.buffer_sizes[input] != expected {
                return Err(GraphError::OperandSizeMismatch {
                    op,
                    expected,
                    got: self.buffer_sizes[input],
                });
            }
        }
        let output = self.alloc(expected, op)?;
        let instruction = if is_add {
            InstructionInfo::ElemWiseBuffersAdd(ElemWiseBuffersAddInstructionInfo {
                input: input_indexes.to_vec(),
                output,
            })
        } else {
            InstructionInfo::ElemWiseBuffersMul(ElemWiseBuffersMulInstructionInfo {
                input: input_indexes.to_vec(),
                output,
            })
        };
        self.instructions.push(instruction);
        Ok(output)
    }

    fn emit_heads_op(
        &mut self,
        input_indexes: &[usize],
        op: &'static str,
        is_add: bool,
    ) -> Result<usize, GraphError> {
        let data = input_indexes[0];
        let heads = input_indexes[1];
        let data_size = self.buffer_sizes[data];
        let heads_size = self.buffer_sizes[heads];
        if !data_size.is_multiple_of(heads_size) {
            return Err(GraphError::HeadsNotDivisible {
                data_size,
                heads_size,
            });
        }
        let output = self.alloc(data_size, op)?;
        let instruction = if is_add {
            InstructionInfo::AddBufferHeads(AddBufferHeadsInstructionInfo {
                input: vec![data, heads],
                output,
            })
        } else {
            InstructionInfo::MultiplyBufferHeads(MultiplyBufferHeadsInstructionInfo {
                input: vec![data, heads],
                output,
            })
        };
        self.instructions.push(instruction);
        Ok(output)
    }

    /// Emits the instruction(s) for `node` (all operands already compiled) and
    /// returns the computation-buffer index holding its value.
    fn emit(
        &mut self,
        node: &Buffer,
        states: &HashMap<BufferId, VisitState>,
        consumer_edges: &HashMap<BufferId, usize>,
        output_id: BufferId,
    ) -> Result<usize, GraphError> {
        let Some(op) = node.producer() else {
            return Err(GraphError::Internal {
                message: "emit called on a producerless buffer".to_string(),
            });
        };

        let mut input_indexes = Vec::with_capacity(node.inputs().len());
        for input in node.inputs() {
            match states.get(&input.id()).copied() {
                Some(VisitState::Done(index)) => input_indexes.push(index),
                _ => {
                    return Err(GraphError::Internal {
                        message: "operand emitted after its consumer".to_string(),
                    });
                }
            }
        }

        match op {
            Op::Dense {
                weight,
                out_size,
                activation,
            } => {
                let input_index = input_indexes[0];
                let input_size = self.buffer_sizes[input_index];
                let slot = self.weight_slot(*weight, [*out_size, input_size])?;
                let output = self.alloc(*out_size, "dense")?;
                self.instructions
                    .push(InstructionInfo::Dot(DotInstructionInfo {
                        input: input_index,
                        output,
                        weights: slot,
                        activation: *activation,
                    }));
                Ok(output)
            }
            Op::Concat => {
                if input_indexes.is_empty() {
                    return Err(GraphError::InsufficientOperands {
                        op: "concat",
                        minimum: 1,
                        got: 0,
                    });
                }
                let total = input_indexes.iter().map(|&i| self.buffer_sizes[i]).sum();
                let output = self.alloc(total, "concat")?;
                let mut internal_index = 0usize;
                for &input in &input_indexes {
                    self.instructions
                        .push(InstructionInfo::Copy(CopyInstructionInfo {
                            input,
                            output,
                            internal_index,
                        }));
                    internal_index += self.buffer_sizes[input];
                }
                Ok(output)
            }
            Op::Gather { indexes } => {
                if indexes.is_empty() {
                    return Err(GraphError::EmptyGather);
                }
                let input = input_indexes[0];
                let input_size = self.buffer_sizes[input];
                for &index in indexes {
                    if index >= input_size {
                        return Err(GraphError::GatherIndexOutOfBounds { index, input_size });
                    }
                }
                let output = self.alloc(indexes.len(), "gather")?;
                self.instructions
                    .push(InstructionInfo::CopyMasked(CopyMaskedInstructionInfo {
                        input,
                        output,
                        indexes: indexes.clone(),
                    }));
                Ok(output)
            }
            Op::Activation { activation } => {
                let target = self.in_place_target(
                    node,
                    input_indexes[0],
                    consumer_edges,
                    output_id,
                    "activation",
                )?;
                self.instructions
                    .push(InstructionInfo::Activation(ActivationInstructionInfo {
                        input: target,
                        activation: *activation,
                    }));
                Ok(target)
            }
            Op::Clip { min, max } => {
                if min.is_none() && max.is_none() {
                    return Err(GraphError::ClipWithoutBounds);
                }
                let size = self.buffer_sizes[input_indexes[0]];
                let parameters_min = match min {
                    Some(bound) => Some(self.push_constant(bound, size)?),
                    None => None,
                };
                let parameters_max = match max {
                    Some(bound) => Some(self.push_constant(bound, size)?),
                    None => None,
                };
                let target = self.in_place_target(
                    node,
                    input_indexes[0],
                    consumer_edges,
                    output_id,
                    "clip",
                )?;
                self.instructions.push(InstructionInfo::ClipElementwise(
                    ClipElementwiseInstructionInfo {
                        input: target,
                        parameters_min,
                        parameters_max,
                    },
                ));
                Ok(target)
            }
            Op::AddConst { value } => {
                let size = self.buffer_sizes[input_indexes[0]];
                let parameters = self.push_constant(value, size)?;
                let target = self.in_place_target(
                    node,
                    input_indexes[0],
                    consumer_edges,
                    output_id,
                    "add_const",
                )?;
                self.instructions
                    .push(InstructionInfo::ElemWiseAdd(ElemWiseAddInstructionInfo {
                        input: target,
                        parameters,
                    }));
                Ok(target)
            }
            Op::MulConst { value } => {
                let size = self.buffer_sizes[input_indexes[0]];
                let parameters = self.push_constant(value, size)?;
                let target = self.in_place_target(
                    node,
                    input_indexes[0],
                    consumer_edges,
                    output_id,
                    "mul_const",
                )?;
                self.instructions
                    .push(InstructionInfo::ElemWiseMul(ElemWiseMulInstructionInfo {
                        input: target,
                        parameters,
                    }));
                Ok(target)
            }
            Op::AddBuffers => self.emit_buffers_op(&input_indexes, "add", true),
            Op::MulBuffers => self.emit_buffers_op(&input_indexes, "mul", false),
            Op::AddHeads => self.emit_heads_op(&input_indexes, "add_heads", true),
            Op::MulHeads => self.emit_heads_op(&input_indexes, "mul_heads", false),
            Op::ReduceSum => {
                let output = self.alloc(1, "reduce_sum")?;
                self.instructions
                    .push(InstructionInfo::ReduceSum(ReduceSumInstructionInfo {
                        input: input_indexes[0],
                        output,
                    }));
                Ok(output)
            }
            Op::Attention { weight } => {
                let query = input_indexes[0];
                let key = input_indexes[1];
                let query_size = self.buffer_sizes[query];
                let key_size = self.buffer_sizes[key];
                let slot = self.weight_slot(*weight, [query_size, key_size])?;
                let output = self.alloc(query_size, "attention")?;
                self.instructions
                    .push(InstructionInfo::Attention(AttentionInstructionInfo {
                        input: query,
                        key,
                        output,
                        weights: slot,
                    }));
                Ok(output)
            }
            Op::MapTransform { map, default_value } => {
                for (map_key, value) in map {
                    if value.len() != default_value.len() {
                        return Err(GraphError::MapValueLengthMismatch {
                            key: map_key.clone(),
                            expected: default_value.len(),
                            got: value.len(),
                        });
                    }
                }
                let map_index = self.maps.len();
                self.maps.push(map.clone());
                let output = self.alloc(default_value.len(), "map_transform")?;
                self.instructions.push(InstructionInfo::MapTransform(
                    MapTransformInstructionInfo {
                        input: input_indexes[0],
                        output,
                        internal_input_index: 0,
                        internal_output_index: 0,
                        map: map_index,
                        size: default_value.len(),
                        default_value: default_value.clone(),
                    },
                ));
                Ok(output)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::graph::Graph;

    #[test]
    fn single_consumer_in_place_op_fuses_onto_producer_buffer() {
        let graph = Graph::new();
        let x = graph.input(3, None);
        let hidden = graph.dense(&x, 4, Some(Activation::Relu));
        let squashed = graph.activation(&hidden, Activation::Tanh);
        let y = graph.dense(&squashed, 2, None);
        let plan = plan(&graph.model(vec![&x], &y)).unwrap();

        assert_eq!(plan.buffer_sizes, vec![3, 4, 2]);
        assert_eq!(plan.instructions.len(), 3);
        assert!(matches!(
            &plan.instructions[1],
            InstructionInfo::Activation(info) if info.input == 1
        ));
    }

    #[test]
    fn fan_out_in_place_op_copies_before_mutating() {
        let graph = Graph::new();
        let x = graph.input(2, None);
        let hidden = graph.dense(&x, 2, None);
        let squashed = graph.activation(&hidden, Activation::Tanh);
        let y = graph.add(&[&hidden, &squashed]);
        let plan = plan(&graph.model(vec![&x], &y)).unwrap();

        assert_eq!(plan.buffer_sizes, vec![2, 2, 2, 2]);
        assert!(matches!(
            &plan.instructions[1],
            InstructionInfo::Copy(info) if info.input == 1 && info.output == 2
        ));
        assert!(matches!(
            &plan.instructions[2],
            InstructionInfo::Activation(info) if info.input == 2
        ));
        assert!(matches!(
            &plan.instructions[3],
            InstructionInfo::ElemWiseBuffersAdd(info) if info.input == vec![1, 2] && info.output == 3
        ));
    }

    #[test]
    fn in_place_output_is_materialized_into_a_fresh_last_buffer() {
        let graph = Graph::new();
        let x = graph.input(2, None);
        let hidden = graph.dense(&x, 3, None);
        let y = graph.activation(&hidden, Activation::Sigmoid);
        let plan = plan(&graph.model(vec![&x], &y)).unwrap();

        assert_eq!(plan.buffer_sizes, vec![2, 3, 3]);
        assert!(matches!(
            &plan.instructions[1],
            InstructionInfo::Copy(info) if info.input == 1 && info.output == 2
        ));
        assert!(matches!(
            &plan.instructions[2],
            InstructionInfo::Activation(info) if info.input == 2
        ));
    }

    #[test]
    fn in_place_op_on_declared_input_copies_first() {
        let graph = Graph::new();
        let x = graph.input(2, None);
        let shifted = graph.add_const(&x, Constant::Scalar(1.0));
        let y = graph.dense(&shifted, 1, None);
        let plan = plan(&graph.model(vec![&x], &y)).unwrap();

        assert_eq!(plan.buffer_sizes, vec![2, 2, 1]);
        assert!(matches!(
            &plan.instructions[0],
            InstructionInfo::Copy(info) if info.input == 0 && info.output == 1
        ));
        assert!(matches!(
            &plan.instructions[1],
            InstructionInfo::ElemWiseAdd(info) if info.input == 1
        ));
    }

    #[test]
    fn chained_in_place_ops_fuse_into_one_buffer() {
        let graph = Graph::new();
        let x = graph.input(3, None);
        let normalized = graph.normalize(&x, vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 8.0]);
        let y = graph.dense(&normalized, 1, None);
        let plan = plan(&graph.model(vec![&x], &y)).unwrap();

        // Copy(x → 1), AddConst(1), MulConst(1), Dot(1 → 2).
        assert_eq!(plan.buffer_sizes, vec![3, 3, 1]);
        assert_eq!(plan.instructions.len(), 4);
        assert!(matches!(
            &plan.instructions[2],
            InstructionInfo::ElemWiseMul(info) if info.input == 1
        ));
        assert_eq!(
            plan.parameters,
            vec![vec![-1.0, -2.0, -3.0], vec![0.5, 0.25, 0.125],]
        );
    }

    #[test]
    fn input_passthrough_output_gets_a_trailing_copy() {
        let graph = Graph::new();
        let x = graph.input(3, None);
        let plan = plan(&graph.model(vec![&x], &x)).unwrap();

        assert_eq!(plan.buffer_sizes, vec![3, 3]);
        assert!(matches!(
            &plan.instructions[0],
            InstructionInfo::Copy(info) if info.input == 0 && info.output == 1
        ));
    }

    #[test]
    fn fan_out_diamond_compiles_shared_node_once() {
        let graph = Graph::new();
        let x = graph.input(4, None);
        let shared = graph.dense(&x, 3, Some(Activation::Relu));
        let left = graph.dense(&shared, 2, None);
        let right = graph.dense(&shared, 2, None);
        let y = graph.add(&[&left, &right]);
        let plan = plan(&graph.model(vec![&x], &y)).unwrap();

        // shared emitted once: input, shared, left, right, add.
        assert_eq!(plan.buffer_sizes, vec![4, 3, 2, 2, 2]);
        let dot_count = plan
            .instructions
            .iter()
            .filter(|instruction| matches!(instruction, InstructionInfo::Dot(_)))
            .count();
        assert_eq!(dot_count, 3);
    }

    #[test]
    fn deep_chain_plans_iteratively_without_stack_overflow() {
        let graph = Graph::new();
        let x = graph.input(1, None);
        let mut current = graph.dense(&x, 1, None);
        for _ in 0..20_000 {
            current = graph.activation(&current, Activation::Relu);
        }
        let plan = plan(&graph.model(vec![&x], &current)).unwrap();

        // All activations except the output one fuse onto the dense buffer;
        // the output activation materializes into a fresh last buffer.
        assert_eq!(plan.buffer_sizes, vec![1, 1, 1]);
        assert_eq!(plan.instructions.len(), 1 + 19_999 + 1 + 1);
    }

    #[test]
    fn self_consuming_op_counts_two_edges_and_disables_fusion() {
        let graph = Graph::new();
        let x = graph.input(2, None);
        let hidden = graph.dense(&x, 2, None);
        let doubled = graph.add(&[&hidden, &hidden]);
        let y = graph.activation(&doubled, Activation::Relu);
        let plan = plan(&graph.model(vec![&x], &y)).unwrap();

        // `doubled` has one consumer (the output activation), so fusion is
        // disabled only because the activation IS the output.
        assert_eq!(plan.buffer_sizes, vec![2, 2, 2, 2]);
        assert!(matches!(
            &plan.instructions[1],
            InstructionInfo::ElemWiseBuffersAdd(info) if info.input == vec![1, 1]
        ));
    }
}
