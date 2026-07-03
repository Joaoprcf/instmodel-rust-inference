//! Weightless graph authoring for instruction models.
//!
//! Declare an architecture once as a small DAG — structure only, no weight
//! data — then compile it with any flat parameter vector θ:
//!
//! - [`ModelGraph::weights_map`] returns the [`ParamLayout`]: the canonical
//!   flat-θ ↔ model offset table (per weight slot in first-seen order, a
//!   row-major `[out × in]` weight run then the `[out]` bias run).
//! - [`ModelGraph::compile`] scatters θ through that layout into a runnable
//!   [`InstructionModelInfo`].
//! - [`ModelGraph::to_model`] builds the executable [`InstructionModel`]
//!   directly.
//!
//! Reusing a [`WeightId`] (via [`Graph::dense_shared`] /
//! [`Graph::attention_shared`]) shares one weight tensor across ops: a single
//! θ slot referenced by several instructions. Constants ([`Constant`]) and
//! maps become `parameters`/`maps` slots and are NOT part of θ.
//!
//! # Example
//!
//! ```
//! use instmodel_inference::activation::Activation;
//! use instmodel_inference::graph::Graph;
//!
//! let graph = Graph::new();
//! let x = graph.input(3, None);
//! let hidden = graph.dense(&x, 4, Some(Activation::Tanh));
//! let y = graph.dense(&hidden, 1, None);
//! let model_graph = graph.model(vec![&x], &y);
//!
//! let layout = model_graph.weights_map().unwrap();
//! assert_eq!(layout.total, (4 * 3 + 4) + (4 + 1));
//!
//! let theta = vec![0.1f32; layout.total];
//! let model = model_graph.to_model(&theta).unwrap();
//! assert_eq!(model.predict(&[1.0, 2.0, 3.0]).unwrap().len(), 1);
//! ```

mod buffer;
mod compile;
mod op;
mod plan;
mod weights;

use std::cell::Cell;
use std::collections::HashMap;

use crate::activation::Activation;
use crate::errors::GraphError;
use crate::instruction_model::InstructionModel;
use crate::instruction_model_info::InstructionModelInfo;
use crate::params::ParamLayout;

pub use buffer::Buffer;
pub use compile::{compile, compile_zeroed, weights_map};
pub use op::Constant;
pub use weights::WeightId;

use op::Op;

/// Weightless graph builder. All methods take `&self`; [`WeightId`]s are
/// minted from an interior counter, so ids are deterministic per graph.
pub struct Graph {
    next_weight: Cell<u32>,
}

impl Graph {
    /// Creates an empty graph builder.
    pub fn new() -> Self {
        Graph {
            next_weight: Cell::new(0),
        }
    }

    /// Declares an input of `size` scalars, optionally naming each scalar.
    ///
    /// Compiled feature names are exposed only when every input scalar of the
    /// whole graph is named.
    pub fn input(&self, size: usize, features: Option<Vec<String>>) -> Buffer {
        Buffer::input(size, features.unwrap_or_default())
    }

    /// Mints a fresh weight handle for explicit sharing via
    /// [`Graph::dense_shared`] / [`Graph::attention_shared`].
    pub fn weight(&self) -> WeightId {
        let id = self.next_weight.get();
        self.next_weight.set(id + 1);
        WeightId(id)
    }

    /// Dense layer with a private weight tensor:
    /// `activation(input · Wᵀ + b)` with `W` of shape `[out_size, input.size()]`.
    pub fn dense(&self, input: &Buffer, out_size: usize, activation: Option<Activation>) -> Buffer {
        self.dense_shared(input, out_size, activation, self.weight())
    }

    /// Dense layer using `weight`; reusing the handle shares one tensor.
    pub fn dense_shared(
        &self,
        input: &Buffer,
        out_size: usize,
        activation: Option<Activation>,
        weight: WeightId,
    ) -> Buffer {
        Buffer::produced(
            out_size,
            Op::Dense {
                weight,
                out_size,
                activation,
            },
            vec![input.clone()],
        )
    }

    /// Concatenates `inputs` along the feature axis.
    pub fn concat(&self, inputs: &[&Buffer]) -> Buffer {
        let size = inputs.iter().map(|buffer| buffer.size()).sum();
        Buffer::produced(
            size,
            Op::Concat,
            inputs.iter().map(|&buffer| buffer.clone()).collect(),
        )
    }

    /// Selects `indexes` (validated at plan time) from `input` into a new
    /// buffer.
    pub fn gather(&self, input: &Buffer, indexes: Vec<usize>) -> Buffer {
        let size = indexes.len();
        Buffer::produced(size, Op::Gather { indexes }, vec![input.clone()])
    }

    /// Applies `activation` element-wise.
    pub fn activation(&self, input: &Buffer, activation: Activation) -> Buffer {
        Buffer::produced(
            input.size(),
            Op::Activation { activation },
            vec![input.clone()],
        )
    }

    /// Clamps element-wise between optional bounds; at least one bound is
    /// required.
    pub fn clip(&self, input: &Buffer, min: Option<Constant>, max: Option<Constant>) -> Buffer {
        Buffer::produced(input.size(), Op::Clip { min, max }, vec![input.clone()])
    }

    /// Adds a constant element-wise.
    pub fn add_const(&self, input: &Buffer, value: Constant) -> Buffer {
        Buffer::produced(input.size(), Op::AddConst { value }, vec![input.clone()])
    }

    /// Multiplies by a constant element-wise.
    pub fn mul_const(&self, input: &Buffer, value: Constant) -> Buffer {
        Buffer::produced(input.size(), Op::MulConst { value }, vec![input.clone()])
    }

    /// Normalization sugar: `(input - mean) / std`, emitted as an element-wise
    /// add of `-mean` followed by a multiply by `1 / std`. The caller is
    /// responsible for stabilizing `std` (no zero entries).
    pub fn normalize(&self, input: &Buffer, mean: Vec<f32>, std: Vec<f32>) -> Buffer {
        let negated_mean: Vec<f32> = mean.into_iter().map(|value| -value).collect();
        let inverse_std: Vec<f32> = std.into_iter().map(|value| 1.0 / value).collect();
        let centered = self.add_const(input, Constant::PerElement(negated_mean));
        self.mul_const(&centered, Constant::PerElement(inverse_std))
    }

    /// Element-wise sum of two or more equally sized buffers.
    pub fn add(&self, inputs: &[&Buffer]) -> Buffer {
        let size = inputs.first().map_or(0, |buffer| buffer.size());
        Buffer::produced(
            size,
            Op::AddBuffers,
            inputs.iter().map(|&buffer| buffer.clone()).collect(),
        )
    }

    /// Element-wise product of two or more equally sized buffers.
    pub fn mul(&self, inputs: &[&Buffer]) -> Buffer {
        let size = inputs.first().map_or(0, |buffer| buffer.size());
        Buffer::produced(
            size,
            Op::MulBuffers,
            inputs.iter().map(|&buffer| buffer.clone()).collect(),
        )
    }

    /// Adds each head value of `heads` across its segment of `data`;
    /// `data.size()` must be a multiple of `heads.size()`.
    pub fn add_heads(&self, data: &Buffer, heads: &Buffer) -> Buffer {
        Buffer::produced(data.size(), Op::AddHeads, vec![data.clone(), heads.clone()])
    }

    /// Multiplies each segment of `data` by its head value from `heads`.
    pub fn mul_heads(&self, data: &Buffer, heads: &Buffer) -> Buffer {
        Buffer::produced(data.size(), Op::MulHeads, vec![data.clone(), heads.clone()])
    }

    /// Sums all elements of `input` into a single scalar.
    pub fn reduce_sum(&self, input: &Buffer) -> Buffer {
        Buffer::produced(1, Op::ReduceSum, vec![input.clone()])
    }

    /// Attention with a private weight tensor of shape
    /// `[query.size(), key.size()]`; the output has the query's size.
    pub fn attention(&self, query: &Buffer, key: &Buffer) -> Buffer {
        self.attention_shared(query, key, self.weight())
    }

    /// Attention using `weight`; reusing the handle shares one tensor.
    pub fn attention_shared(&self, query: &Buffer, key: &Buffer, weight: WeightId) -> Buffer {
        Buffer::produced(
            query.size(),
            Op::Attention { weight },
            vec![query.clone(), key.clone()],
        )
    }

    /// Keyed lookup: reads the first scalar of `key` and emits the mapped
    /// vector (or `default_value` when the key is absent). All map values must
    /// share `default_value`'s length.
    pub fn map_transform(
        &self,
        key: &Buffer,
        map: HashMap<String, Vec<f32>>,
        default_value: Vec<f32>,
    ) -> Buffer {
        let size = default_value.len();
        Buffer::produced(
            size,
            Op::MapTransform { map, default_value },
            vec![key.clone()],
        )
    }

    /// Finalizes the graph: `inputs` become computation buffers `0..n` in
    /// declaration order, `output` is the value the compiled model returns.
    pub fn model(&self, inputs: Vec<&Buffer>, output: &Buffer) -> ModelGraph {
        ModelGraph {
            inputs: inputs.into_iter().cloned().collect(),
            output: output.clone(),
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// A finalized graph: declared inputs plus the output value.
pub struct ModelGraph {
    pub(crate) inputs: Vec<Buffer>,
    pub(crate) output: Buffer,
}

impl ModelGraph {
    /// The canonical flat-θ ↔ model offset table, without weight data.
    pub fn weights_map(&self) -> Result<ParamLayout, GraphError> {
        compile::weights_map(self)
    }

    /// Compiles the graph with flat parameters `theta` into a runnable
    /// [`InstructionModelInfo`].
    pub fn compile(&self, theta: &[f32]) -> Result<InstructionModelInfo, GraphError> {
        compile::compile(self, theta)
    }

    /// Compiles the graph with all parameters zeroed — a template for
    /// workflows that immediately overwrite θ in place
    /// ([`InstructionModel::apply_theta`]).
    pub fn compile_zeroed(&self) -> Result<InstructionModelInfo, GraphError> {
        compile::compile_zeroed(self)
    }

    /// Compiles and builds the executable model in one step.
    pub fn to_model(&self, theta: &[f32]) -> Result<InstructionModel, GraphError> {
        Ok(InstructionModel::new(self.compile(theta)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_ids_are_deterministic_per_graph() {
        let graph = Graph::new();
        assert_eq!(graph.weight(), WeightId(0));
        assert_eq!(graph.weight(), WeightId(1));

        let fresh = Graph::default();
        assert_eq!(fresh.weight(), WeightId(0));
    }

    #[test]
    fn builder_tracks_buffer_sizes() {
        let graph = Graph::new();
        let a = graph.input(3, None);
        let b = graph.input(2, None);

        assert_eq!(graph.concat(&[&a, &b]).size(), 5);
        assert_eq!(graph.dense(&a, 7, None).size(), 7);
        assert_eq!(graph.gather(&a, vec![0, 2]).size(), 2);
        assert_eq!(graph.reduce_sum(&a).size(), 1);
        assert_eq!(graph.attention(&a, &b).size(), 3);
        assert_eq!(graph.add_heads(&a, &a).size(), 3);
        assert_eq!(
            graph.map_transform(&b, HashMap::new(), vec![0.0; 4]).size(),
            4
        );
        assert_eq!(
            graph
                .normalize(&a, vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0])
                .size(),
            3
        );
    }

    #[test]
    fn named_inputs_keep_their_feature_names() {
        let graph = Graph::new();
        let named = graph.input(2, Some(vec!["a".to_string(), "b".to_string()]));
        let unnamed = graph.input(2, None);
        assert!(named.is_input());
        assert!(unnamed.is_input());
    }
}
