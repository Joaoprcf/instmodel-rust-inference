//! Graph operations and constant operands.

use std::collections::HashMap;

use crate::activation::Activation;

use super::weights::WeightId;

/// A constant operand for element-wise graph ops ([`add_const`], [`mul_const`],
/// [`clip`]), resolved at plan time into a `parameters` slot of the target
/// buffer's size. Constants are NOT part of the flat θ vector.
///
/// [`add_const`]: super::Graph::add_const
/// [`mul_const`]: super::Graph::mul_const
/// [`clip`]: super::Graph::clip
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    /// Broadcast a single value across the whole buffer.
    Scalar(f32),
    /// One value per buffer element; the length must match the buffer size.
    PerElement(Vec<f32>),
}

/// The producer of a non-input buffer.
#[derive(Clone)]
pub(crate) enum Op {
    /// Dense layer, emitted as `DOT`: `output = activation(input · Wᵀ + b)`.
    /// The weight shape is topological (`[out_size, input.size()]`) so it is
    /// never stored on the op.
    Dense {
        weight: WeightId,
        out_size: usize,
        activation: Option<Activation>,
    },
    /// Concatenation along the feature axis: one `COPY` per input at a running
    /// offset into the combined buffer.
    Concat,
    /// Indexed selection from a single input (`COPY_MASKED`).
    Gather { indexes: Vec<usize> },
    /// Element-wise activation (`ACTIVATION`, in-place capable).
    Activation { activation: Activation },
    /// Element-wise clamp (`CLIP_ELEMENTWISE`, in-place capable).
    Clip {
        min: Option<Constant>,
        max: Option<Constant>,
    },
    /// Element-wise constant addition (`ADD_ELEMENTWISE`, in-place capable).
    AddConst { value: Constant },
    /// Element-wise constant multiplication (`MUL_ELEMENTWISE`, in-place capable).
    MulConst { value: Constant },
    /// Element-wise sum of N equally sized buffers (`ADD_ELEMENTWISE_BUFFERS`).
    AddBuffers,
    /// Element-wise product of N equally sized buffers (`MULTIPLY_ELEMENTWISE_BUFFERS`).
    MulBuffers,
    /// Head-wise broadcast addition (`ADD_BUFFER_HEADS`); inputs `[data, heads]`.
    AddHeads,
    /// Head-wise broadcast multiplication (`MULTIPLY_BUFFER_HEADS`); inputs `[data, heads]`.
    MulHeads,
    /// Sum of all elements into a single scalar (`REDUCE_SUM`).
    ReduceSum,
    /// Attention (`ATTENTION`); inputs `[query, key]`, weight shape
    /// `[query.size(), key.size()]`.
    Attention { weight: WeightId },
    /// Keyed lookup emitting a fixed-size vector (`MAP_TRANSFORM`).
    MapTransform {
        map: HashMap<String, Vec<f32>>,
        default_value: Vec<f32>,
    },
}
