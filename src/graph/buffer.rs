//! DAG node: a value flowing through the graph.
//!
//! A [`Buffer`] is a reference-counted handle to a node. Cloning a `Buffer`
//! shares the same underlying node, so feeding one buffer into several
//! consumers is a genuine DAG (the node is compiled once, not duplicated).
//! Node identity for the planner's visited maps is the address of the shared
//! inner node (`Rc::as_ptr`) — deterministic and scoped to the live graph,
//! with no process-global counter.

use std::rc::Rc;

use super::op::Op;

/// Stable node identity for the planner's visited maps.
pub(crate) type BufferId = *const BufferInner;

/// A value produced by an input declaration or a graph op.
///
/// Cheap to clone; clones share identity (`Rc`).
#[derive(Clone)]
pub struct Buffer(pub(crate) Rc<BufferInner>);

pub(crate) struct BufferInner {
    /// Width of this value (number of scalars).
    pub(crate) size: usize,
    /// The op that produced this buffer; `None` for declared inputs.
    pub(crate) producer: Option<Op>,
    /// Producer inputs (empty for declared inputs).
    pub(crate) inputs: Vec<Buffer>,
    /// Feature names — only meaningful on input buffers; may be empty.
    pub(crate) features: Vec<String>,
}

impl Buffer {
    /// Creates a declared input buffer.
    pub(crate) fn input(size: usize, features: Vec<String>) -> Self {
        Buffer(Rc::new(BufferInner {
            size,
            producer: None,
            inputs: Vec::new(),
            features,
        }))
    }

    /// Creates a buffer produced by `producer` from `inputs`.
    pub(crate) fn produced(size: usize, producer: Op, inputs: Vec<Buffer>) -> Self {
        Buffer(Rc::new(BufferInner {
            size,
            producer: Some(producer),
            inputs,
            features: Vec::new(),
        }))
    }

    /// Width of this value.
    pub fn size(&self) -> usize {
        self.0.size
    }

    /// True for declared inputs (no producer).
    pub fn is_input(&self) -> bool {
        self.0.producer.is_none()
    }

    /// The producing op, or `None` for a declared input.
    pub(crate) fn producer(&self) -> Option<&Op> {
        self.0.producer.as_ref()
    }

    /// The producer's input buffers.
    pub(crate) fn inputs(&self) -> &[Buffer] {
        &self.0.inputs
    }

    /// Declared feature names (empty unless an input with names).
    pub(crate) fn features(&self) -> &[String] {
        &self.0.features
    }

    /// Stable identity for DAG dedup: the address of the shared inner node.
    pub(crate) fn id(&self) -> BufferId {
        Rc::as_ptr(&self.0)
    }
}

impl Drop for BufferInner {
    /// Drains the input chain iteratively so dropping a deep graph cannot
    /// overflow the stack through recursive `Rc` drops.
    fn drop(&mut self) {
        let mut stack: Vec<Buffer> = std::mem::take(&mut self.inputs);
        while let Some(buffer) = stack.pop() {
            if let Ok(mut inner) = Rc::try_unwrap(buffer.0) {
                stack.append(&mut inner.inputs);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_identity() {
        let buffer = Buffer::input(3, Vec::new());
        let clone = buffer.clone();
        assert_eq!(buffer.id(), clone.id());
        assert!(buffer.is_input());
        assert_eq!(buffer.size(), 3);
    }

    #[test]
    fn produced_buffers_record_their_op_and_inputs() {
        let input = Buffer::input(4, Vec::new());
        let produced = Buffer::produced(1, Op::ReduceSum, vec![input.clone()]);
        assert!(!produced.is_input());
        assert_eq!(produced.size(), 1);
        assert_eq!(produced.inputs().len(), 1);
        assert_eq!(produced.inputs()[0].id(), input.id());
    }

    #[test]
    fn deep_graphs_drop_iteratively_without_stack_overflow() {
        let mut buffer = Buffer::input(1, Vec::new());
        for _ in 0..200_000 {
            buffer = Buffer::produced(1, Op::ReduceSum, vec![buffer]);
        }
        drop(buffer);
    }
}
