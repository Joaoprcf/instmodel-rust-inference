//! Weight identity handles.
//!
//! The graph is weightless: it carries no weight *data*, only a [`WeightId`]
//! per dense/attention op. Reusing the same `WeightId` across two ops SHARES
//! one weight tensor — it becomes a single entry in the compiled model's
//! `weights`/`bias` arrays and a single slot in the
//! [`ParamLayout`](crate::params::ParamLayout), with every referencing
//! instruction pointing at the same index.
//!
//! Handles are minted per-graph by [`Graph::weight`](super::Graph::weight), so
//! ids are deterministic and start at 0 for every graph.

/// Identity of a (possibly shared) weight tensor. See module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightId(pub(crate) u32);
