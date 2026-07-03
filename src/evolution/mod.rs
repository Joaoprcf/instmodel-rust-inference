//! Evolution-strategies training toolkit.
//!
//! Everything a black-box training loop needs, with zero extra
//! dependencies:
//!
//! - [`EsOptimizer`] — OpenAI-style ES: mirrored sampling, centered-rank
//!   fitness shaping, SGD with momentum, fully deterministic per
//!   `(seed, step, pair)`.
//! - [`GaussianStream`] / [`perturbation_seed`] / [`fill_perturbation`] —
//!   the counter-based noise primitives, exposed so external evaluators
//!   (e.g. GPU-side reconstruction) can regenerate any perturbation.
//! - [`cosine_anneal`] — σ / learning-rate decay schedule.
//!
//! Candidates flow into models either via
//! [`InstructionModel::apply_theta`](crate::InstructionModel::apply_theta)
//! (in-place CPU mutation) or
//! [`PopulationPack::write_candidate`](crate::gpu::PopulationPack::write_candidate)
//! (zero-repack GPU packing); both consume the same canonical flat-θ order
//! produced by [`ModelGraph::weights_map`](crate::graph::ModelGraph::weights_map).

pub mod noise;
pub mod optimizer;
pub mod schedule;

pub use noise::{GaussianStream, fill_perturbation, perturbation_seed};
pub use optimizer::{EsConfig, EsOptimizer};
pub use schedule::cosine_anneal;
