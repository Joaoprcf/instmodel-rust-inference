//! Evolution strategies: [`EsOptimizer`], deterministic counter-based noise (exposed so external
//! evaluators can regenerate perturbations) and [`cosine_anneal`]. Candidates are θ vectors for
//! `InstructionModel::apply_theta` or `PopulationPack::write_candidate`.

pub mod noise;
pub mod optimizer;
pub mod schedule;

pub use noise::{GaussianStream, fill_perturbation, perturbation_seed};
pub use optimizer::{EsConfig, EsOptimizer};
pub use schedule::cosine_anneal;
