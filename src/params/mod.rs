//! Flat parameter (θ) mapping and in-place model mutation.
//!
//! [`ParamLayout`] defines the crate's canonical flat parameter order — per
//! weight slot, a row-major weight run followed by a bias run — and converts
//! between flat θ vectors and the jagged weights/bias of
//! [`crate::InstructionModelInfo`]. The `apply` module wires that layout into
//! [`crate::InstructionModel`] as `apply_theta`/`read_theta`/`try_clone`,
//! enabling evolutionary-strategy loops to mutate a model's parameters without
//! rebuilding it.

mod apply;
pub mod layout;

pub use layout::{JaggedParams, ParamCopy, ParamKind, ParamLayout};

pub(crate) use apply::ThetaBinding;
