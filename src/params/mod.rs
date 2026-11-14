//! Flat parameter vector (θ) layout ([`ParamLayout`]) and in-place model mutation
//! (`InstructionModel::apply_theta`/`read_theta`/`try_clone`).

mod apply;
pub mod layout;

pub use layout::{JaggedParams, ParamCopy, ParamKind, ParamLayout};

pub(crate) use apply::ThetaBinding;
