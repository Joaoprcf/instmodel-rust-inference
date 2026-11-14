//! GPU inference: [`GpuModel`] packs a model into one f32 blob (header, 8-word instructions,
//! weights in θ order, parameters) that the WGSL `predict()` from [`shaders`] or
//! [`get_specialized_wgsl_lanes`] executes inside your own kernel. [`PopulationPack`] tiles one
//! blob per candidate; the `gpu-runtime` feature adds a wgpu host ([`PopulationEvaluator`]).

pub mod errors;
pub mod gpu_instruction;
pub mod gpu_model;
pub mod population;
#[cfg(feature = "gpu-runtime")]
pub mod runtime;
pub mod shaders;
pub mod specialize;

#[cfg(feature = "gpu-runtime")]
pub use errors::GpuRuntimeError;
pub use errors::{GpuModelError, GpuModelResult, PopulationError};
pub use gpu_instruction::GpuInstruction;
pub use gpu_model::GpuModel;
pub use population::PopulationPack;
#[cfg(feature = "gpu-runtime")]
pub use runtime::{GpuContext, GpuContextOptions, PopulationEvaluator};
pub use shaders::{get_instmodel_wgsl, get_instmodel_wgsl_lanes};
pub use specialize::get_specialized_wgsl_lanes;
