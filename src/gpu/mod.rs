//! GPU neural network inference module.
//!
//! This module provides GPU-embeddable inference functions that can be
//! called from within existing GPU kernels using WGSL and wgpu.
//!
//! # Architecture
//!
//! The GPU inference is designed for embedding within other GPU kernels,
//! particularly useful for RL scenarios where each episode runs in its
//! own GPU thread and can call model.predict() multiple times.
//!
//! # Memory Layout
//!
//! All model data is packed into a single contiguous f32 array:
//! - Header (13 f32s) - model metadata
//! - Instructions (N x 8 f32s each)
//! - Weights (flattened f32 array)
//! - Parameters (flattened f32 array)
//!
//! # Usage
//!
//! ```ignore
//! use instmodel_inference::gpu::{GpuModel, shaders::get_instmodel_wgsl};
//! use instmodel_inference::InstructionModelInfo;
//!
//! // Create GPU model from existing InstructionModelInfo
//! let gpu_model = GpuModel::from_info(&model_info)?;
//!
//! // Get WGSL shader source to embed in your kernel
//! let wgsl_source = get_instmodel_wgsl(4096);
//!
//! // In your WGSL kernel:
//! // var compute_buffer: array<f32, MAX_SIZE>;
//! // // ... copy input to compute_buffer ...
//! // predict(&model_data, &compute_buffer);
//! // // ... read output from compute_buffer[output_start..] ...
//! ```

pub mod errors;
pub mod gpu_instruction;
pub mod gpu_model;
pub mod shaders;

pub use errors::{GpuModelError, GpuModelResult};
pub use gpu_instruction::GpuInstruction;
pub use gpu_model::GpuModel;
pub use shaders::get_instmodel_wgsl;
