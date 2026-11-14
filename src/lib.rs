//! Inference for instmodel JSON models: a validated instruction sequence executed over one
//! unified f32 buffer, plus graph authoring, in-place θ mutation, ES training and GPU/WGSL paths.

pub mod activation;
pub mod benchmarks;
pub mod errors;
pub mod evolution;
pub mod gpu;
pub mod graph;
pub mod high_performance_execution_utils;
pub mod instruction_model;
pub mod instruction_model_info;
mod instruction_model_validation;
pub mod instructions;
pub mod parallel_predict;
pub mod params;
pub mod utils;

pub use activation::Activation;
pub use evolution::{EsConfig, EsOptimizer};
pub use graph::{Constant, Graph, ModelGraph, WeightId};
pub use instruction_model::InstructionModel;
pub use instruction_model_info::{InstructionModelInfo, ValidationData};
pub use parallel_predict::{ParallelPredictOutput, PredictConfig};
pub use params::ParamLayout;
