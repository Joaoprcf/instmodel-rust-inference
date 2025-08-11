//! Neural inference library for executing optimized computation sequences.
//!
//! This library provides functionality to run neural network inference using
//! a sequence of instructions that operate on computation buffers. It supports
//! various operations like dot products, activations, copying, and element-wise operations.

pub mod activation;
pub mod errors;
pub mod instruction_model;
pub mod instruction_model_info;
pub mod instructions;
pub mod utils;

pub use activation::Activation;
pub use instruction_model::InstructionModel;
pub use instruction_model_info::{InstructionModelInfo, ValidationData};
