//! Benchmark suite for neural inference performance testing.
//!
//! This library provides a comprehensive benchmarking framework for comparing
//! manual implementations against the instruction-based neural network framework.

pub mod benchmark_errors;
pub mod benchmark_runner;
pub mod benchmark_types;
pub mod instructions;
pub mod performance_metrics;

pub use benchmark_errors::{BenchmarkError, BenchmarkResult};
pub use benchmark_runner::{BenchmarkRunner, ConfigLoader};
pub use benchmark_types::{
    BenchmarkConfig, DotProductConfig, ElementWiseOpsConfig, NetworkConfig, PerformanceResults,
};
pub use instructions::{
    FrameworkDotProduct, FrameworkElementWiseOps, ManualDotProduct, ManualElementWiseOps,
    create_dot_product_test_data, create_element_wise_test_data,
};
