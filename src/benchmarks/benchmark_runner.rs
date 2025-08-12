//! Core benchmark execution logic.

use super::benchmark_errors::{BenchmarkError, BenchmarkResult};
use super::benchmark_types::{DotProductConfig, ElementWiseOpsConfig};
use super::instructions::{
    FrameworkDotProduct, FrameworkElementWiseOps, ManualDotProduct, ManualElementWiseOps,
    create_dot_product_test_data, create_element_wise_test_data,
};
use super::performance_metrics::{
    benchmark_method, print_performance_analysis, verify_outputs_match,
};
use log::{error, info, warn};
use std::fs;

/// Configuration loader that handles JSON files with fallbacks
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load a configuration file with fallback to defaults
    pub fn load_config<T: serde::de::DeserializeOwned + Default>(
        path: &str,
        config_name: &str,
    ) -> BenchmarkResult<T>
    where
        T: Default,
    {
        match fs::read_to_string(path) {
            Ok(content) => {
                serde_json::from_str(&content).map_err(|e| BenchmarkError::ConfigParseError {
                    path: path.to_string(),
                    source: e,
                })
            }
            Err(_) => {
                warn!(
                    "Config file '{}' not found, using default configuration for {}",
                    path, config_name
                );
                Ok(T::default())
            }
        }
    }

    /// Load dot product configuration
    pub fn load_dot_product_config() -> BenchmarkResult<DotProductConfig> {
        Self::load_config("configs/dot_product.json", "dot_product")
    }

    /// Load element-wise operations configuration
    pub fn load_element_wise_config() -> BenchmarkResult<ElementWiseOpsConfig> {
        Self::load_config(
            "configs/element_wise_buffer_ops.json",
            "element_wise_buffer_ops",
        )
    }
}

/// Main benchmark runner
pub struct BenchmarkRunner;

impl BenchmarkRunner {
    /// Run all available benchmarks
    pub fn run_all_benchmarks() -> BenchmarkResult<()> {
        info!("Starting comprehensive benchmark suite");

        let mut errors = Vec::new();

        // Run dot product benchmark
        if let Err(e) = Self::run_dot_product_benchmark() {
            error!("Dot product benchmark failed: {}", e);
            errors.push(e);
        }

        // Run element-wise operations benchmark
        if let Err(e) = Self::run_element_wise_benchmark() {
            error!("Element-wise operations benchmark failed: {}", e);
            errors.push(e);
        }

        if errors.is_empty() {
            info!("All benchmarks completed successfully");
            Ok(())
        } else {
            Err(BenchmarkError::BenchmarkExecutionError {
                benchmark_name: "all".to_string(),
                message: format!("Some benchmarks failed: {} errors", errors.len()),
            })
        }
    }

    /// Run a specific benchmark by name
    pub fn run_benchmark(benchmark_name: &str) -> BenchmarkResult<()> {
        match benchmark_name {
            "dot_product" => Self::run_dot_product_benchmark(),
            "element_wise_buffer_ops" => Self::run_element_wise_benchmark(),
            _ => Err(BenchmarkError::BenchmarkExecutionError {
                benchmark_name: benchmark_name.to_string(),
                message: "Unknown benchmark name".to_string(),
            }),
        }
    }

    /// List available benchmarks
    pub fn list_benchmarks() {
        println!("Available benchmarks:");
        println!("  dot_product             - Neural network with dot product operations");
        println!("  element_wise_buffer_ops - Element-wise buffer operations");
    }

    /// Run dot product neural network benchmark
    fn run_dot_product_benchmark() -> BenchmarkResult<()> {
        let config = ConfigLoader::load_dot_product_config()?;
        config.validate()?;

        info!("{}", "=".repeat(80));
        info!("Neural Network Performance Benchmark");
        info!(
            "Network Architecture: {} inputs -> {} hidden (ReLU) -> {} outputs (Sigmoid)",
            config.network_config.input_size,
            config.network_config.hidden_size,
            config.network_config.output_size
        );
        info!("{}", "=".repeat(80));

        let (weights_layer1, bias_layer1, weights_layer2, bias_layer2, inputs) =
            create_dot_product_test_data(&config);

        // Create models
        let manual_model = ManualDotProduct::new(
            weights_layer1.clone(),
            bias_layer1.clone(),
            weights_layer2.clone(),
            bias_layer2.clone(),
        );
        let framework_model = FrameworkDotProduct::new(
            &config,
            &weights_layer1,
            &bias_layer1,
            &weights_layer2,
            &bias_layer2,
        )?;
        let manual_buffer_len = manual_model.required_buffer_len();
        let required_memory = framework_model.required_memory();

        // Verify outputs match
        info!("Verifying output consistency between manual and framework implementations...");
        let mut manual_verification_buffer = vec![0.0f32; manual_buffer_len];
        let manual_result = manual_model
            .compute_with_buffer(&inputs, &mut manual_verification_buffer)
            .to_vec();
        let mut framework_verification_buffer = vec![0.0f32; required_memory];
        let framework_result = framework_model
            .run_with_buffer(&inputs, &mut framework_verification_buffer)?
            .to_vec();

        if verify_outputs_match(&manual_result, &framework_result) {
            info!("✅ Outputs match - implementations are consistent");
            info!(
                "   Sample values: {:?}",
                &manual_result[..5.min(manual_result.len())]
            );
        } else {
            error!("❌ Outputs do not match - there may be an implementation bug");
            return Err(BenchmarkError::BenchmarkExecutionError {
                benchmark_name: "dot_product".to_string(),
                message: "Output verification failed".to_string(),
            });
        }

        info!("{}", "=".repeat(80));
        info!("Performance Benchmarks");
        info!("{}", "=".repeat(80));

        let num_executions = config.num_executions;
        let mut results = Vec::new();

        // 1. Manual implementation (pre-allocated buffer)
        let mut manual_benchmark_buffer = vec![0.0f32; manual_buffer_len];
        let manual_results = benchmark_method("Manual Implementation", num_executions, || {
            let _ = manual_model.compute_with_buffer(&inputs, &mut manual_benchmark_buffer);
        });
        results.push(manual_results);

        // 2. Framework implementation (pre-allocated buffer)
        let mut framework_benchmark_buffer = vec![0.0f32; required_memory];
        let framework_buffer_results =
            benchmark_method("Framework Implementation", num_executions, || {
                framework_model
                    .run_with_buffer(&inputs, &mut framework_benchmark_buffer)
                    .expect("Prediction failed");
            });
        results.push(framework_buffer_results);

        // Print analysis
        print_performance_analysis(&results);

        println!("\n💾 Memory Requirements:");
        println!(
            "   Framework buffer size: {} floats ({} KB)",
            required_memory,
            (required_memory * 4) / 1024
        );
        println!(
            "   Manual buffer size: {} floats ({} KB)",
            manual_buffer_len,
            (manual_buffer_len * 4) / 1024
        );

        println!("\n{}", "=".repeat(80));
        println!("Benchmark Complete");
        println!("{}", "=".repeat(80));

        Ok(())
    }

    /// Run element-wise operations benchmark
    fn run_element_wise_benchmark() -> BenchmarkResult<()> {
        let config = ConfigLoader::load_element_wise_config()?;
        config.validate()?;

        info!("{}", "=".repeat(80));
        info!("Element-wise Operations Performance Benchmark");
        info!("Buffer sizes: {:?}", config.buffer_sizes);
        info!("Input size: {}", config.input_size);
        info!("Operations: {:?}", config.operations);
        info!("{}", "=".repeat(80));

        let input_data = create_element_wise_test_data(&config);

        // Create models
        let manual_model = ManualElementWiseOps::new(&config);
        let framework_model = FrameworkElementWiseOps::new(&config)?;
        let manual_buffer_len = manual_model.required_buffer_len();
        let required_memory = framework_model.required_memory();

        // Verify outputs match
        info!("Verifying output consistency between manual and framework implementations...");
        let mut manual_verification_buffer = vec![0.0f32; manual_buffer_len];
        let (manual_add, manual_mul) =
            manual_model.compute_with_buffer(&input_data, &mut manual_verification_buffer);
        let mut framework_verification_buffer = vec![0.0f32; required_memory];
        let (framework_add, framework_mul) =
            framework_model.run_with_buffer(&input_data, &mut framework_verification_buffer)?;

        if verify_outputs_match(manual_add, framework_add)
            && verify_outputs_match(manual_mul, framework_mul)
        {
            info!("✅ Outputs match - implementations are consistent");
            info!(
                "   Sample add values: {:?}",
                &manual_add[..5.min(manual_add.len())]
            );
            info!(
                "   Sample multiply values: {:?}",
                &manual_mul[..5.min(manual_mul.len())]
            );
        } else {
            error!("❌ Outputs do not match - there may be an implementation bug");
            return Err(BenchmarkError::BenchmarkExecutionError {
                benchmark_name: "element_wise_ops".to_string(),
                message: "Output verification failed".to_string(),
            });
        }

        info!("{}", "=".repeat(80));
        info!("Performance Benchmarks");
        info!("{}", "=".repeat(80));

        let num_executions = config.num_executions;
        let mut results = Vec::new();

        // 1. Manual implementation (pre-allocated buffer)
        let mut manual_benchmark_buffer = vec![0.0f32; manual_buffer_len];
        let manual_results = benchmark_method("Manual Implementation", num_executions, || {
            let _ = manual_model.compute_with_buffer(&input_data, &mut manual_benchmark_buffer);
        });
        results.push(manual_results);

        // 2. Framework implementation (pre-allocated buffer)
        let mut framework_benchmark_buffer = vec![0.0f32; required_memory];
        let framework_buffer_results =
            benchmark_method("Framework Implementation", num_executions, || {
                framework_model
                    .run_with_buffer(&input_data, &mut framework_benchmark_buffer)
                    .expect("Computation failed");
            });
        results.push(framework_buffer_results);

        // Print analysis
        print_performance_analysis(&results);

        println!("\n💾 Memory Requirements:");
        println!(
            "   Framework buffer size: {} floats ({} KB)",
            required_memory,
            (required_memory * 4) / 1024
        );
        println!(
            "   Manual buffer size: {} floats ({} KB)",
            manual_buffer_len,
            (manual_buffer_len * 4) / 1024
        );

        println!("\n{}", "=".repeat(80));
        println!("Benchmark Complete");
        println!("{}", "=".repeat(80));

        Ok(())
    }
}
