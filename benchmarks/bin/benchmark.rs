//! Main benchmark CLI executable.

use env_logger;
use instmodel_inference::benchmarks::{BenchmarkResult, BenchmarkRunner};
use log::error;
use std::env;

fn main() {
    // Initialize logger
    env_logger::init();

    let result = run_benchmarks();

    if let Err(e) = result {
        error!("Benchmark execution failed: {}", e);
        std::process::exit(1);
    }
}

fn run_benchmarks() -> BenchmarkResult<()> {
    let args: Vec<String> = env::args().collect();

    match args.len() {
        1 => {
            // No arguments - run all benchmarks
            BenchmarkRunner::run_all_benchmarks()
        }
        2 => {
            match args[1].as_str() {
                "--list" => {
                    BenchmarkRunner::list_benchmarks();
                    Ok(())
                }
                benchmark_name => {
                    // Run specific benchmark
                    BenchmarkRunner::run_benchmark(benchmark_name)
                }
            }
        }
        3 => {
            if args[1] == "--benchmark" {
                let benchmark_name = &args[2];
                BenchmarkRunner::run_benchmark(benchmark_name)
            } else {
                print_usage();
                Ok(())
            }
        }
        _ => {
            print_usage();
            Ok(())
        }
    }
}

fn print_usage() {
    println!("Usage:");
    println!("  cargo run --bin benchmark --release                    # Run all benchmarks");
    println!("  cargo run --bin benchmark --release -- --list         # List available benchmarks");
    println!("  cargo run --bin benchmark --release -- <benchmark>    # Run specific benchmark");
    println!("  cargo run --bin benchmark --release -- --benchmark <benchmark>");
    println!();
    println!("Available benchmarks:");
    println!("  dot_product             - Neural network with dot product operations");
    println!("  element_wise_buffer_ops - Element-wise buffer operations");
}
