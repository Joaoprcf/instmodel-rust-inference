//! Performance measurement utilities for benchmarks.

use super::benchmark_types::PerformanceResults;
use log::info;
use std::time::Instant;

/// Benchmark execution function that measures performance
pub fn benchmark_method<F>(
    name: &str,
    num_executions: u32,
    mut benchmark_fn: F,
) -> PerformanceResults
where
    F: FnMut(),
{
    info!("Benchmarking {} ({} executions)...", name, num_executions);

    // Warm-up runs to ensure consistent performance measurement
    for _ in 0..5 {
        benchmark_fn();
    }

    let start = Instant::now();
    for i in 0..num_executions {
        benchmark_fn();
        if (i + 1) % (num_executions / 10) == 0 {
            info!("  Progress: {}/{}", i + 1, num_executions);
        }
    }
    let duration = start.elapsed();

    PerformanceResults::new(name.to_string(), duration.as_nanos(), num_executions)
}

/// Prints detailed performance analysis
pub fn print_performance_analysis(results: &[PerformanceResults]) {
    if results.is_empty() {
        return;
    }

    let baseline = &results[0]; // First result is typically the baseline

    println!("\n{}", "=".repeat(80));
    println!("Detailed Results");
    println!("{}", "=".repeat(80));

    for result in results {
        println!("\n📊 {}", result.method);
        println!(
            "   Average time: {:.3} ms ({} ns)",
            result.average_time_ms, result.average_time_ns
        );
        println!(
            "   Total time: {:.3} ms",
            result.total_time_ns as f64 / 1_000_000.0
        );
        println!("   Executions: {}", result.num_executions);

        if result.method != baseline.method {
            println!(
                "   Overhead vs baseline: {:.2}x ({:.1}%)",
                result.overhead_ratio(baseline),
                result.overhead_percentage(baseline)
            );
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("Performance Analysis");
    println!("{}", "=".repeat(80));

    // Speed rankings
    println!("\n🚀 Speed Rankings (fastest to slowest):");
    let mut sorted_results = results.to_vec();
    sorted_results.sort_by_key(|r| r.average_time_ns);

    for (i, result) in sorted_results.iter().enumerate() {
        let rank_emoji = match i {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        println!(
            "   {} {}: {:.3} ms",
            rank_emoji, result.method, result.average_time_ms
        );
    }

    // Simplified overhead analysis when manual and framework results are present
    let manual_result = results.iter().find(|r| r.method.contains("Manual"));
    let framework_result = results.iter().find(|r| r.method.contains("Framework"));

    if let (Some(manual), Some(framework)) = (manual_result, framework_result) {
        println!("\n📈 Framework Overhead Analysis:");
        println!(
            "   Framework vs Manual: {:.2}x overhead ({:.1}%)",
            framework.overhead_ratio(manual),
            framework.overhead_percentage(manual)
        );
    }
}

/// Verifies that outputs from different implementations match
pub fn verify_outputs_match(manual_output: &[f32], framework_output: &[f32]) -> bool {
    const EPSILON: f32 = 1e-6;
    if manual_output.len() != framework_output.len() {
        return false;
    }
    for (manual, framework) in manual_output.iter().zip(framework_output.iter()) {
        if (manual - framework).abs() > EPSILON {
            println!(
                "Output mismatch: manual={}, framework={}, diff={}",
                manual,
                framework,
                (manual - framework).abs()
            );
            return false;
        }
    }
    true
}
