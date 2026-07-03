//! ES-workflow benchmarks: in-place mutation vs full rebuild (the headline
//! 1.0.0 number), population pack write throughput, optimizer step overhead
//! across parameter counts, and model clone cost.
//!
//! Run with: `cargo run --release --bin es_benchmark`

use std::hint::black_box;
use std::time::Instant;

use instmodel_inference::InstructionModel;
use instmodel_inference::activation::Activation;
use instmodel_inference::errors::{EvolutionError, GraphError, InstructionModelError};
use instmodel_inference::evolution::{EsConfig, EsOptimizer, GaussianStream};
use instmodel_inference::gpu::{PopulationError, PopulationPack};
use instmodel_inference::graph::{Graph, ModelGraph};
use log::{error, info};
use thiserror::Error;

#[derive(Error, Debug)]
enum BenchError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Model(#[from] InstructionModelError),
    #[error(transparent)]
    Evolution(#[from] EvolutionError),
    #[error(transparent)]
    Population(#[from] PopulationError),
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    if let Err(e) = run() {
        error!("es_benchmark failed: {e}");
        std::process::exit(1);
    }
}

/// The benchmark model used across all sections: 250 → 300 (ReLU) → 200
/// (Sigmoid), 135,500 trainable parameters.
fn benchmark_graph() -> ModelGraph {
    let graph = Graph::new();
    let x = graph.input(250, None);
    let hidden = graph.dense(&x, 300, Some(Activation::Relu));
    let output = graph.dense(&hidden, 200, Some(Activation::Sigmoid));
    graph.model(vec![&x], &output)
}

fn benchmark_theta(len: usize) -> Vec<f32> {
    let mut stream = GaussianStream::new(99);
    let mut theta = vec![0.0f32; len];
    stream.fill_gaussian_f32(&mut theta);
    for value in &mut theta {
        *value *= 0.05;
    }
    theta
}

fn run() -> Result<(), BenchError> {
    let graph = benchmark_graph();
    let layout = graph.weights_map()?;
    let theta = benchmark_theta(layout.total);
    info!(
        "model: 250 → 300 (ReLU) → 200 (Sigmoid), θ = {} parameters",
        layout.total
    );

    bench_apply_vs_rebuild(&graph, &theta)?;
    bench_pack_write_throughput(&graph, &theta)?;
    bench_ask_tell_overhead()?;
    bench_try_clone(&graph, &theta)?;
    Ok(())
}

/// Headline: mutating a live model in place vs recreating it from scratch.
fn bench_apply_vs_rebuild(graph: &ModelGraph, theta: &[f32]) -> Result<(), BenchError> {
    const REBUILD_ITERS: u32 = 50;
    const APPLY_ITERS: u32 = 2_000;

    let start = Instant::now();
    for _ in 0..REBUILD_ITERS {
        let model = InstructionModel::new(graph.compile(theta)?)?;
        black_box(&model);
    }
    let rebuild_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(REBUILD_ITERS);

    let mut model = InstructionModel::new(graph.compile_zeroed()?)?;
    let start = Instant::now();
    for _ in 0..APPLY_ITERS {
        model.apply_theta(theta)?;
        black_box(&model);
    }
    let apply_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(APPLY_ITERS);

    info!("== in-place mutation vs rebuild ==");
    info!("full rebuild (compile + validate + schedule): {rebuild_us:>10.2} µs/iter");
    info!("apply_theta (in-place overwrite):             {apply_us:>10.2} µs/iter");
    info!("speedup: {:.0}×", rebuild_us / apply_us);
    Ok(())
}

/// Throughput of memcpy-splicing candidates into a packed GPU population.
fn bench_pack_write_throughput(graph: &ModelGraph, theta: &[f32]) -> Result<(), BenchError> {
    const CANDIDATES: usize = 32;
    const REPS: usize = 20;

    let mut pack = PopulationPack::from_graph(graph, CANDIDATES)?;
    let start = Instant::now();
    for _ in 0..REPS {
        for candidate in 0..CANDIDATES {
            pack.write_candidate(candidate, theta)
                .map_err(PopulationError::from)?;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let writes = (REPS * CANDIDATES) as f64;
    let bytes = writes * std::mem::size_of_val(theta) as f64;
    black_box(pack.as_bytes());

    info!("== population pack writes ({CANDIDATES} candidates) ==");
    info!("write_candidate: {:>10.2} µs/write", elapsed * 1e6 / writes);
    info!("throughput:      {:>10.1} MB/s", bytes / elapsed / 1e6);
    Ok(())
}

/// Pure optimizer overhead (noise generation, ranking, update) with a
/// constant-time fitness function.
fn bench_ask_tell_overhead() -> Result<(), BenchError> {
    const STEPS: u32 = 10;

    info!("== EsOptimizer ask + tell overhead (pairs = 16) ==");
    for p in [1_000usize, 10_000, 100_000] {
        let config = EsConfig {
            pairs: 16,
            sigma: 0.1,
            learning_rate: 0.05,
            momentum: 0.9,
            seed: 5,
        };
        let mut optimizer = EsOptimizer::new(vec![0.0; p], config)?;
        let mut fitness = vec![0.0f64; optimizer.population_size()];

        let start = Instant::now();
        for _ in 0..STEPS {
            let population = optimizer.ask()?;
            for (fit, candidate) in fitness.iter_mut().zip(population.chunks(p)) {
                *fit = -candidate[0] * candidate[0];
            }
            optimizer.tell(&fitness)?;
        }
        let ms_per_step = start.elapsed().as_secs_f64() * 1e3 / f64::from(STEPS);
        black_box(optimizer.theta());
        info!("θ = {p:>7} params: {ms_per_step:>8.3} ms/step");
    }
    Ok(())
}

/// Cost of cloning a live model (per-worker copies for parallel evaluation).
fn bench_try_clone(graph: &ModelGraph, theta: &[f32]) -> Result<(), BenchError> {
    const ITERS: u32 = 100;

    let mut model = InstructionModel::new(graph.compile_zeroed()?)?;
    model.apply_theta(theta)?;

    let start = Instant::now();
    for _ in 0..ITERS {
        let clone = model.try_clone()?;
        black_box(&clone);
    }
    let clone_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(ITERS);

    info!("== try_clone ==");
    info!("try_clone: {clone_us:>10.2} µs/clone");
    Ok(())
}
