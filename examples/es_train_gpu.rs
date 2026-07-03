//! GPU-population evolution-strategies training (`gpu-runtime` feature).
//!
//! Same task as `es_train`, but every candidate of every step is evaluated
//! in a single GPU dispatch: the optimizer's population buffer is written
//! straight into a packed [`PopulationEvaluator`] (weights region is
//! byte-identical to canonical θ, so writes are pure memcpy) and the whole
//! population × batch grid runs in one compute pass.
//!
//! Run with: `cargo run --release --example es_train_gpu --features gpu-runtime`

use instmodel_inference::InstructionModel;
use instmodel_inference::activation::Activation;
use instmodel_inference::errors::{EvolutionError, GraphError, InstructionModelError};
use instmodel_inference::evolution::{EsConfig, EsOptimizer, GaussianStream, cosine_anneal};
use instmodel_inference::gpu::{
    GpuContext, GpuContextOptions, GpuRuntimeError, PopulationEvaluator,
};
use instmodel_inference::graph::{Graph, ModelGraph};
use log::{error, info};
use thiserror::Error;

const FEATURE_SIZE: usize = 8;
const ROWS: usize = 48;
const TOTAL_STEPS: usize = 300;
const BASE_SIGMA: f64 = 0.1;
const BASE_LEARNING_RATE: f64 = 0.1;

#[derive(Error, Debug)]
enum ExampleError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Model(#[from] InstructionModelError),
    #[error(transparent)]
    Evolution(#[from] EvolutionError),
    #[error(transparent)]
    Runtime(#[from] GpuRuntimeError),
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    if let Err(e) = run() {
        error!("es_train_gpu failed: {e}");
        std::process::exit(1);
    }
}

/// input(8, named) → normalize → dense(16, Tanh) → dense(8, Tanh) → dense(1).
fn build_graph() -> ModelGraph {
    let graph = Graph::new();
    let names = (0..FEATURE_SIZE).map(|i| format!("x{i}")).collect();
    let x = graph.input(FEATURE_SIZE, Some(names));
    let normalized = graph.normalize(&x, vec![0.0; FEATURE_SIZE], vec![1.0; FEATURE_SIZE]);
    let hidden = graph.dense(&normalized, 16, Some(Activation::Tanh));
    let hidden = graph.dense(&hidden, 8, Some(Activation::Tanh));
    let output = graph.dense(&hidden, 1, None);
    graph.model(vec![&x], &output)
}

/// Synthetic regression task: y = tanh(x · w_true), inputs standard normal.
fn make_dataset() -> (Vec<f32>, Vec<f32>) {
    let w_true = [0.6, -0.4, 0.25, 0.9, -0.7, 0.3, -0.2, 0.5];
    let mut stream = GaussianStream::new(42);
    let mut inputs = vec![0.0f32; ROWS * FEATURE_SIZE];
    stream.fill_gaussian_f32(&mut inputs);
    let targets = inputs
        .chunks_exact(FEATURE_SIZE)
        .map(|row| {
            let dot: f32 = row.iter().zip(w_true.iter()).map(|(x, w)| x * w).sum();
            dot.tanh()
        })
        .collect();
    (inputs, targets)
}

fn run() -> Result<(), ExampleError> {
    let context = match GpuContext::new(&GpuContextOptions::default()) {
        Ok(context) => context,
        Err(e) => {
            error!("no usable GPU adapter ({e}); this example needs real GPU hardware");
            return Ok(());
        }
    };
    info!("using adapter: {}", context.adapter_info().name);

    let model_graph = build_graph();
    let layout = model_graph.weights_map()?;
    let (inputs, targets) = make_dataset();

    let mut init_stream = GaussianStream::new(7);
    let theta_init: Vec<f64> = (0..layout.total)
        .map(|_| 0.1 * init_stream.next_normal())
        .collect();

    let config = EsConfig {
        pairs: 16,
        sigma: BASE_SIGMA,
        learning_rate: BASE_LEARNING_RATE,
        momentum: 0.0,
        seed: 1234,
    };
    let mut optimizer = EsOptimizer::new(theta_init, config)?;

    let mut evaluator =
        PopulationEvaluator::from_graph(&context, &model_graph, optimizer.population_size(), ROWS)?;
    info!(
        "population evaluator ready: {} candidates × {} rows per dispatch, θ = {} params",
        evaluator.n_candidates(),
        evaluator.batch_size(),
        evaluator.theta_len()
    );

    let mut outputs = vec![0.0f32; optimizer.population_size() * ROWS];
    let mut fitness = vec![0.0f64; optimizer.population_size()];

    for step in 0..TOTAL_STEPS {
        optimizer.set_sigma(BASE_SIGMA * cosine_anneal(step, TOTAL_STEPS, 0.2))?;
        optimizer.set_learning_rate(BASE_LEARNING_RATE * cosine_anneal(step, TOTAL_STEPS, 0.05))?;

        optimizer.ask()?;
        evaluator.write_population_f64(optimizer.population())?;
        evaluator.evaluate(&inputs, &mut outputs)?;

        for (candidate, fit) in fitness.iter_mut().enumerate() {
            let predictions = &outputs[candidate * ROWS..(candidate + 1) * ROWS];
            let squared_error: f64 = predictions
                .iter()
                .zip(targets.iter())
                .map(|(p, t)| f64::from(p - t).powi(2))
                .sum();
            *fit = -(squared_error / ROWS as f64);
        }
        optimizer.tell(&fitness)?;

        if step % 50 == 0 || step == TOTAL_STEPS - 1 {
            let best = fitness.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            info!("step {step:>4}: best candidate MSE {:.6}", -best);
        }
    }

    // Verify the GPU-trained θ on the CPU interpreter.
    let mut theta_f32 = Vec::with_capacity(layout.total);
    optimizer.theta_f32_into(&mut theta_f32);
    let mut model = InstructionModel::new(model_graph.compile_zeroed()?)?;
    model.apply_theta(&theta_f32)?;
    let mut squared_error = 0.0f64;
    for (row, target) in inputs.chunks_exact(FEATURE_SIZE).zip(targets.iter()) {
        let prediction = model.predict(row)?[0];
        squared_error += f64::from(prediction - target).powi(2);
    }
    info!(
        "finished {TOTAL_STEPS} steps: CPU-verified final train MSE {:.6}",
        squared_error / ROWS as f64
    );
    Ok(())
}
