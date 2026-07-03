//! End-to-end CPU evolution-strategies training on a graph-authored model.
//!
//! The 1.0.0 workflow: author a weightless graph, size θ from its canonical
//! layout, mutate ONE prebuilt model in place per candidate (`apply_theta`,
//! no rebuild), and train with the deterministic OpenAI-ES optimizer.
//! Finishes with a serde round-trip of the compiled model.
//!
//! Run with: `cargo run --release --example es_train`

use instmodel_inference::activation::Activation;
use instmodel_inference::errors::{EvolutionError, GraphError, InstructionModelError};
use instmodel_inference::evolution::{EsConfig, EsOptimizer, GaussianStream, cosine_anneal};
use instmodel_inference::graph::{Graph, ModelGraph};
use instmodel_inference::{InstructionModel, InstructionModelInfo};
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
    #[error("JSON round-trip failed: {0}")]
    Json(#[from] serde_json::Error),
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    if let Err(e) = run() {
        error!("es_train failed: {e}");
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

fn dataset_mse(
    model: &InstructionModel,
    inputs: &[f32],
    targets: &[f32],
) -> Result<f64, InstructionModelError> {
    let mut squared_error = 0.0f64;
    for (row, target) in inputs.chunks_exact(FEATURE_SIZE).zip(targets.iter()) {
        let prediction = model.predict(row)?[0];
        squared_error += f64::from(prediction - target).powi(2);
    }
    Ok(squared_error / targets.len() as f64)
}

fn run() -> Result<(), ExampleError> {
    let model_graph = build_graph();
    let layout = model_graph.weights_map()?;
    info!(
        "graph compiled: {} trainable parameters across {} weight slots",
        layout.total,
        layout.weight_shapes.len()
    );

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

    // One model for the whole run: every candidate is a pure in-place
    // weight overwrite, never a rebuild.
    let mut model = InstructionModel::new(model_graph.compile_zeroed()?)?;

    let mut theta_f32 = Vec::with_capacity(layout.total);
    let mut fitness = vec![0.0f64; optimizer.population_size()];

    for step in 0..TOTAL_STEPS {
        // Rank-shaped ES gradients keep O(1) magnitude even at the optimum,
        // so both σ and the learning rate must decay for θ to settle.
        optimizer.set_sigma(BASE_SIGMA * cosine_anneal(step, TOTAL_STEPS, 0.2))?;
        optimizer.set_learning_rate(BASE_LEARNING_RATE * cosine_anneal(step, TOTAL_STEPS, 0.05))?;

        optimizer.ask()?;
        for (candidate, fit) in fitness.iter_mut().enumerate() {
            optimizer.candidate_f32_into(candidate, &mut theta_f32)?;
            model.apply_theta(&theta_f32)?;
            *fit = -dataset_mse(&model, &inputs, &targets)?;
        }
        optimizer.tell(&fitness)?;

        if step % 50 == 0 || step == TOTAL_STEPS - 1 {
            optimizer.theta_f32_into(&mut theta_f32);
            model.apply_theta(&theta_f32)?;
            let mse = dataset_mse(&model, &inputs, &targets)?;
            info!("step {step:>4}: train MSE {mse:.6}");
        }
    }

    optimizer.theta_f32_into(&mut theta_f32);
    model.apply_theta(&theta_f32)?;
    let final_mse = dataset_mse(&model, &inputs, &targets)?;
    info!("finished {TOTAL_STEPS} steps: final train MSE {final_mse:.6}");

    // Ship the result: compile the graph with the trained θ and prove the
    // serialized model round-trips to identical predictions.
    let compiled = model_graph.compile(&theta_f32)?;
    let json = serde_json::to_string(&compiled)?;
    let reloaded: InstructionModelInfo = serde_json::from_str(&json)?;
    let reloaded_model = InstructionModel::new(reloaded)?;
    for row in inputs.chunks_exact(FEATURE_SIZE) {
        let live = model.predict(row)?;
        let restored = reloaded_model.predict(row)?;
        assert_eq!(live, restored, "serde round-trip changed predictions");
    }
    info!(
        "serde round-trip verified bitwise over {ROWS} rows ({} bytes of JSON)",
        json.len()
    );
    Ok(())
}
