//! PopulationEvaluator integration tests (`gpu-runtime` feature).
//!
//! Every GPU-touching test self-skips when no usable adapter is available.

#![cfg(feature = "gpu-runtime")]

use instmodel_inference::InstructionModel;
use instmodel_inference::activation::Activation;
use instmodel_inference::evolution::{EsConfig, EsOptimizer};
use instmodel_inference::gpu::{
    GpuContext, GpuContextOptions, GpuRuntimeError, PopulationEvaluator,
};
use instmodel_inference::graph::{Graph, ModelGraph};

const TOLERANCE: f32 = 1e-4;

fn try_context() -> Option<GpuContext> {
    match GpuContext::new(&GpuContextOptions::default()) {
        Ok(context) => Some(context),
        Err(error) => {
            eprintln!("skipping: no usable GPU ({error})");
            None
        }
    }
}

/// input(3) → normalize → dense(4, Tanh) → dense(2).
fn demo_graph() -> ModelGraph {
    let graph = Graph::new();
    let x = graph.input(3, None);
    let normalized = graph.normalize(&x, vec![0.5; 3], vec![2.0; 3]);
    let hidden = graph.dense(&normalized, 4, Some(Activation::Tanh));
    let output = graph.dense(&hidden, 2, None);
    graph.model(vec![&x], &output)
}

fn candidate_theta(candidate: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|k| ((candidate * 31 + k) as f32) * 0.01 - 0.15)
        .collect()
}

fn batch_inputs(batch_size: usize, feature_size: usize) -> Vec<f32> {
    (0..batch_size * feature_size)
        .map(|k| (k as f32) * 0.125 - 1.0)
        .collect()
}

/// CPU reference: apply each candidate's theta in place and predict each row.
fn cpu_reference(
    model_graph: &ModelGraph,
    thetas: &[Vec<f32>],
    inputs: &[f32],
    batch_size: usize,
) -> Vec<f32> {
    let info = model_graph.compile_zeroed().unwrap();
    let mut model = InstructionModel::new(info).unwrap();
    let feature_size = model.get_feature_size();
    let mut expected = Vec::new();
    for theta in thetas {
        model.apply_theta(theta).unwrap();
        for row in 0..batch_size {
            let input = &inputs[row * feature_size..(row + 1) * feature_size];
            expected.extend(model.predict(input).unwrap());
        }
    }
    expected
}

fn compare(expected: &[f32], actual: &[f32], name: &str) {
    assert_eq!(expected.len(), actual.len(), "{name}: length mismatch");
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e - a).abs();
        assert!(
            diff < TOLERANCE,
            "{name}: mismatch at {i}: expected {e}, got {a} (diff {diff})"
        );
    }
}

#[test]
fn evaluator_matches_cpu_apply_theta() {
    let Some(context) = try_context() else {
        return;
    };
    let model_graph = demo_graph();
    let n_candidates = 5;
    let batch_size = 4;
    let mut evaluator =
        PopulationEvaluator::from_graph(&context, &model_graph, n_candidates, batch_size).unwrap();

    let thetas: Vec<Vec<f32>> = (0..n_candidates)
        .map(|c| candidate_theta(c, evaluator.theta_len()))
        .collect();
    for (candidate, theta) in thetas.iter().enumerate() {
        evaluator.write_candidate(candidate, theta).unwrap();
    }

    let inputs = batch_inputs(batch_size, evaluator.feature_size());
    let mut outputs = vec![0.0f32; n_candidates * batch_size * evaluator.output_size()];
    evaluator.evaluate(&inputs, &mut outputs).unwrap();

    let expected = cpu_reference(&model_graph, &thetas, &inputs, batch_size);
    compare(&expected, &outputs, "evaluator_parity");
}

#[test]
fn write_population_matches_optimizer_shape() {
    let Some(context) = try_context() else {
        return;
    };
    let model_graph = demo_graph();
    let layout = model_graph.weights_map().unwrap();
    let config = EsConfig {
        pairs: 3,
        sigma: 0.2,
        learning_rate: 0.1,
        momentum: 0.0,
        seed: 23,
    };
    let mut optimizer = EsOptimizer::new(vec![0.05; layout.total], config).unwrap();

    let batch_size = 3;
    let mut evaluator = PopulationEvaluator::from_graph(
        &context,
        &model_graph,
        optimizer.population_size(),
        batch_size,
    )
    .unwrap();

    optimizer.ask().unwrap();
    evaluator
        .write_population_f64(optimizer.population())
        .unwrap();

    let inputs = batch_inputs(batch_size, evaluator.feature_size());
    let mut outputs =
        vec![0.0f32; optimizer.population_size() * batch_size * evaluator.output_size()];
    evaluator.evaluate(&inputs, &mut outputs).unwrap();

    let mut buffer = Vec::new();
    let thetas: Vec<Vec<f32>> = (0..optimizer.population_size())
        .map(|candidate| {
            optimizer
                .candidate_f32_into(candidate, &mut buffer)
                .unwrap();
            buffer.clone()
        })
        .collect();
    let expected = cpu_reference(&model_graph, &thetas, &inputs, batch_size);
    compare(&expected, &outputs, "write_population_parity");
}

#[test]
fn buffers_are_reused_across_evaluate_calls() {
    let Some(context) = try_context() else {
        return;
    };
    let model_graph = demo_graph();
    let batch_size = 2;
    let mut evaluator =
        PopulationEvaluator::from_graph(&context, &model_graph, 1, batch_size).unwrap();
    let inputs = batch_inputs(batch_size, evaluator.feature_size());
    let mut first = vec![0.0f32; batch_size * evaluator.output_size()];
    let mut second = vec![0.0f32; batch_size * evaluator.output_size()];

    let theta_a = candidate_theta(0, evaluator.theta_len());
    let theta_b = candidate_theta(7, evaluator.theta_len());

    evaluator.write_candidate(0, &theta_a).unwrap();
    evaluator.evaluate(&inputs, &mut first).unwrap();
    evaluator.write_candidate(0, &theta_b).unwrap();
    evaluator.evaluate(&inputs, &mut second).unwrap();

    assert_ne!(first, second, "new weights must change the outputs");
    let expected_first = cpu_reference(&model_graph, &[theta_a], &inputs, batch_size);
    let expected_second = cpu_reference(&model_graph, &[theta_b], &inputs, batch_size);
    compare(&expected_first, &first, "reuse_first");
    compare(&expected_second, &second, "reuse_second");
}

#[test]
fn dimension_errors() {
    let Some(context) = try_context() else {
        return;
    };
    let model_graph = demo_graph();

    assert!(matches!(
        PopulationEvaluator::from_graph(&context, &model_graph, 2, 0),
        Err(GpuRuntimeError::ZeroBatchSize)
    ));

    let mut evaluator = PopulationEvaluator::from_graph(&context, &model_graph, 2, 2).unwrap();

    let mut outputs = vec![0.0f32; 2 * 2 * evaluator.output_size()];
    assert!(matches!(
        evaluator.evaluate(&[0.0; 3], &mut outputs),
        Err(GpuRuntimeError::InputLengthMismatch { .. })
    ));

    let inputs = batch_inputs(2, evaluator.feature_size());
    let mut short_outputs = vec![0.0f32; 1];
    assert!(matches!(
        evaluator.evaluate(&inputs, &mut short_outputs),
        Err(GpuRuntimeError::OutputLengthMismatch { .. })
    ));

    assert!(matches!(
        evaluator.write_population_f64(&[0.0]),
        Err(GpuRuntimeError::PopulationLengthMismatch { .. })
    ));
}

#[test]
fn software_adapters_are_rejected_by_default() {
    let options = GpuContextOptions {
        allow_software_adapter: false,
        force_fallback_adapter: true,
        required_limits: None,
    };
    match GpuContext::new(&options) {
        Err(GpuRuntimeError::SoftwareAdapterRejected { .. }) | Err(GpuRuntimeError::NoAdapter) => {}
        Err(other) => panic!("unexpected error requesting fallback adapter: {other}"),
        Ok(context) => panic!(
            "fallback adapter was not rejected: {:?}",
            context.adapter_info()
        ),
    }
}
