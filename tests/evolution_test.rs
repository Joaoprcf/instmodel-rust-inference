//! EsOptimizer integration tests: determinism, mirrored sampling, the exact
//! update rule, convergence, and interop with the θ-mutation APIs.

use instmodel_inference::activation::Activation;
use instmodel_inference::errors::EvolutionError;
use instmodel_inference::evolution::{EsConfig, EsOptimizer, cosine_anneal, fill_perturbation};
use instmodel_inference::gpu::PopulationPack;
use instmodel_inference::graph::Graph;
use instmodel_inference::{InstructionModel, InstructionModelInfo};

fn config(pairs: usize, sigma: f64, learning_rate: f64, momentum: f64, seed: u64) -> EsConfig {
    EsConfig {
        pairs,
        sigma,
        learning_rate,
        momentum,
        seed,
    }
}

#[test]
fn mirrored_pairs_are_symmetric_around_theta() {
    let theta = vec![0.5, -1.25, 3.0, 0.0];
    let mut optimizer = EsOptimizer::new(theta.clone(), config(6, 0.2, 0.1, 0.0, 11)).unwrap();
    let population = optimizer.ask().unwrap();

    let p = theta.len();
    for i in 0..6 {
        let plus = &population[2 * i * p..(2 * i + 1) * p];
        let minus = &population[(2 * i + 1) * p..(2 * i + 2) * p];
        for k in 0..p {
            assert_eq!(
                plus[k] + minus[k],
                2.0 * theta[k],
                "pair {i} not mirrored at component {k}"
            );
            assert_ne!(plus[k], minus[k], "pair {i} has zero perturbation at {k}");
        }
    }
}

#[test]
fn identical_optimizers_follow_identical_trajectories() {
    let theta = vec![1.0, -2.0, 0.5];
    let cfg = config(4, 0.1, 0.05, 0.9, 99);
    let mut a = EsOptimizer::new(theta.clone(), cfg.clone()).unwrap();
    let mut b = EsOptimizer::new(theta, cfg).unwrap();

    for _ in 0..10 {
        let pop_a = a.ask().unwrap().to_vec();
        let pop_b = b.ask().unwrap().to_vec();
        assert_eq!(pop_a, pop_b);

        let fitness: Vec<f64> = pop_a
            .chunks(a.theta_len())
            .map(|candidate| -candidate.iter().map(|v| v * v).sum::<f64>())
            .collect();
        a.tell(&fitness).unwrap();
        b.tell(&fitness).unwrap();
        assert_eq!(a.theta(), b.theta());
        assert_eq!(a.velocity(), b.velocity());
    }
    assert_eq!(a.step(), 10);
}

#[test]
fn single_step_matches_hand_computed_update() {
    // 1 pair, 1 parameter: the whole update collapses to
    // theta += lr * (util_plus - util_minus) * eps / (2 * sigma).
    let sigma = 0.5;
    let learning_rate = 0.25;
    let seed = 7;
    let mut optimizer =
        EsOptimizer::new(vec![0.0], config(1, sigma, learning_rate, 0.0, seed)).unwrap();

    let mut eps = vec![0.0];
    fill_perturbation(seed, 0, 0, &mut eps);

    let population = optimizer.ask().unwrap().to_vec();
    assert_eq!(population[0], sigma * eps[0]);
    assert_eq!(population[1], -sigma * eps[0]);

    // Reward the plus candidate: utilities are +0.5 / -0.5, coeff = 1.
    optimizer.tell(&[1.0, 0.0]).unwrap();
    let expected = learning_rate * eps[0] / (2.0 * sigma);
    assert_eq!(optimizer.theta()[0], expected);
    assert_eq!(optimizer.velocity()[0], eps[0] / (2.0 * sigma));
}

#[test]
fn momentum_accumulates_velocity() {
    let sigma = 0.5;
    let momentum = 0.5;
    let seed = 3;
    let mut optimizer = EsOptimizer::new(vec![0.0], config(1, sigma, 0.1, momentum, seed)).unwrap();

    let mut eps0 = vec![0.0];
    let mut eps1 = vec![0.0];
    fill_perturbation(seed, 0, 0, &mut eps0);
    fill_perturbation(seed, 1, 0, &mut eps1);

    optimizer.ask().unwrap();
    optimizer.tell(&[1.0, 0.0]).unwrap();
    let v0 = eps0[0] / (2.0 * sigma);
    assert_eq!(optimizer.velocity()[0], v0);

    optimizer.ask().unwrap();
    optimizer.tell(&[1.0, 0.0]).unwrap();
    let v1 = momentum * v0 + eps1[0] / (2.0 * sigma);
    assert_eq!(optimizer.velocity()[0], v1);
}

#[test]
fn sigma_is_captured_at_ask_time() {
    let ask_sigma = 0.5;
    let seed = 13;
    let mut optimizer = EsOptimizer::new(vec![0.0], config(1, ask_sigma, 1.0, 0.0, seed)).unwrap();

    let mut eps = vec![0.0];
    fill_perturbation(seed, 0, 0, &mut eps);

    optimizer.ask().unwrap();
    // A schedule adjusting sigma mid-step must not corrupt the update.
    optimizer.set_sigma(0.001).unwrap();
    optimizer.tell(&[1.0, 0.0]).unwrap();

    let expected = eps[0] / (2.0 * ask_sigma);
    assert_eq!(optimizer.theta()[0], expected);
}

#[test]
fn maximizing_identity_moves_theta_up() {
    // Fitness = candidate value itself: regardless of the noise sign the
    // rank-shaped gradient must push theta upward every step.
    let mut optimizer = EsOptimizer::new(vec![0.0], config(1, 0.1, 0.1, 0.0, 21)).unwrap();
    let mut previous = 0.0;
    for _ in 0..20 {
        let population = optimizer.ask().unwrap().to_vec();
        optimizer.tell(&population).unwrap();
        let current = optimizer.theta()[0];
        assert!(current > previous, "theta did not increase: {current}");
        previous = current;
    }
}

#[test]
fn sphere_function_converges() {
    // Rank-shaped ES gradients keep O(1) magnitude even at the optimum, so
    // the noise floor scales with the effective step size — anneal both
    // sigma and the learning rate (as es_strategy does) to converge tightly.
    let theta0 = vec![1.5, -2.0, 0.75, -0.5, 1.0];
    let total_steps = 400;
    let mut optimizer = EsOptimizer::new(theta0, config(16, 0.1, 0.1, 0.0, 5)).unwrap();

    for step in 0..total_steps {
        optimizer
            .set_sigma(0.1 * cosine_anneal(step, total_steps, 0.2))
            .unwrap();
        optimizer
            .set_learning_rate(0.1 * cosine_anneal(step, total_steps, 0.05))
            .unwrap();
        let population = optimizer.ask().unwrap().to_vec();
        let fitness: Vec<f64> = population
            .chunks(optimizer.theta_len())
            .map(|candidate| -candidate.iter().map(|v| v * v).sum::<f64>())
            .collect();
        optimizer.tell(&fitness).unwrap();
    }

    let distance: f64 = optimizer.theta().iter().map(|v| v * v).sum();
    assert!(distance < 0.05, "sphere did not converge: {distance}");
}

#[test]
fn error_paths() {
    let mut optimizer = EsOptimizer::new(vec![0.0, 0.0], config(2, 0.1, 0.1, 0.0, 1)).unwrap();

    assert!(matches!(
        optimizer.tell(&[0.0; 4]),
        Err(EvolutionError::CallOrder { .. })
    ));

    optimizer.ask().unwrap();
    assert!(matches!(
        optimizer.ask(),
        Err(EvolutionError::CallOrder { .. })
    ));
    assert!(matches!(
        optimizer.tell(&[0.0; 3]),
        Err(EvolutionError::FitnessLengthMismatch {
            expected: 4,
            got: 3
        })
    ));
    assert!(matches!(
        optimizer.tell(&[0.0, f64::NAN, 0.0, 0.0]),
        Err(EvolutionError::NonFiniteFitness { index: 1, .. })
    ));
    // The failed tells left the population pending; a valid tell still works.
    optimizer.tell(&[0.0, 1.0, 2.0, 3.0]).unwrap();

    assert!(matches!(
        optimizer.candidate(4),
        Err(EvolutionError::CandidateOutOfBounds { index: 4, count: 4 })
    ));
    assert!(matches!(
        optimizer.set_sigma(-1.0),
        Err(EvolutionError::InvalidConfig { .. })
    ));
    assert!(matches!(
        optimizer.set_learning_rate(f64::INFINITY),
        Err(EvolutionError::InvalidConfig { .. })
    ));
}

#[test]
fn candidates_flow_into_cpu_and_gpu_theta_apis() {
    // End-to-end wiring: graph → optimizer sized from weights_map →
    // candidate_f32_into → apply_theta (CPU) and write_candidate (GPU pack).
    let graph = Graph::new();
    let x = graph.input(3, None);
    let hidden = graph.dense(&x, 4, Some(Activation::Tanh));
    let output = graph.dense(&hidden, 1, None);
    let model_graph = graph.model(vec![&x], &output);

    let layout = model_graph.weights_map().unwrap();
    let mut optimizer =
        EsOptimizer::new(vec![0.0; layout.total], config(2, 0.1, 0.1, 0.0, 17)).unwrap();
    let mut pack = PopulationPack::from_graph(&model_graph, optimizer.population_size()).unwrap();
    let info: InstructionModelInfo = model_graph.compile_zeroed().unwrap();
    let mut cpu_model = InstructionModel::new(info).unwrap();

    optimizer.ask().unwrap();
    let mut buffer = Vec::new();
    for candidate in 0..optimizer.population_size() {
        optimizer
            .candidate_f32_into(candidate, &mut buffer)
            .unwrap();
        assert_eq!(buffer.len(), layout.total);
        cpu_model.apply_theta(&buffer).unwrap();
        pack.write_candidate(candidate, &buffer).unwrap();
        assert_eq!(pack.candidate_theta(candidate).unwrap(), &buffer[..]);
    }

    let prediction = cpu_model.predict(&[0.5, -0.5, 1.0]).unwrap();
    assert_eq!(prediction.len(), 1);
    assert!(prediction[0].is_finite());
}
