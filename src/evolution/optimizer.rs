//! OpenAI-style evolution strategies: mirrored sampling, centered-rank
//! utilities, and an SGD-with-momentum update over an f64 parameter vector.

use crate::errors::EvolutionError;

use super::noise::fill_perturbation;

/// Configuration for [`EsOptimizer`].
#[derive(Debug, Clone)]
pub struct EsConfig {
    /// Number of mirrored perturbation pairs; the population holds
    /// `2 * pairs` candidates.
    pub pairs: usize,
    /// Perturbation scale σ (must be positive and finite).
    pub sigma: f64,
    /// Learning rate for the θ update (must be positive and finite).
    pub learning_rate: f64,
    /// Momentum coefficient in `[0, 1)`; `0.0` disables momentum.
    pub momentum: f64,
    /// Master seed; together with the step counter and pair index it fully
    /// determines every perturbation.
    pub seed: u64,
}

impl Default for EsConfig {
    fn default() -> Self {
        EsConfig {
            pairs: 16,
            sigma: 0.1,
            learning_rate: 0.05,
            momentum: 0.9,
            seed: 0,
        }
    }
}

impl EsConfig {
    fn validate(&self) -> Result<(), EvolutionError> {
        if self.pairs == 0 {
            return Err(EvolutionError::InvalidConfig {
                message: "pairs must be at least 1".to_string(),
            });
        }
        if !self.sigma.is_finite() || self.sigma <= 0.0 {
            return Err(EvolutionError::InvalidConfig {
                message: format!("sigma must be positive and finite, got {}", self.sigma),
            });
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(EvolutionError::InvalidConfig {
                message: format!(
                    "learning_rate must be positive and finite, got {}",
                    self.learning_rate
                ),
            });
        }
        if !self.momentum.is_finite() || !(0.0..1.0).contains(&self.momentum) {
            return Err(EvolutionError::InvalidConfig {
                message: format!("momentum must be in [0, 1), got {}", self.momentum),
            });
        }
        Ok(())
    }
}

/// OpenAI-ES optimizer with mirrored sampling and centered-rank fitness
/// shaping.
///
/// One step is an `ask`/`tell` round trip:
///
/// - [`ask`](Self::ask) fills the population buffer with `2 * pairs`
///   candidates, pair-adjacent (`2i` = `θ + σ·ε_i`, `2i + 1` = `θ − σ·ε_i`),
///   deterministically from `(seed, step, i)`.
/// - [`tell`](Self::tell) takes one fitness per candidate (higher is
///   better), rank-shapes all `2 * pairs` values jointly to centered
///   utilities in `[-0.5, 0.5]`, estimates the gradient
///   `Σ (util₂ᵢ − util₂ᵢ₊₁)·ε_i / (2N·σ)`, and applies SGD with momentum.
///
/// θ and velocity are kept in f64; every buffer is preallocated in
/// [`new`](Self::new), so `ask`/`tell` are allocation-free.
///
/// # Example
///
/// ```
/// use instmodel_inference::evolution::{EsConfig, EsOptimizer};
///
/// let config = EsConfig {
///     pairs: 8,
///     sigma: 0.1,
///     learning_rate: 0.1,
///     momentum: 0.0,
///     seed: 7,
/// };
/// let mut optimizer = EsOptimizer::new(vec![1.0, -1.0], config).unwrap();
/// for _ in 0..60 {
///     let population = optimizer.ask().unwrap().to_vec();
///     let fitness: Vec<f64> = population
///         .chunks(optimizer.theta_len())
///         .map(|candidate| -candidate.iter().map(|v| v * v).sum::<f64>())
///         .collect();
///     optimizer.tell(&fitness).unwrap();
/// }
/// let distance: f64 = optimizer.theta().iter().map(|v| v * v).sum();
/// assert!(distance < 0.5, "sphere fitness did not improve: {distance}");
/// ```
#[derive(Debug, Clone)]
pub struct EsOptimizer {
    pairs: usize,
    sigma: f64,
    learning_rate: f64,
    momentum: f64,
    seed: u64,
    theta: Vec<f64>,
    velocity: Vec<f64>,
    population: Vec<f64>,
    eps: Vec<f64>,
    grad: Vec<f64>,
    rank_idx: Vec<usize>,
    util: Vec<f64>,
    step: usize,
    sigma_used: f64,
    awaiting_tell: bool,
}

impl EsOptimizer {
    /// Creates an optimizer starting at `theta_init`.
    pub fn new(theta_init: Vec<f64>, config: EsConfig) -> Result<Self, EvolutionError> {
        config.validate()?;
        if theta_init.is_empty() {
            return Err(EvolutionError::InvalidConfig {
                message: "theta_init must not be empty".to_string(),
            });
        }
        if let Some(bad) = theta_init.iter().position(|v| !v.is_finite()) {
            return Err(EvolutionError::InvalidConfig {
                message: format!("theta_init[{bad}] is not finite: {}", theta_init[bad]),
            });
        }

        let p = theta_init.len();
        let two_n = 2 * config.pairs;
        Ok(EsOptimizer {
            pairs: config.pairs,
            sigma: config.sigma,
            learning_rate: config.learning_rate,
            momentum: config.momentum,
            seed: config.seed,
            velocity: vec![0.0; p],
            population: vec![0.0; two_n * p],
            eps: vec![0.0; config.pairs * p],
            grad: vec![0.0; p],
            rank_idx: vec![0; two_n],
            util: vec![0.0; two_n],
            step: 0,
            sigma_used: config.sigma,
            awaiting_tell: false,
            theta: theta_init,
        })
    }

    /// Generates the next population and returns it as one flat buffer of
    /// `population_size() * theta_len()` values, pair-adjacent
    /// (`2i` = `θ + σ·ε_i`, `2i + 1` = `θ − σ·ε_i`).
    ///
    /// The σ in effect now is captured for the matching
    /// [`tell`](Self::tell), so schedules may adjust it between steps
    /// without corrupting an in-flight update.
    pub fn ask(&mut self) -> Result<&[f64], EvolutionError> {
        if self.awaiting_tell {
            return Err(EvolutionError::CallOrder {
                message: "ask() called again before tell() consumed the previous population"
                    .to_string(),
            });
        }
        let p = self.theta.len();
        self.sigma_used = self.sigma;
        for i in 0..self.pairs {
            let eps = &mut self.eps[i * p..(i + 1) * p];
            fill_perturbation(self.seed, self.step, i, eps);
            let pair = &mut self.population[2 * i * p..(2 * i + 2) * p];
            let (plus, minus) = pair.split_at_mut(p);
            for k in 0..p {
                let delta = self.sigma_used * eps[k];
                plus[k] = self.theta[k] + delta;
                minus[k] = self.theta[k] - delta;
            }
        }
        self.awaiting_tell = true;
        Ok(&self.population)
    }

    /// Consumes one fitness per candidate (higher is better, aligned with
    /// the last [`ask`](Self::ask) population) and updates θ.
    pub fn tell(&mut self, fitness: &[f64]) -> Result<(), EvolutionError> {
        if !self.awaiting_tell {
            return Err(EvolutionError::CallOrder {
                message: "tell() called without a pending ask() population".to_string(),
            });
        }
        let two_n = 2 * self.pairs;
        if fitness.len() != two_n {
            return Err(EvolutionError::FitnessLengthMismatch {
                expected: two_n,
                got: fitness.len(),
            });
        }
        if let Some(bad) = fitness.iter().position(|v| !v.is_finite()) {
            return Err(EvolutionError::NonFiniteFitness {
                index: bad,
                value: fitness[bad],
            });
        }

        rank_utilities_into(fitness, &mut self.rank_idx, &mut self.util);

        let p = self.theta.len();
        self.grad.fill(0.0);
        for i in 0..self.pairs {
            let coeff = self.util[2 * i] - self.util[2 * i + 1];
            let eps = &self.eps[i * p..(i + 1) * p];
            for (grad, &eps_k) in self.grad.iter_mut().zip(eps.iter()) {
                *grad += coeff * eps_k;
            }
        }
        let scale = 1.0 / (two_n as f64 * self.sigma_used);
        for k in 0..p {
            let g = self.grad[k] * scale;
            self.velocity[k] = self.momentum * self.velocity[k] + g;
            self.theta[k] += self.learning_rate * self.velocity[k];
        }

        self.step += 1;
        self.awaiting_tell = false;
        Ok(())
    }

    /// Current parameter vector.
    pub fn theta(&self) -> &[f64] {
        &self.theta
    }

    /// Current momentum velocity.
    pub fn velocity(&self) -> &[f64] {
        &self.velocity
    }

    /// Number of completed `ask`/`tell` steps.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Parameter count per candidate.
    pub fn theta_len(&self) -> usize {
        self.theta.len()
    }

    /// Number of candidates per population (`2 * pairs`).
    pub fn population_size(&self) -> usize {
        2 * self.pairs
    }

    /// The population buffer from the most recent [`ask`](Self::ask)
    /// (all zeros before the first call).
    pub fn population(&self) -> &[f64] {
        &self.population
    }

    /// One candidate of the most recent population.
    pub fn candidate(&self, index: usize) -> Result<&[f64], EvolutionError> {
        if index >= self.population_size() {
            return Err(EvolutionError::CandidateOutOfBounds {
                index,
                count: self.population_size(),
            });
        }
        let p = self.theta.len();
        Ok(&self.population[index * p..(index + 1) * p])
    }

    /// Replaces σ for subsequent [`ask`](Self::ask) calls (an in-flight
    /// step keeps the σ captured at its `ask`).
    pub fn set_sigma(&mut self, sigma: f64) -> Result<(), EvolutionError> {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(EvolutionError::InvalidConfig {
                message: format!("sigma must be positive and finite, got {sigma}"),
            });
        }
        self.sigma = sigma;
        Ok(())
    }

    /// Replaces the learning rate for subsequent updates.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<(), EvolutionError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(EvolutionError::InvalidConfig {
                message: format!("learning_rate must be positive and finite, got {learning_rate}"),
            });
        }
        self.learning_rate = learning_rate;
        Ok(())
    }

    /// Writes θ narrowed to f32 into `out` (cleared first) — the shape
    /// consumed by
    /// [`InstructionModel::apply_theta`](crate::InstructionModel::apply_theta)
    /// and
    /// [`PopulationPack::write_candidate`](crate::gpu::PopulationPack::write_candidate).
    pub fn theta_f32_into(&self, out: &mut Vec<f32>) {
        out.clear();
        out.extend(self.theta.iter().map(|&v| v as f32));
    }

    /// Writes candidate `index` of the most recent population narrowed to
    /// f32 into `out` (cleared first).
    pub fn candidate_f32_into(
        &self,
        index: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), EvolutionError> {
        let candidate = self.candidate(index)?;
        out.clear();
        out.extend(candidate.iter().map(|&v| v as f32));
        Ok(())
    }
}

/// Centered rank utilities in `[-0.5, 0.5]` (higher fitness → higher
/// utility), written into caller-owned `util`; `idx` is sorted-index
/// scratch. Allocation-free.
fn rank_utilities_into(fitness: &[f64], idx: &mut [usize], util: &mut [f64]) {
    let m = fitness.len();
    for (i, slot) in idx.iter_mut().enumerate() {
        *slot = i;
    }
    idx.sort_unstable_by(|&a, &b| fitness[a].total_cmp(&fitness[b]));
    let denom = (m as f64 - 1.0).max(1.0);
    for (rank, &i) in idx.iter().enumerate() {
        util[i] = rank as f64 / denom - 0.5;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_utilities_are_centered_and_ordered() {
        let fitness = [0.3, -1.0, 2.5, 0.0];
        let mut idx = vec![0; 4];
        let mut util = vec![0.0; 4];
        rank_utilities_into(&fitness, &mut idx, &mut util);

        // Ranks asc: -1.0 < 0.0 < 0.3 < 2.5 → utilities -0.5, -1/6, 1/6, 0.5.
        assert_eq!(util[1], -0.5);
        assert_eq!(util[2], 0.5);
        assert!((util[3] - (1.0 / 3.0 - 0.5)).abs() < 1e-12);
        assert!((util[0] - (2.0 / 3.0 - 0.5)).abs() < 1e-12);
        let sum: f64 = util.iter().sum();
        assert!(sum.abs() < 1e-12, "utilities not centered: {sum}");
    }

    #[test]
    fn config_validation() {
        let theta = vec![0.0; 3];
        let bad_pairs = EsConfig {
            pairs: 0,
            ..EsConfig::default()
        };
        assert!(matches!(
            EsOptimizer::new(theta.clone(), bad_pairs),
            Err(EvolutionError::InvalidConfig { .. })
        ));

        let bad_sigma = EsConfig {
            sigma: 0.0,
            ..EsConfig::default()
        };
        assert!(matches!(
            EsOptimizer::new(theta.clone(), bad_sigma),
            Err(EvolutionError::InvalidConfig { .. })
        ));

        let bad_momentum = EsConfig {
            momentum: 1.0,
            ..EsConfig::default()
        };
        assert!(matches!(
            EsOptimizer::new(theta.clone(), bad_momentum),
            Err(EvolutionError::InvalidConfig { .. })
        ));

        assert!(matches!(
            EsOptimizer::new(vec![], EsConfig::default()),
            Err(EvolutionError::InvalidConfig { .. })
        ));
        assert!(matches!(
            EsOptimizer::new(vec![f64::NAN], EsConfig::default()),
            Err(EvolutionError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn phase_guard_enforces_ask_tell_alternation() {
        let mut optimizer = EsOptimizer::new(vec![0.0], EsConfig::default()).unwrap();
        assert!(matches!(
            optimizer.tell(&vec![0.0; optimizer.population_size()]),
            Err(EvolutionError::CallOrder { .. })
        ));
        optimizer.ask().unwrap();
        assert!(matches!(
            optimizer.ask(),
            Err(EvolutionError::CallOrder { .. })
        ));
        optimizer
            .tell(&vec![0.0; optimizer.population_size()])
            .unwrap();
        optimizer.ask().unwrap();
    }
}
