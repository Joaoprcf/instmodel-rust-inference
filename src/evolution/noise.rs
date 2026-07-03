//! Deterministic counter-based Gaussian noise for ES perturbations.
//!
//! The stream is a pure function of its seed — a `splitmix64` integer
//! generator feeding Box–Muller (one spare cached per pair) — so the noise,
//! and therefore a whole ES update, never depends on thread scheduling or
//! iteration order.

/// `splitmix64` finalizer, used to mix `(master_seed, step, index)` into a
/// well-distributed stream seed.
#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Per-perturbation noise seed: a pure function of
/// `(master_seed, step, index)`, so perturbation `index` of step `step` can
/// be regenerated anywhere — any thread, any machine — with identical bits.
#[inline]
pub fn perturbation_seed(master_seed: u64, step: usize, index: usize) -> u64 {
    let step_mix = (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let index_mix = (index as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    splitmix64(master_seed ^ step_mix ^ index_mix)
}

/// Deterministic standard-normal stream: `splitmix64` counter feeding
/// Box–Muller with the antithetic spare cached between draws.
#[derive(Debug, Clone)]
pub struct GaussianStream {
    state: u64,
    spare: f64,
    has_spare: bool,
}

impl GaussianStream {
    /// Creates a stream; identical seeds yield identical sample sequences.
    pub fn new(seed: u64) -> Self {
        GaussianStream {
            state: seed,
            spare: 0.0,
            has_spare: false,
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]` from the top 53 bits (full f64 mantissa), so the
    /// Box–Muller logarithm never sees zero.
    #[inline]
    fn next_open01(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) * (1.0 / 9_007_199_254_740_992.0) // 2^-53
    }

    /// Next standard-normal sample.
    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        if self.has_spare {
            self.has_spare = false;
            return self.spare;
        }
        let u1 = self.next_open01();
        let u2 = self.next_open01();
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
        self.spare = r * s;
        self.has_spare = true;
        r * c
    }

    /// Fills `out` with standard-normal samples.
    pub fn fill_gaussian_f64(&mut self, out: &mut [f64]) {
        for value in out.iter_mut() {
            *value = self.next_normal();
        }
    }

    /// Fills `out` with standard-normal samples narrowed to f32; the sample
    /// sequence is the f64 stream, so f32 fills stay consistent with f64
    /// fills from the same seed.
    pub fn fill_gaussian_f32(&mut self, out: &mut [f32]) {
        for value in out.iter_mut() {
            *value = self.next_normal() as f32;
        }
    }
}

/// Fills `eps` with the deterministic noise for perturbation
/// `(master_seed, step, pair_index)`.
pub fn fill_perturbation(master_seed: u64, step: usize, pair_index: usize, eps: &mut [f64]) {
    let mut stream = GaussianStream::new(perturbation_seed(master_seed, step, pair_index));
    stream.fill_gaussian_f64(eps);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_coordinates_reproduce_identical_noise() {
        let mut a = vec![0.0; 64];
        let mut b = vec![0.0; 64];
        fill_perturbation(42, 3, 7, &mut a);
        fill_perturbation(42, 3, 7, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn any_coordinate_change_changes_the_noise() {
        let mut base = vec![0.0; 64];
        let mut other_index = vec![0.0; 64];
        let mut other_step = vec![0.0; 64];
        let mut other_seed = vec![0.0; 64];
        fill_perturbation(42, 3, 7, &mut base);
        fill_perturbation(42, 3, 8, &mut other_index);
        fill_perturbation(42, 4, 7, &mut other_step);
        fill_perturbation(43, 3, 7, &mut other_seed);
        assert_ne!(base, other_index);
        assert_ne!(base, other_step);
        assert_ne!(base, other_seed);
    }

    #[test]
    fn samples_have_standard_normal_moments() {
        let mut stream = GaussianStream::new(1234);
        let n = 100_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let x = stream.next_normal();
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean} too far from 0");
        assert!((var - 1.0).abs() < 0.03, "variance {var} too far from 1");
    }

    #[test]
    fn f32_fill_matches_narrowed_f64_fill() {
        let mut wide = vec![0.0f64; 32];
        let mut narrow = vec![0.0f32; 32];
        GaussianStream::new(9).fill_gaussian_f64(&mut wide);
        GaussianStream::new(9).fill_gaussian_f32(&mut narrow);
        let expected: Vec<f32> = wide.iter().map(|&v| v as f32).collect();
        assert_eq!(narrow, expected);
    }
}
