//! Annealing schedules for σ / learning-rate decay.

/// Cosine annealing factor in `[final_frac, 1]`.
///
/// Returns `1.0` at `step == 0`, decaying along a half cosine to
/// `final_frac` at `step == total_steps - 1` (clamped there for any later
/// step). With `total_steps <= 1` there is nothing to anneal and the factor
/// is constant `1.0`.
///
/// Typical use: `optimizer.set_sigma(base_sigma * cosine_anneal(step, total_steps, 0.1))`.
pub fn cosine_anneal(step: usize, total_steps: usize, final_frac: f64) -> f64 {
    if total_steps <= 1 {
        return 1.0;
    }
    let t = (step as f64 / (total_steps as f64 - 1.0)).min(1.0);
    final_frac + (1.0 - final_frac) * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints() {
        assert_eq!(cosine_anneal(0, 100, 0.1), 1.0);
        let last = cosine_anneal(99, 100, 0.1);
        assert!((last - 0.1).abs() < 1e-12);
    }

    #[test]
    fn clamps_past_the_last_step() {
        let beyond = cosine_anneal(500, 100, 0.25);
        assert!((beyond - 0.25).abs() < 1e-12);
    }

    #[test]
    fn monotonically_decreasing() {
        let mut previous = f64::INFINITY;
        for step in 0..50 {
            let value = cosine_anneal(step, 50, 0.05);
            assert!(value <= previous, "not decreasing at step {step}");
            previous = value;
        }
    }

    #[test]
    fn degenerate_schedules_stay_constant() {
        assert_eq!(cosine_anneal(0, 0, 0.1), 1.0);
        assert_eq!(cosine_anneal(0, 1, 0.1), 1.0);
        assert_eq!(cosine_anneal(5, 1, 0.1), 1.0);
    }
}
