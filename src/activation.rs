//! Activation functions for neural network operations.
//!
//! This module provides various activation functions commonly used in neural networks,
//! including ReLU, Sigmoid, Softmax, and others. Each activation function is implemented
//! with numerical stability in mind.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents the type of activation function to be applied.
/// Note: A None value indicates that no activation function should be applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Activation {
    /// Rectified Linear Unit activation function: f(x) = max(0, x).
    Relu,
    /// Sigmoid activation function: f(x) = 1 / (1 + exp(-x)).
    Sigmoid,
    /// Softmax activation function:
    ///
    /// Applies the numerically stable Softmax function across a vector to produce a probability distribution:
    /// ```text
    /// Softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
    /// ```
    ///
    /// This implementation mirrors Keras and TensorFlow's approach to ensure numerical stability by preventing
    /// potential overflow or underflow during exponentiation.
    Softmax,
    /// Square root activation function: f(x) = sqrt(x) for x > 0, 0 for x <= 0.
    Sqrt,
    /// Natural logarithm activation function: f(x) = log(x + 1) for x > 0, 0 for x <= 0.
    Log,
    /// Base-10 logarithm activation function: f(x) = log10(x + 1) for x > 0, 0 for x <= 0.
    Log10,
    /// Hyperbolic tangent activation function: f(x) = tanh(x).
    Tanh,
    /// Inverse activation function: f(x) = 1 - x.
    Inverse,
}

impl Activation {
    /// Get activation by string name.
    pub fn get_by_name(type_name: &str) -> Option<Self> {
        let map: HashMap<&str, Activation> = [
            ("RELU", Activation::Relu),
            ("SIGMOID", Activation::Sigmoid),
            ("SOFTMAX", Activation::Softmax),
            ("SQRT", Activation::Sqrt),
            ("LOG", Activation::Log),
            ("LOG10", Activation::Log10),
            ("TANH", Activation::Tanh),
            ("INVERSE", Activation::Inverse),
        ]
        .iter()
        .cloned()
        .collect();

        map.get(type_name).copied()
    }

    /// Apply the activation function to a single value.
    pub fn apply_single(self, x: f32) -> f32 {
        match self {
            Activation::Relu => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Sqrt => {
                if x > 0.0 {
                    x.sqrt()
                } else {
                    0.0
                }
            }
            Activation::Log => {
                if x > 0.0 {
                    (x + 1.0).ln()
                } else {
                    0.0
                }
            }
            Activation::Log10 => {
                if x > 0.0 {
                    (x + 1.0).log10()
                } else {
                    0.0
                }
            }
            Activation::Tanh => x.tanh(),
            Activation::Inverse => 1.0 - x,
            Activation::Softmax => {
                // Softmax for a single value doesn't make much sense, but we'll return exp(x)
                // The proper softmax should be applied to a vector
                x.exp()
            }
        }
    }

    /// Apply the activation function to a slice of values in place.
    pub fn apply_in_place(self, values: &mut [f32]) {
        match self {
            Activation::Softmax => {
                // Numerically stable softmax implementation
                let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0f32;

                // Compute exp(x_i - max) and accumulate sum
                for val in values.iter_mut() {
                    *val = (*val - max_val).exp();
                    sum += *val;
                }

                // Normalize by sum
                for val in values.iter_mut() {
                    *val /= sum;
                }
            }
            _ => {
                // For other activations, apply element-wise
                for val in values.iter_mut() {
                    *val = self.apply_single(*val);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DELTA: f32 = 0.00005;

    #[test]
    fn test_relu() {
        assert!((Activation::Relu.apply_single(1.0) - 1.0).abs() < DELTA);
        assert!((Activation::Relu.apply_single(-1.0) - 0.0).abs() < DELTA);
        assert!((Activation::Relu.apply_single(0.5) - 0.5).abs() < DELTA);
    }

    #[test]
    fn test_sigmoid() {
        assert!((Activation::Sigmoid.apply_single(1.0) - 0.7311).abs() < DELTA);
        assert!((Activation::Sigmoid.apply_single(0.0) - 0.5).abs() < DELTA);
        assert!((Activation::Sigmoid.apply_single(-0.5) - 0.3775).abs() < DELTA);
    }

    #[test]
    fn test_softmax() {
        let mut values = [1.0, 2.0, 3.0];
        Activation::Softmax.apply_in_place(&mut values);

        // Expected outputs: [0.09003057, 0.24472847, 0.66524096]
        assert!((values[0] - 0.09003057).abs() < DELTA);
        assert!((values[1] - 0.24472847).abs() < DELTA);
        assert!((values[2] - 0.66524096).abs() < DELTA);
    }

    #[test]
    fn test_sqrt() {
        assert!((Activation::Sqrt.apply_single(4.0) - 2.0).abs() < DELTA);
        assert!((Activation::Sqrt.apply_single(-1.0) - 0.0).abs() < DELTA);
        assert!((Activation::Sqrt.apply_single(9.0) - 3.0).abs() < DELTA);
    }

    #[test]
    fn test_log() {
        assert!((Activation::Log.apply_single(1.0) - 2.0_f32.ln()).abs() < DELTA);
        assert!((Activation::Log.apply_single(0.0) - 0.0).abs() < DELTA);
        assert!((Activation::Log.apply_single(9.0) - 10.0_f32.ln()).abs() < DELTA);
    }

    #[test]
    fn test_log10() {
        assert!((Activation::Log10.apply_single(9.0) - 1.0).abs() < DELTA);
        assert!((Activation::Log10.apply_single(0.0) - 0.0).abs() < DELTA);
        assert!((Activation::Log10.apply_single(99.0) - 2.0).abs() < DELTA);
    }

    #[test]
    fn test_tanh() {
        assert!((Activation::Tanh.apply_single(0.0) - 0.0).abs() < DELTA);
        assert!((Activation::Tanh.apply_single(1.0) - 1.0_f32.tanh()).abs() < DELTA);
        assert!((Activation::Tanh.apply_single(-1.0) - (-1.0_f32).tanh()).abs() < DELTA);
    }

    #[test]
    fn test_inverse() {
        assert!((Activation::Inverse.apply_single(1.0) - 0.0).abs() < DELTA);
        assert!((Activation::Inverse.apply_single(0.0) - 1.0).abs() < DELTA);
        assert!((Activation::Inverse.apply_single(-1.0) - 2.0).abs() < DELTA);
    }

    #[test]
    fn test_get_by_name() {
        assert_eq!(Activation::get_by_name("RELU"), Some(Activation::Relu));
        assert_eq!(
            Activation::get_by_name("SIGMOID"),
            Some(Activation::Sigmoid)
        );
        assert_eq!(
            Activation::get_by_name("SOFTMAX"),
            Some(Activation::Softmax)
        );
        assert_eq!(Activation::get_by_name("INVALID"), None);
    }
}
