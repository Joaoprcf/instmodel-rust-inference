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
    /// Gaussian Error Linear Unit: f(x) = x * 0.5 * (1 + erf(x / sqrt(2))).
    /// Uses the exact erf formulation matching TensorFlow's default.
    Gelu,
}

#[inline(always)]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn sqrt_activation(x: f32) -> f32 {
    if x > 0.0 { x.sqrt() } else { 0.0 }
}

#[inline(always)]
fn log_activation(x: f32) -> f32 {
    if x > 0.0 { (x + 1.0).ln() } else { 0.0 }
}

#[inline(always)]
fn log10_activation(x: f32) -> f32 {
    if x > 0.0 { (x + 1.0).log10() } else { 0.0 }
}

#[inline(always)]
fn tanh_activation(x: f32) -> f32 {
    x.tanh()
}

#[inline(always)]
fn inverse_activation(x: f32) -> f32 {
    1.0 - x
}

/// Compute the error function using the Abramowitz and Stegun approximation.
/// Maximum error: ~1.5×10^-7
#[inline(always)]
fn erf(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    // Abramowitz and Stegun formula 7.1.26
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_72;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.327_591_1;

    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();

    sign * y
}

/// GeLU activation: f(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
#[inline(always)]
fn gelu_activation(x: f32) -> f32 {
    const SQRT_2_INV: f32 = std::f32::consts::FRAC_1_SQRT_2; // 1 / sqrt(2)
    x * 0.5 * (1.0 + erf(x * SQRT_2_INV))
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
            ("GELU", Activation::Gelu),
        ]
        .iter()
        .cloned()
        .collect();

        map.get(type_name).copied()
    }

    /// Apply the activation function to a single value.
    pub fn apply_single(self, x: f32) -> f32 {
        match self {
            Activation::Relu => relu(x),
            Activation::Sigmoid => sigmoid(x),
            Activation::Sqrt => sqrt_activation(x),
            Activation::Log => log_activation(x),
            Activation::Log10 => log10_activation(x),
            Activation::Tanh => tanh_activation(x),
            Activation::Inverse => inverse_activation(x),
            Activation::Gelu => gelu_activation(x),
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
            Activation::Relu => {
                for val in values.iter_mut() {
                    *val = relu(*val);
                }
            }
            Activation::Sigmoid => {
                for val in values.iter_mut() {
                    *val = sigmoid(*val);
                }
            }
            Activation::Sqrt => {
                for val in values.iter_mut() {
                    *val = sqrt_activation(*val);
                }
            }
            Activation::Log => {
                for val in values.iter_mut() {
                    *val = log_activation(*val);
                }
            }
            Activation::Log10 => {
                for val in values.iter_mut() {
                    *val = log10_activation(*val);
                }
            }
            Activation::Tanh => {
                for val in values.iter_mut() {
                    *val = tanh_activation(*val);
                }
            }
            Activation::Inverse => {
                for val in values.iter_mut() {
                    *val = inverse_activation(*val);
                }
            }
            Activation::Gelu => {
                for val in values.iter_mut() {
                    *val = gelu_activation(*val);
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
    fn test_gelu() {
        // GeLU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        // Expected values computed from TensorFlow/NumPy with higher precision
        const GELU_DELTA: f32 = 0.001;
        assert!((Activation::Gelu.apply_single(-2.0) - (-0.0454)).abs() < GELU_DELTA);
        assert!((Activation::Gelu.apply_single(-1.0) - (-0.1587)).abs() < GELU_DELTA);
        assert!((Activation::Gelu.apply_single(0.0) - 0.0).abs() < DELTA);
        assert!((Activation::Gelu.apply_single(1.0) - 0.8413).abs() < GELU_DELTA);
        assert!((Activation::Gelu.apply_single(2.0) - 1.9545).abs() < GELU_DELTA);
    }

    #[test]
    fn test_gelu_in_place() {
        let mut values = [-2.0, -1.0, 0.0, 1.0, 2.0];
        Activation::Gelu.apply_in_place(&mut values);

        const GELU_DELTA: f32 = 0.001;
        assert!((values[0] - (-0.0454)).abs() < GELU_DELTA);
        assert!((values[1] - (-0.1587)).abs() < GELU_DELTA);
        assert!((values[2] - 0.0).abs() < DELTA);
        assert!((values[3] - 0.8413).abs() < GELU_DELTA);
        assert!((values[4] - 1.9545).abs() < GELU_DELTA);
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
        assert_eq!(Activation::get_by_name("GELU"), Some(Activation::Gelu));
        assert_eq!(Activation::get_by_name("INVALID"), None);
    }
}
