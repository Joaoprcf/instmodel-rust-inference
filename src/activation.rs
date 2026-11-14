//! Activation functions.

use serde::{Deserialize, Serialize};

/// Element-wise (or, for `Softmax`, vector-wise) activation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Activation {
    /// `max(0, x)`.
    Relu,
    /// `1 / (1 + exp(-x))`.
    Sigmoid,
    /// `exp(x_i - max(x)) / sum_j exp(x_j - max(x))` over the whole buffer.
    Softmax,
    /// `sqrt(x)` for `x > 0`, else 0.
    Sqrt,
    /// `ln(x + 1)` for `x > 0`, else 0.
    Log,
    /// `log10(x + 1)` for `x > 0`, else 0.
    Log10,
    /// `tanh(x)`.
    Tanh,
    /// `1 - x`.
    Inverse,
    /// Exact-erf GELU: `x * 0.5 * (1 + erf(x / sqrt(2)))` (TensorFlow default).
    Gelu,
    /// `ln(1 + exp(x))` with TensorFlow's thresholds.
    Softplus,
    /// `exp(clamp(x, -88, 88))`, keeping the result finite in f32.
    Exp,
    /// numpy `sign`: -1, +1, or `x` itself for ±0 and NaN.
    Sign,
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

/// Abramowitz & Stegun 7.1.26 (max error ~1.5e-7).
#[inline(always)]
fn erf(x: f32) -> f32 {
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_72;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.327_591_1;
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

#[inline(always)]
fn gelu_activation(x: f32) -> f32 {
    x * 0.5 * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

#[inline(always)]
fn softplus_activation(x: f32) -> f32 {
    // TensorFlow's threshold: ln(f32::EPSILON) + 2.
    const THRESHOLD: f32 = -13.9424;
    if x > -THRESHOLD {
        x
    } else if x < THRESHOLD {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

#[inline(always)]
fn exp_activation(x: f32) -> f32 {
    x.clamp(-88.0, 88.0).exp()
}

#[inline(always)]
fn sign_activation(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

#[inline(always)]
fn map(values: &mut [f32], f: impl Fn(f32) -> f32) {
    for v in values {
        *v = f(*v);
    }
}

impl Activation {
    /// Parses the JSON name (e.g. `"RELU"`).
    pub fn get_by_name(type_name: &str) -> Option<Self> {
        Some(match type_name {
            "RELU" => Activation::Relu,
            "SIGMOID" => Activation::Sigmoid,
            "SOFTMAX" => Activation::Softmax,
            "SQRT" => Activation::Sqrt,
            "LOG" => Activation::Log,
            "LOG10" => Activation::Log10,
            "TANH" => Activation::Tanh,
            "INVERSE" => Activation::Inverse,
            "GELU" => Activation::Gelu,
            "SOFTPLUS" => Activation::Softplus,
            "EXP" => Activation::Exp,
            "SIGN" => Activation::Sign,
            _ => return None,
        })
    }

    /// Applies the activation to one value (`Softmax` degenerates to `exp(x)`).
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
            Activation::Softplus => softplus_activation(x),
            Activation::Exp => exp_activation(x),
            Activation::Sign => sign_activation(x),
            Activation::Softmax => x.exp(),
        }
    }

    /// Applies the activation to `values` in place.
    pub fn apply_in_place(self, values: &mut [f32]) {
        match self {
            Activation::Softmax => {
                let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0f32;
                for val in values.iter_mut() {
                    *val = (*val - max_val).exp();
                    sum += *val;
                }
                for val in values.iter_mut() {
                    *val /= sum;
                }
            }
            // One arm per variant so each loop monomorphizes (and vectorizes) separately.
            Activation::Relu => map(values, relu),
            Activation::Sigmoid => map(values, sigmoid),
            Activation::Sqrt => map(values, sqrt_activation),
            Activation::Log => map(values, log_activation),
            Activation::Log10 => map(values, log10_activation),
            Activation::Tanh => map(values, tanh_activation),
            Activation::Inverse => map(values, inverse_activation),
            Activation::Gelu => map(values, gelu_activation),
            Activation::Softplus => map(values, softplus_activation),
            Activation::Exp => map(values, exp_activation),
            Activation::Sign => map(values, sign_activation),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Activation::*;

    #[test]
    fn element_wise_values() {
        let cases: &[(Activation, f32, f32, f32)] = &[
            (Relu, 1.0, 1.0, 5e-5),
            (Relu, -1.0, 0.0, 5e-5),
            (Sigmoid, 1.0, 0.7311, 5e-5),
            (Sigmoid, 0.0, 0.5, 5e-5),
            (Sigmoid, -0.5, 0.3775, 5e-5),
            (Sqrt, 4.0, 2.0, 5e-5),
            (Sqrt, -1.0, 0.0, 5e-5),
            (Log, 1.0, std::f32::consts::LN_2, 5e-5),
            (Log, 0.0, 0.0, 5e-5),
            (Log10, 99.0, 2.0, 5e-5),
            (Log10, 0.0, 0.0, 5e-5),
            (Tanh, -1.0, -0.761_594_2, 5e-5),
            (Inverse, -1.0, 2.0, 5e-5),
            (Gelu, -2.0, -0.0454, 1e-3),
            (Gelu, -1.0, -0.1587, 1e-3),
            (Gelu, 0.0, 0.0, 5e-5),
            (Gelu, 1.0, 0.8413, 1e-3),
            (Gelu, 2.0, 1.9545, 1e-3),
            (Softplus, 0.0, std::f32::consts::LN_2, 1e-7),
            (Softplus, 1.0, 1.313_261_6, 1e-7),
            (Softplus, -1.0, 0.313_261_7, 1e-7),
            (Softplus, 100.0, 100.0, 1e-7),
            (Softplus, -100.0, 0.0, 1e-7),
            (Exp, 1.0, std::f32::consts::E, 5e-5),
            (Exp, -1000.0, 0.0, 5e-5),
            (Sign, 3.5, 1.0, 0.0),
            (Sign, -2.0, -1.0, 0.0),
            (Sign, 0.0, 0.0, 0.0),
        ];
        for &(activation, x, expected, tol) in cases {
            let single = activation.apply_single(x);
            let mut in_place = [x];
            activation.apply_in_place(&mut in_place);
            assert!(
                (single - expected).abs() <= tol,
                "{activation:?}({x}) = {single}"
            );
            assert_eq!(
                single.to_bits(),
                in_place[0].to_bits(),
                "{activation:?}({x})"
            );
        }
        assert!((Exp.apply_single(1000.0) - 88.0_f32.exp()).abs() < 1.0);
        assert!(Sign.apply_single(f32::NAN).is_nan());
    }

    #[test]
    fn softmax_normalizes_vector() {
        let mut values = [1.0, 2.0, 3.0];
        Softmax.apply_in_place(&mut values);
        for (v, e) in values
            .iter()
            .zip([0.090_030_57, 0.244_728_47, 0.665_240_96])
        {
            assert!((v - e).abs() < 5e-5);
        }
    }

    #[test]
    fn names_round_trip() {
        for name in [
            "RELU", "SIGMOID", "SOFTMAX", "SQRT", "LOG", "LOG10", "TANH", "INVERSE", "GELU",
            "SOFTPLUS", "EXP", "SIGN",
        ] {
            let activation = Activation::get_by_name(name).unwrap();
            assert_eq!(serde_json::to_value(activation).unwrap(), name);
        }
        assert_eq!(Activation::get_by_name("INVALID"), None);
    }
}
