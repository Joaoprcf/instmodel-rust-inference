// activations.wgsl - All 9 activation functions for GPU neural network inference

// Activation type constants
const ACTIVATION_NONE: u32 = 0u;
const ACTIVATION_RELU: u32 = 1u;
const ACTIVATION_SIGMOID: u32 = 2u;
const ACTIVATION_SOFTMAX: u32 = 3u;
const ACTIVATION_TANH: u32 = 4u;
const ACTIVATION_SQRT: u32 = 5u;
const ACTIVATION_LOG: u32 = 6u;
const ACTIVATION_LOG10: u32 = 7u;
const ACTIVATION_INVERSE: u32 = 8u;
const ACTIVATION_GELU: u32 = 9u;

// Log base 10 constant: 1 / ln(10)
const LOG10_E: f32 = 0.4342944819032518;

// 1 / sqrt(2) for GeLU
const SQRT_2_INV: f32 = 0.7071067811865476;

fn activation_relu(x: f32) -> f32 {
    return max(x, 0.0);
}

fn activation_sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn activation_tanh(x: f32) -> f32 {
    return tanh(x);
}

fn activation_sqrt(x: f32) -> f32 {
    if x > 0.0 {
        return sqrt(x);
    }
    return 0.0;
}

fn activation_log(x: f32) -> f32 {
    if x > 0.0 {
        return log(x + 1.0);
    }
    return 0.0;
}

fn activation_log10(x: f32) -> f32 {
    if x > 0.0 {
        return log(x + 1.0) * LOG10_E;
    }
    return 0.0;
}

fn activation_inverse(x: f32) -> f32 {
    return 1.0 - x;
}

// Error function approximation using Abramowitz and Stegun formula 7.1.26
// Maximum error: ~1.5×10^-7
fn erf_approx(x: f32) -> f32 {
    let sign = select(-1.0, 1.0, x >= 0.0);
    let abs_x = abs(x);

    // Abramowitz and Stegun constants
    let a1: f32 = 0.254829592;
    let a2: f32 = -0.284496736;
    let a3: f32 = 1.421413741;
    let a4: f32 = -1.453152027;
    let a5: f32 = 1.061405429;
    let p: f32 = 0.3275911;

    let t = 1.0 / (1.0 + p * abs_x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-abs_x * abs_x);

    return sign * y;
}

// GeLU activation: f(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
fn activation_gelu(x: f32) -> f32 {
    return x * 0.5 * (1.0 + erf_approx(x * SQRT_2_INV));
}

// Apply single-value activation (non-softmax)
fn apply_activation_single(x: f32, activation_type: u32) -> f32 {
    switch activation_type {
        case ACTIVATION_RELU: {
            return activation_relu(x);
        }
        case ACTIVATION_SIGMOID: {
            return activation_sigmoid(x);
        }
        case ACTIVATION_TANH: {
            return activation_tanh(x);
        }
        case ACTIVATION_SQRT: {
            return activation_sqrt(x);
        }
        case ACTIVATION_LOG: {
            return activation_log(x);
        }
        case ACTIVATION_LOG10: {
            return activation_log10(x);
        }
        case ACTIVATION_INVERSE: {
            return activation_inverse(x);
        }
        case ACTIVATION_GELU: {
            return activation_gelu(x);
        }
        default: {
            return x;
        }
    }
}
