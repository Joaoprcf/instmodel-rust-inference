// activations.wgsl - All 8 activation functions for GPU neural network inference

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

// Log base 10 constant: 1 / ln(10)
const LOG10_E: f32 = 0.4342944819032518;

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
        default: {
            return x;
        }
    }
}
