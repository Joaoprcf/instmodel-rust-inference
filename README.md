# instmodel-rust-inference

A high-performance neural network inference library for Rust that executes optimized computation sequences through a unified buffer architecture.

## Installation

```bash
cargo add instmodel-rust-inference
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
instmodel-rust-inference = "<version>"
```

## Overview

This library provides a lightweight, zero-dependency neural network inference engine. Models are defined as a sequence of instructions that operate on computation buffers, enabling efficient memory reuse and predictable performance.

**Key Features:**
- Instruction-based execution model for neural network inference
- Support for common neural network operations (dot product, activations, attention, etc.)
- JSON serialization/deserialization for model configuration
- Built-in model validation
- Memory-efficient unified buffer architecture

## Quick Start

### Simple Neural Network

```rust
use instmodel_rust_inference::{
    InstructionModel, InstructionModelInfo, Activation,
    instruction_model_info::{InstructionInfo, DotInstructionInfo},
};

// Define a simple single-layer neural network
// Input: 2 features -> Output: 1 value
let model_info = InstructionModelInfo {
    features: Some(vec!["feature1".to_string(), "feature2".to_string()]),
    feature_size: None,
    computation_buffer_sizes: vec![2, 1],  // input buffer: 2, output buffer: 1
    instructions: vec![
        InstructionInfo::Dot(DotInstructionInfo {
            input: 0,      // read from buffer 0
            output: 1,     // write to buffer 1
            weights: 0,    // use weights at index 0
            activation: Some(Activation::Sigmoid),
        })
    ],
    weights: vec![vec![vec![0.5, -0.3]]],  // shape: [1, 2]
    bias: vec![vec![0.1]],                  // shape: [1]
    parameters: None,
    maps: None,
    validation_data: None,
};

let model = InstructionModel::new(model_info)?;

// Run inference
let input = vec![1.0, 0.5];
let output = model.predict(&input)?;
println!("Prediction: {}", output[0]);

// Or get a single output value directly
let result = model.predict_single(&input)?;
```

### Multi-Layer Neural Network

```rust
use instmodel_rust_inference::{
    InstructionModel, InstructionModelInfo, Activation,
    instruction_model_info::{InstructionInfo, DotInstructionInfo},
};

// 2 inputs -> 2 hidden (ReLU) -> 1 output (Sigmoid)
let model_info = InstructionModelInfo {
    features: Some(vec!["x1".to_string(), "x2".to_string()]),
    feature_size: None,
    computation_buffer_sizes: vec![2, 2, 1],
    instructions: vec![
        // Hidden layer with ReLU
        InstructionInfo::Dot(DotInstructionInfo {
            input: 0,
            output: 1,
            weights: 0,
            activation: Some(Activation::Relu),
        }),
        // Output layer with Sigmoid
        InstructionInfo::Dot(DotInstructionInfo {
            input: 1,
            output: 2,
            weights: 1,
            activation: Some(Activation::Sigmoid),
        }),
    ],
    weights: vec![
        // Hidden layer weights [2, 2]
        vec![vec![2.0, 0.5], vec![-2.0, -0.5]],
        // Output layer weights [1, 2]
        vec![vec![0.5, -1.0]],
    ],
    bias: vec![
        vec![0.25, -0.25],  // Hidden layer bias
        vec![2.0],           // Output layer bias
    ],
    parameters: None,
    maps: None,
    validation_data: None,
};

let model = InstructionModel::new(model_info)?;
let result = model.predict_single(&[1.0, -1.0])?;
```

### Loading from JSON

Models can be defined in JSON format and loaded at runtime:

```rust
use instmodel_rust_inference::{InstructionModel, InstructionModelInfo};

let json_config = r#"
{
  "features": ["feature1", "feature2"],
  "buffer_sizes": [2, 2, 1],
  "instructions": [
    {
      "type": "DOT",
      "input": 0,
      "output": 1,
      "weights": 0,
      "activation": "RELU"
    },
    {
      "type": "DOT",
      "input": 1,
      "output": 2,
      "weights": 1,
      "activation": "SIGMOID"
    }
  ],
  "weights": [
    [[2.0, 0.5], [-2.0, -0.5]],
    [[0.5, -1.0]]
  ],
  "bias": [
    [0.25, -0.25],
    [2.0]
  ]
}
"#;

let model_info: InstructionModelInfo = serde_json::from_str(json_config)?;
let model = InstructionModel::new(model_info)?;
```

### Logistic Regression

Create a logistic regression model directly from coefficients:

```rust
use instmodel_rust_inference::{InstructionModel, InstructionModelInfo};
use std::collections::HashMap;

let mut coefficients = HashMap::new();
coefficients.insert("age".to_string(), 0.05);
coefficients.insert("income".to_string(), 0.001);
coefficients.insert("constant".to_string(), -2.5);  // bias term

let model_info = InstructionModelInfo::from_logistic_regression_model(
    coefficients,
    Some(vec!["age".to_string(), "income".to_string()]),  // feature order
)?;

let model = InstructionModel::new(model_info)?;
let probability = model.predict_single(&[35.0, 50000.0])?;
```

### Using the Builder Pattern

```rust
use instmodel_rust_inference::{
    InstructionModelInfo, InstructionModel,
    instruction_model_info::{InstructionInfo, DotInstructionInfo},
};

let model_info = InstructionModelInfo::builder()
    .feature_size(2)
    .computation_buffer_sizes(vec![2, 1])
    .instructions(vec![
        InstructionInfo::Dot(DotInstructionInfo {
            input: 0,
            output: 1,
            weights: 0,
            activation: None,
        })
    ])
    .weights(vec![vec![vec![1.0, 1.0]]])
    .bias(vec![vec![0.0]])
    .build()?;

let model = InstructionModel::new(model_info)?;
```

## Supported Operations

### Activation Functions

| Activation | Description |
|------------|-------------|
| `Relu` | f(x) = max(0, x) |
| `Sigmoid` | f(x) = 1 / (1 + exp(-x)) |
| `Softmax` | Numerically stable softmax over a buffer |
| `Tanh` | f(x) = tanh(x) |
| `Sqrt` | f(x) = sqrt(x) for x > 0, else 0 |
| `Log` | f(x) = ln(x + 1) for x > 0, else 0 |
| `Log10` | f(x) = log10(x + 1) for x > 0, else 0 |
| `Inverse` | f(x) = 1 - x |

### Instruction Types

| Instruction | JSON Type | Description |
|-------------|-----------|-------------|
| Dot Product | `DOT` | Matrix multiplication with optional activation |
| Copy | `COPY` | Copy buffer contents to another location |
| Copy Masked | `COPY_MASKED` | Copy specific indices from a buffer |
| Activation | `ACTIVATION` | Apply activation function in-place |
| Element-wise Add | `ADD_ELEMENTWISE` | Add parameters element-wise |
| Element-wise Multiply | `MUL_ELEMENTWISE` | Multiply by parameters element-wise |
| Buffers Add | `ADD_ELEMENTWISE_BUFFERS` | Sum multiple buffers |
| Buffers Multiply | `MULTIPLY_ELEMENTWISE_BUFFERS` | Multiply multiple buffers element-wise |
| Reduce Sum | `REDUCE_SUM` | Sum all values in a buffer to a single value |
| Attention | `ATTENTION` | Attention mechanism (linear + softmax + element-wise) |
| Map Transform | `MAP_TRANSFORM` | Lookup and transform using a map |

## Advanced Usage

### External Buffer Management

For high-performance scenarios, you can manage the computation buffer yourself:

```rust
let model = InstructionModel::new(model_info)?;

// Allocate buffer once
let mut buffer = vec![0.0f32; model.required_memory()];

// Reuse buffer for multiple predictions
for input in inputs {
    // Copy input to buffer
    buffer[..input.len()].copy_from_slice(&input);

    // Run inference
    model.predict_with_buffer(&mut buffer)?;

    // Read output
    let output = model.get_output(&buffer, 0);
}
```

### Model Validation

Include validation data to verify model correctness on creation:

```rust
use instmodel_rust_inference::instruction_model_info::ValidationData;

let model_info = InstructionModelInfo {
    // ... model configuration ...
    validation_data: Some(ValidationData {
        inputs: vec![
            vec![1.0, -1.0],
            vec![-1.0, 1.0],
        ],
        expected_outputs: vec![
            vec![0.9466],
            vec![0.8808],
        ],
    }),
    // ...
};

// Model creation will fail if outputs don't match expected values
let model = InstructionModel::new(model_info)?;
```

### Array Features

Features can specify array sizes using bracket notation:

```rust
let model_info = InstructionModelInfo {
    features: Some(vec![
        "scalar_feature".to_string(),    // size: 1
        "embedding[64]".to_string(),     // size: 64
        "another_scalar".to_string(),    // size: 1
    ]),
    // Total feature size: 1 + 64 + 1 = 66
    computation_buffer_sizes: vec![66, 32, 1],
    // ...
};
```

## Architecture

The library uses a unified buffer architecture where all computation buffers are laid out contiguously in memory. Instructions read from and write to specific regions of this buffer:

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Buffer 0   │  Buffer 1   │  Buffer 2   │  Buffer 3   │
│  (Input)    │  (Hidden)   │  (Hidden)   │  (Output)   │
└─────────────┴─────────────┴─────────────┴─────────────┘
              ▲             │             │
              └─────────────┴─────────────┘
                    Instructions operate on
                    buffer regions by index
```

## License

MIT