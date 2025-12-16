# instmodel-rust-inference

A high-performance neural network inference library for Rust that executes optimized computation sequences through a unified buffer architecture.

## Installation

```bash
cargo add instmodel_inference
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
instmodel_inference = "<version>"
```

## Overview

This library provides a lightweight, zero-dependency neural network inference engine. Models are defined as a sequence of instructions that operate on computation buffers, enabling efficient memory reuse and predictable performance.

**Key Features:**

- Instruction-based execution model for neural network inference
- Support for common neural network operations (dot product, activations, attention, etc.)
- JSON serialization/deserialization for model configuration
- Built-in model validation
- Memory-efficient unified buffer architecture

## Benchmarks

These benchmarks measure a simple 2-layer dense network:

- Model: `250 -> 300 -> 200` (`ReLU` then `Sigmoid`)
- Samples: `200,000`
- Threads: `16` (8 physical cores + hyperthreading)
- Warmup: included

### Performance (CPU: AMD Ryzen 9 5900HX)

| Implementation                     |       Time | Inferences/sec |
| ---------------------------------- | ---------: | -------------: |
| Rust (sequential)                  |     2.686s |         74,460 |
| Rust (parallel, default threads)   | **0.355s** |        563,516 |
| TensorFlow CPU (`batch_size=8192`) |     0.458s |        436,261 |

### Memory Footprint

| Implementation | Model (weights+bias) | Input + Output | Compute Buffers |    Total |
| -------------- | -------------------: | -------------: | --------------: | -------: |
| Rust (seq)     |             529.3 KB |       343.3 MB |      **2.9 KB** | 343.8 MB |
| Rust (par)     |             529.3 KB |       343.3 MB |     **46.9 KB** | 343.9 MB |
| TensorFlow     |             529.3 KB |       343.3 MB |     **23.4 MB** | 367.3 MB |

Model weights are shared across all threads/batches (not replicated). Rust parallel uses 16× more compute buffer memory than sequential (one buffer per thread), but still 500× less than TensorFlow's batch buffer.

**Note:** On smaller models or fewer inferences, TensorFlow's performance degrades significantly due to Python/framework overhead, JIT compilation, and batch scheduling. Rust maintains consistent low-latency performance regardless of scale.

### How to run

```bash
# Rust
cargo run --release --bin parallel_benchmark

# TensorFlow (CPU)
python3 benchmarks/tensorflow_benchmark.py
```

## Quick Start

### Simple Neural Network

```rust
use instmodel_inference::{
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
use instmodel_inference::{
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
use instmodel_inference::{InstructionModel, InstructionModelInfo};

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
use instmodel_inference::{InstructionModel, InstructionModelInfo};
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
use instmodel_inference::{
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

| Activation | Description                              |
| ---------- | ---------------------------------------- |
| `Relu`     | f(x) = max(0, x)                         |
| `Sigmoid`  | f(x) = 1 / (1 + exp(-x))                 |
| `Softmax`  | Numerically stable softmax over a buffer |
| `Tanh`     | f(x) = tanh(x)                           |
| `Sqrt`     | f(x) = sqrt(x) for x > 0, else 0         |
| `Log`      | f(x) = ln(x + 1) for x > 0, else 0       |
| `Log10`    | f(x) = log10(x + 1) for x > 0, else 0    |
| `Inverse`  | f(x) = 1 - x                             |

### Instruction Types

| Instruction           | JSON Type                      | Description                                           |
| --------------------- | ------------------------------ | ----------------------------------------------------- |
| Dot Product           | `DOT`                          | Matrix multiplication with optional activation        |
| Copy                  | `COPY`                         | Copy buffer contents to another location              |
| Copy Masked           | `COPY_MASKED`                  | Copy specific indices from a buffer                   |
| Activation            | `ACTIVATION`                   | Apply activation function in-place                    |
| Element-wise Add      | `ADD_ELEMENTWISE`              | Add parameters element-wise                           |
| Element-wise Multiply | `MUL_ELEMENTWISE`              | Multiply by parameters element-wise                   |
| Buffers Add           | `ADD_ELEMENTWISE_BUFFERS`      | Sum multiple buffers                                  |
| Buffers Multiply      | `MULTIPLY_ELEMENTWISE_BUFFERS` | Multiply multiple buffers element-wise                |
| Reduce Sum            | `REDUCE_SUM`                   | Sum all values in a buffer to a single value          |
| Attention             | `ATTENTION`                    | Attention mechanism (linear + softmax + element-wise) |
| Map Transform         | `MAP_TRANSFORM`                | Lookup and transform using a map                      |

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

### Parallel Prediction

For batch inference across multiple threads:

```rust
use instmodel_inference::{InstructionModel, PredictConfig};

let model = InstructionModel::new(model_info)?;

// Flatten all inputs into a contiguous buffer
// For 1000 samples with 250 features each:
let inputs: Vec<f32> = samples.iter().flatten().copied().collect();

// Default config uses all available CPU cores
let config = PredictConfig::new();
let result = model.predict_parallel(&inputs, config)?;

// Access results
let all_outputs = result.as_slice();
let first_sample = result.get_result(0)?;

// Copy to your own buffer
let mut my_buffer = vec![0.0f32; result.len()];
result.copy_results(&mut my_buffer)?;
```

With custom configuration:

```rust
let config = PredictConfig::new()
    .with_threads(8)                    // Use 8 threads
    .with_slice_result_buffer(0, 100);  // Only return first 100 samples

let result = model.predict_parallel(&inputs, config)?;
```

### GPU-Embeddable Inference (WGSL)

The library provides GPU-embeddable inference functions in WGSL that can be called from within your own GPU compute shaders. This is particularly useful for RL simulations where each episode runs in its own GPU thread.

**Rust Side - Prepare the model:**

```rust
use instmodel_inference::gpu::{GpuModel, get_instmodel_wgsl};

// Convert your model to GPU format
let gpu_model = GpuModel::from_info(&model_info)?;

// Get the model data as bytes for GPU buffer
let model_bytes = gpu_model.as_bytes();

// Get WGSL shader code to include in your kernel
let wgsl_functions = get_instmodel_wgsl(gpu_model.compute_buffer_size() as u32);
```

**WGSL Side - Use in your compute shader:**

```wgsl
// Your shader bindings
@group(0) @binding(0) var<storage, read> model_data: array<f32>;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputs: array<f32>;

// Include the generated instmodel functions (via string replacement at runtime)
// This provides: predict(), get_feature_size(), get_output_size(), get_output_start()

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;

    // Each thread has its own compute buffer (function-local)
    var compute_buffer: array<f32, 1024>;  // Size from gpu_model.compute_buffer_size()

    // Model offset (0 for single model, or index into packed multi-model buffer)
    let model_offset: u32 = 0u;

    // Copy input to compute buffer
    let feature_size = get_feature_size(model_offset);
    let input_offset = thread_id * feature_size;
    for (var i: u32 = 0u; i < feature_size; i = i + 1u) {
        compute_buffer[i] = inputs[input_offset + i];
    }

    // Run inference - this executes all model instructions
    predict(model_offset, &compute_buffer);

    // Copy output from compute buffer
    let output_start = get_output_start(model_offset);
    let output_size = get_output_size(model_offset);
    let output_offset = thread_id * output_size;
    for (var i: u32 = 0u; i < output_size; i = i + 1u) {
        outputs[output_offset + i] = compute_buffer[output_start + i];
    }
}
```

**Key GPU Functions Available:**

| Function | Description |
| --- | --- |
| `predict(model_offset, &compute_buffer)` | Execute all model instructions |
| `get_feature_size(model_offset)` | Get input feature count |
| `get_output_size(model_offset)` | Get output size |
| `get_output_start(model_offset)` | Get output position in compute buffer |
| `get_compute_buffer_size(model_offset)` | Get required compute buffer size |
| `get_full_model_size(model_offset)` | Get total model size (for multi-model packing) |

**Why GPU-Embedded Inference?**

For RL and simulation workloads, the model data stays on GPU and each thread can call `predict()` multiple times per episode without CPU<->GPU transfers. This eliminates transfer overhead and enables massive parallelism across episodes.

### Model Validation

Include validation data to verify model correctness on creation:

```rust
use instmodel_inference::instruction_model_info::ValidationData;

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
