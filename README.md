# instmodel-rust-inference

A high-performance neural network inference library for Rust that executes optimized computation sequences through a unified buffer architecture — now with graph-based model authoring, in-place mutable weights, a deterministic evolution-strategies optimizer, and a GPU population evaluator built for evolutionary RL workloads.

## Installation

```bash
cargo add instmodel_inference
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
instmodel_inference = "1.0"
```

MSRV: Rust 1.89 (AVX-512 intrinsics used by the SIMD dot-product path).

## Overview

Models are defined as a sequence of instructions that operate on computation buffers, enabling efficient memory reuse and predictable performance.

**Key Features:**

- Weightless graph DSL: author an architecture once, compile it with any flat parameter vector θ
- Canonical flat-θ ↔ model mapping (`ParamLayout`) — a documented public contract
- In-place mutable models: `apply_theta` overwrites weights without rebuilding (~13× faster than recreation)
- Deterministic OpenAI-style evolution strategies (`EsOptimizer`) with counter-based noise
- Zero-repack GPU population packing and a feature-gated wgpu evaluation host
- Instruction-based execution model with GPU-embeddable WGSL inference
- JSON serialization/deserialization, built-in validation, parallel batch prediction

## Quick Start: Graph Authoring (recommended)

Declare the architecture as a small DAG — structure only, no weight data — then compile it with any flat θ:

```rust
use instmodel_inference::activation::Activation;
use instmodel_inference::graph::Graph;

let graph = Graph::new();
let x = graph.input(8, None);
let normalized = graph.normalize(&x, vec![0.0; 8], vec![1.0; 8]);
let hidden = graph.dense(&normalized, 16, Some(Activation::Tanh));
let y = graph.dense(&hidden, 1, None);
let model_graph = graph.model(vec![&x], &y);

// The canonical flat-θ layout is derived from structure alone.
let layout = model_graph.weights_map()?;
let theta = vec![0.01f32; layout.total];

// Compile and run.
let model = model_graph.to_model(&theta)?;
let output = model.predict(&[0.0; 8])?;
```

All 14 instruction types are authorable: `dense` / `dense_shared`, `concat`, `gather`, `activation`, `clip`, `add_const` / `mul_const`, `normalize`, `add` / `mul` (n-ary buffers), `add_heads` / `mul_heads`, `reduce_sum`, `attention` / `attention_shared`, and `map_transform`. Reusing a `WeightId` (via the `_shared` variants) shares one weight tensor — a single θ slot referenced by several instructions.

### The canonical θ order (public contract)

θ is laid out per weight slot in first-seen emission order: the row-major `[out × in]` weight run, then the `[out]` bias run. `ParamLayout::from_info(&model_graph.compile(&theta)?)` always equals `model_graph.weights_map()?`, and the GPU blob's weights region is byte-identical to θ. Constants (`normalize`, `clip` bounds, …) and maps are *parameters*, not θ — they are never trained or overwritten by θ writes.

## Mutable Models: No Rebuild, No Latency Loss

A compiled model can have its full weight set swapped in place:

```rust
let mut model = model_graph.to_model(&theta)?;

model.apply_theta(&new_theta)?;      // in-place overwrite, allocation-free
model.read_theta(&mut theta_out)?;   // read weights back in canonical order
let worker = model.try_clone()?;     // independent copy for another thread
```

`cargo run --release --bin es_benchmark` measures the difference on a 250 → 300 → 200 network (135,500 parameters, laptop-class hardware):

| Operation | Time |
| --- | ---: |
| Full rebuild (compile + validate + schedule) | ~105 µs |
| `apply_theta` (in-place overwrite) | ~8 µs |
| `try_clone` | ~9 µs |

## Evolution Strategies Training

The `evolution` module is a deterministic OpenAI-ES toolkit: mirrored sampling, centered-rank fitness shaping, SGD with momentum, and counter-based Gaussian noise that is a pure function of `(seed, step, index)` — results never depend on thread scheduling.

```rust
use instmodel_inference::evolution::{cosine_anneal, EsConfig, EsOptimizer};

let layout = model_graph.weights_map()?;
let mut optimizer = EsOptimizer::new(vec![0.0; layout.total], EsConfig::default())?;
let mut model = model_graph.to_model(&vec![0.0; layout.total])?;

let mut theta_f32 = Vec::new();
let mut fitness = vec![0.0f64; optimizer.population_size()];

for step in 0..total_steps {
    // Rank-shaped gradients keep O(1) magnitude even at the optimum,
    // so anneal BOTH sigma and the learning rate for θ to settle.
    optimizer.set_sigma(base_sigma * cosine_anneal(step, total_steps, 0.2))?;
    optimizer.set_learning_rate(base_lr * cosine_anneal(step, total_steps, 0.05))?;

    optimizer.ask()?;
    for candidate in 0..optimizer.population_size() {
        optimizer.candidate_f32_into(candidate, &mut theta_f32)?;
        model.apply_theta(&theta_f32)?;   // mutate, don't rebuild
        fitness[candidate] = evaluate(&model);
    }
    optimizer.tell(&fitness)?;
}
```

See `examples/es_train.rs` for the complete runnable version (training, logging, and a serde round-trip of the final model):

```bash
cargo run --release --example es_train
```

Optimizer overhead per `ask` + `tell` step (pairs = 16, constant-time fitness): ~0.2 ms at 1k parameters, ~1.8 ms at 10k, ~20 ms at 100k.

## GPU Population Evaluation (`gpu-runtime`)

`PopulationPack` (always available, no wgpu dependency) tiles a model into one contiguous blob, one copy per candidate. Because the blob's weights region is byte-identical to canonical θ, writing a candidate is a pure memcpy — ~40 GB/s in the benchmark, never a repack.

With the `gpu-runtime` feature, `PopulationEvaluator` runs the whole population against a shared input batch in a single compute dispatch:

```rust
use instmodel_inference::gpu::{GpuContext, GpuContextOptions, PopulationEvaluator};

let context = GpuContext::new(&GpuContextOptions::default())?;
let mut evaluator = PopulationEvaluator::from_graph(
    &context, &model_graph, optimizer.population_size(), batch_size)?;

let mut outputs = vec![0.0f32; evaluator.n_candidates() * batch_size * evaluator.output_size()];

optimizer.ask()?;
evaluator.write_population_f64(optimizer.population())?;
evaluator.evaluate(&inputs, &mut outputs)?;   // one dispatch, sync readback
// outputs are candidate-major: [n_candidates × batch × output_size]
```

Buffers and the compiled pipeline persist across `evaluate` calls; per step you pay one upload, one dispatch, and one readback. By default software/CPU adapters (llvmpipe, SwiftShader) are rejected so a training loop fails loudly instead of silently crawling — opt in with `allow_software_adapter`.

```bash
cargo run --release --example es_train_gpu --features gpu-runtime
```

## Feature Flags

| Feature | Default | Description |
| --- | --- | --- |
| `gpu-runtime` | off | wgpu evaluation host: `GpuContext`, `PopulationEvaluator` (adds `wgpu` + `pollster`) |
| `gpu-benchmark` | off | GPU benchmark binary; implies `gpu-runtime` |

Everything else — graph DSL, `ParamLayout`, `apply_theta`, `EsOptimizer`, `PopulationPack`, GPU blob serialization, and WGSL generation — is available with no features and no GPU dependencies.

## Determinism

- **CPU**: bit-exact. `apply_theta` followed by `predict` matches a freshly rebuilt model bitwise; the serde JSON round-trip preserves predictions exactly.
- **ES noise**: perturbation `i` of step `s` is a pure function of `(seed, s, i)` (splitmix64 + Box–Muller). Two optimizers with the same seed produce identical trajectories on any machine or thread count.
- **GPU**: deterministic per device; CPU/GPU parity is exact for copy-like operations and within ~1e-4 for dot-product paths (f32 accumulation order differs).

## Inference Benchmarks

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
# Rust inference throughput
cargo run --release --bin parallel_benchmark

# ES workflow (mutation vs rebuild, pack writes, optimizer overhead)
cargo run --release --bin es_benchmark

# TensorFlow (CPU)
python3 benchmarks/tensorflow_benchmark.py
```

## Low-Level Model Definition

The graph DSL compiles down to `InstructionModelInfo`; you can also build that structure directly.

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

// Or get a single output value directly
let result = model.predict_single(&input)?;
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

The JSON format is unchanged from 0.9 — existing model files load as-is.

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
| `Gelu`     | Gaussian Error Linear Unit               |
| `Softplus` | f(x) = ln(1 + exp(x))                    |
| `Exp`      | f(x) = exp(x)                            |
| `Sign`     | f(x) = sign(x) ∈ {-1, 0, 1}              |

### Instruction Types

| Instruction           | JSON Type                      | Description                                           |
| --------------------- | ------------------------------ | ----------------------------------------------------- |
| Dot Product           | `DOT`                          | Matrix multiplication with optional activation        |
| Copy                  | `COPY`                         | Copy buffer contents to another location              |
| Copy Masked           | `COPY_MASKED`                  | Copy specific indices from a buffer                   |
| Activation            | `ACTIVATION`                   | Apply activation function in-place                    |
| Element-wise Add      | `ADD_ELEMENTWISE`              | Add parameters element-wise                           |
| Element-wise Multiply | `MUL_ELEMENTWISE`              | Multiply by parameters element-wise                   |
| Element-wise Clip     | `CLIP_ELEMENTWISE`             | Clamp between optional parameter bounds               |
| Buffers Add           | `ADD_ELEMENTWISE_BUFFERS`      | Sum multiple buffers                                  |
| Buffers Multiply      | `MULTIPLY_ELEMENTWISE_BUFFERS` | Multiply multiple buffers element-wise                |
| Add Buffer Heads      | `ADD_BUFFER_HEADS`             | Add a per-head value to each segment of a buffer      |
| Multiply Buffer Heads | `MULTIPLY_BUFFER_HEADS`        | Multiply each segment of a buffer by a per-head value |
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

**Blob version note:** 1.0.0 bumps the GPU blob to version 2 (it adds opcodes for masked copy, clip, n-ary buffer ops, head ops, and reduce-sum). Older WGSL interpreters silently skip unknown opcodes, so always pair the blob and the generated WGSL from the same crate version. `MAP_TRANSFORM` and `ATTENTION` remain CPU-only.

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

The model output is always the last computation buffer, on CPU and GPU alike.

## License

MIT
