# instmodel-rust-inference

This repo is meant to be used with coding agents, not read manually: point your agent at the
crate docs (`cargo doc --open`) and the tests, which are the authoritative reference.

Run neural networks exported as instmodel JSON (from `instmodel` / `instmodel-rust`) with a
small, validated, allocation-free CPU engine (SIMD dense layers, multi-threaded batches), or
author them in Rust with a weightless graph DSL, mutate weights in place for evolution
strategies, and evaluate whole populations on the GPU via WGSL/wgpu (`gpu-runtime` feature).
The flat parameter order is documented once, on `ParamLayout`.

## Install

```bash
cargo add instmodel_inference                          # CPU, graph DSL, ES, WGSL generation
cargo add instmodel_inference --features gpu-runtime   # + wgpu evaluation host
```

## Example

```rust
use instmodel_inference::{InstructionModel, InstructionModelInfo};

let info: InstructionModelInfo = serde_json::from_str(r#"{
    "feature_size": 2,
    "buffer_sizes": [2, 1],
    "instructions": [{"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "SIGMOID"}],
    "weights": [[[0.5, -0.25]]],
    "bias": [[0.1]]
}"#)?;
let model = InstructionModel::new(info)?;   // validates every index/size up front
let y = model.predict(&[1.0, 2.0])?;        // output = last buffer
```

## Sequence models

A causal transformer exported by instmodel as `format_version` 2 runs one token per
`step`. Its key/value caches and feature scanners live in a `SequenceState` the caller
owns: keep one per sequence, `clone` it to branch, `truncate` it to rewind, or save it
with `to_bytes`. Stepping does not allocate, and one `SequenceModel` serves any number of
states from any number of threads.

```rust
use instmodel_inference::{InstructionModelInfo, SequenceModel, argmax_allowed};

let info: InstructionModelInfo = serde_json::from_str(&std::fs::read_to_string("model.json")?)?;
let model = SequenceModel::new(info)?;      // validates, replays validation_sequences
let mut state = model.new_state();
let mut token = 3;
for _ in 0..10 {
    let logits = model.step(&mut state, token)?; // next-token logits
    token = argmax_allowed(logits, &[0, 1, 2]).unwrap_or(0);
}
let saved = state.to_bytes();
let restored = model.state_from_bytes(&saved)?; // continues bit-identically
```

`InstructionModel` rejects models with stateful instructions (`SEGMENT_FEATURES`,
`CAUSAL_ATTENTION`); the GPU path supports none of the four sequence instructions.
`cargo run --release --bin sequence_benchmark` reports microseconds per token.

## License

MIT
