# Changelog

## 1.0.0 — 2026-07-03

First stable release. 0.9 was a pure-inference engine; 1.0.0 adds the full
authoring-and-training layer for evolutionary / RL workloads on top of it.

### Added

- **`graph` module** — weightless graph DSL covering all 14 instruction
  types (`Graph`, `ModelGraph`, `Buffer`, `WeightId`, `Constant`).
  `weights_map()` derives the canonical flat-θ layout from structure alone;
  `compile(θ)` / `compile_zeroed()` / `to_model(θ)` produce runnable models.
  Weight sharing via `dense_shared` / `attention_shared`.
- **`params` module** — `ParamLayout`, `ParamCopy`, `ParamKind`: the
  canonical flat-θ ↔ model offset table (per weight slot in first-seen
  order, row-major `[out × in]` weight run then `[out]` bias run). This
  ordering is now a documented public contract.
- **In-place mutable models** — `InstructionModel::{apply_theta,
  read_theta, theta_len, param_layout, try_clone}`: overwrite a live
  model's full weight set without rebuilding (~13× faster than
  recreation on the 135k-parameter benchmark model).
- **`evolution` module** — deterministic OpenAI-style ES toolkit:
  `EsOptimizer` / `EsConfig` (mirrored sampling, centered-rank shaping,
  SGD with momentum, f64 state, allocation-free `ask`/`tell`),
  counter-based `GaussianStream` + `perturbation_seed` +
  `fill_perturbation` (noise is a pure function of `(seed, step, index)`),
  and `cosine_anneal`.
- **`gpu::PopulationPack`** — tiles a model into one contiguous blob, one
  copy per candidate; the weights region is byte-identical to canonical θ,
  so candidate writes are pure memcpys (zero repacking). Available without
  any feature flag.
- **`gpu-runtime` feature** — wgpu evaluation host: `GpuContext` /
  `GpuContextOptions` (software adapters rejected by default) and
  `PopulationEvaluator` (persistent buffers, whole population × batch in
  one dispatch, synchronous readback, `write_population_f64` matching
  `EsOptimizer::population()`).
- **Seven new GPU opcodes** (blob version 2): `COPY_MASKED`,
  `CLIP_ELEMENTWISE`, `ELEM_WISE_BUFFERS_ADD`, `ELEM_WISE_BUFFERS_MUL`,
  `MULTIPLY_BUFFER_HEADS`, `ADD_BUFFER_HEADS`, `REDUCE_SUM`, with matching
  WGSL interpreter cases. `MAP_TRANSFORM` and `ATTENTION` remain CPU-only.
- **New error types** — `GraphError`, `EvolutionError`, `PopulationError`,
  `GpuRuntimeError` (feature-gated).
- **Examples and benchmarks** — `examples/es_train.rs` (CPU ES loop with
  in-place mutation and a serde round-trip), `examples/es_train_gpu.rs`
  (GPU population evaluation; requires `gpu-runtime`), and the
  `es_benchmark` binary (mutation vs rebuild, pack write throughput,
  optimizer overhead, clone cost).

### Changed

- Error enums (`InstructionModelError`, `GpuModelError`, and all new ones)
  are `#[non_exhaustive]` and gained variants — the one deliberate
  compatibility-affecting change of this release; match with a wildcard
  arm.
- `GPU_MODEL_VERSION` bumped 1 → 2 for the new opcodes. Older WGSL
  interpreters silently skip unknown opcodes, so pair the blob and the
  generated WGSL from the same crate version.
- The `gpu-benchmark` feature now forwards to the new `gpu-runtime`
  feature (`gpu-benchmark = ["gpu-runtime"]`); enabling it behaves as
  before.
- `GpuModel` exposes `weights_offset()`, `theta_len()`, `write_theta()`,
  and `read_theta()` for direct weights-region access.
- Declared `rust-version = "1.89"` (MSRV; set by the stabilized AVX-512
  intrinsics in the SIMD dot-product path).

### Unchanged guarantees

- The JSON model format is fully backward compatible; 0.9 model files
  load unchanged.
- All 0.9 inference APIs (`InstructionModel::new`, `predict`,
  `predict_single`, `predict_parallel`, `predict_with_buffer`, GPU blob
  embedding, WGSL generation) are source-compatible.
- The model output remains the last computation buffer, on CPU and GPU.
