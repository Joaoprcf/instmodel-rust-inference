# Changelog

## 1.2.0

- Sequence models (JSON `format_version` 2, all new fields optional): `SequenceModel`
  runs a causal transformer's step program one token at a time, and `SequenceState`
  holds its key/value caches and feature scanners for the caller (`clone`, exact
  `truncate`, `reset`, versioned `to_bytes` / `SequenceModel::state_from_bytes` bound to
  a model fingerprint). `step` does not allocate. Loading replays `validation_sequences`
  (1e-3 absolute, identical argmax). `argmax_allowed` picks the first maximum among
  allowed tokens.
- Instructions `EMBEDDING`, `LAYER_NORM` (stateless; also usable in `InstructionModel`,
  and `EMBEDDING` shares weight slots and θ with a tied readout `DOT`), `SEGMENT_FEATURES`
  and `CAUSAL_ATTENTION` (stateful; `SequenceModel` only). `Instruction::apply_with_state`
  defaults to `apply`.
- `InstructionModelInfo` gains `format_version`, `sequence` and `validation_sequences`;
  code that builds it with a struct literal must add them (`None`).
- `InstructionModel::new` rejects stateful instructions (`StatefulInstruction`) and
  unknown `format_version`s; `GpuModel` rejects the four new instructions.
- `sequence_benchmark` binary.

## 1.1.1

- Security: every instruction reference in model JSON (COPY `internal_index`,
  COPY_MASKED indexes, parameter/map indices and sizes, MAP_TRANSFORM offsets,
  element-wise buffer sizes, buffer-heads shapes) is validated before execution, on
  both the CPU and `GpuModel::from_info` paths, using the existing error variants.
  Malformed models previously panicked, wrote into neighbouring buffers, or (ATTENTION)
  accessed memory out of bounds.
- Removed unsound `unsafe` from ATTENTION; overflow-safe size math in feature counting,
  buffer offsets, GPU dispatch/buffer sizing and specialized WGSL generation.
- Internal dedupe, terser docs and tests; no API, JSON or numerical changes for valid models.

## 1.1.0

- `gpu::get_specialized_wgsl_lanes`: topology-specialized straight-line `predict()`
  WGSL; one shader serves every `PopulationPack` candidate.
- `gpu::get_instmodel_wgsl_lanes`: lane-sliced shared-memory interpreter variant.
- `GpuContextOptions::required_limits` to request larger device limits.

## 1.0.0

- `graph`: weightless graph DSL (`Graph`, `ModelGraph`, `Buffer`, `WeightId`,
  `Constant`) with `weights_map` / `compile` / `compile_zeroed` / `to_model` and
  weight sharing.
- `params`: `ParamLayout`, `ParamCopy`, `ParamKind` — the public flat-θ order.
- `InstructionModel::{apply_theta, read_theta, theta_len, param_layout, try_clone}`.
- `evolution`: `EsOptimizer`/`EsConfig`, counter-based noise, `cosine_anneal`.
- `gpu::PopulationPack`; `gpu-runtime` feature with `GpuContext` and
  `PopulationEvaluator`.
- GPU opcodes COPY_MASKED, CLIP_ELEMENTWISE, ELEM_WISE_BUFFERS_ADD/MUL,
  MULTIPLY/ADD_BUFFER_HEADS, REDUCE_SUM; blob version 2 (pair blobs with WGSL from the
  same crate version). MAP_TRANSFORM and ATTENTION stay CPU-only.
- Error enums are `#[non_exhaustive]`; new `GraphError`, `EvolutionError`,
  `PopulationError`, `GpuRuntimeError`.
- `gpu-benchmark` forwards to `gpu-runtime`; MSRV 1.89.
- JSON format and 0.9 inference APIs unchanged.
