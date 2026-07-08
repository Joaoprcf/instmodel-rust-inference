//! Model-topology-specialized WGSL codegen (lane-sliced shared-memory form).
//!
//! [`get_specialized_wgsl_lanes`] emits a straight-line `predict()` for one
//! packed model blob: the instruction stream is resolved at generation time,
//! so the shader does no per-call instruction decoding, every buffer offset
//! and loop bound is a literal, and constant parameter vectors (element-wise
//! constants, clip bounds, gather pointer lists) are inlined as literals.
//! Only the per-candidate weights are still read from `model_data`, relative
//! to the runtime `model_offset`, so one generated shader serves every
//! candidate of a `PopulationPack` (all candidates share the topology).
//!
//! The generated source is standalone like
//! [`get_instmodel_wgsl_lanes`](crate::gpu::shaders::get_instmodel_wgsl_lanes)
//! (same `compute_buffers` / `CB_LANES` / `cb_index` contract, same
//! `@workgroup_size(CB_LANES)` requirement) and must not be combined with the
//! other variants in one module.

use crate::gpu::errors::GpuModelError;
use crate::gpu::gpu_instruction::{CLIP_BOUND_NONE, activation_types, opcodes};

const HEADER_MAGIC_INDEX: usize = 0;
const HEADER_VERSION_INDEX: usize = 1;
const HEADER_COMPUTE_BUFFER_SIZE_INDEX: usize = 4;
const HEADER_INSTRUCTION_COUNT_INDEX: usize = 5;
const HEADER_INSTRUCTIONS_OFFSET_INDEX: usize = 6;
const HEADER_WEIGHTS_OFFSET_INDEX: usize = 7;
const HEADER_PARAMS_OFFSET_INDEX: usize = 9;
const HEADER_FULL_MODEL_SIZE_INDEX: usize = 12;

const EXPECTED_MAGIC: u32 = 0x494D5047;
const SUPPORTED_VERSION: u32 = 2;
const INSTRUCTION_FIELDS: usize = 8;

/// Above this many multiply-accumulate terms a DOT is emitted as literal-bound
/// loops instead of straight-line code, to keep the shader size sane.
const DOT_UNROLL_TERM_LIMIT: u32 = 4096;
/// Element-wise instructions beyond this size do not fit the "small episode
/// model" shape this generator targets.
const ELEMENTWISE_UNROLL_LIMIT: u32 = 1024;

/// Generates a topology-specialized, lane-sliced `predict()` WGSL module for
/// the packed model blob (one candidate's f32 slice — e.g.
/// `&pack_f32[..pack.model_stride()]`).
///
/// See the module docs for the embedding contract. `lanes` must equal the
/// embedding kernel's workgroup size.
pub fn get_specialized_wgsl_lanes(model_blob: &[f32], lanes: u32) -> Result<String, GpuModelError> {
    if lanes == 0 {
        return Err(GpuModelError::InvalidModelBinary {
            message: "lane count must be at least 1".to_string(),
        });
    }
    let blob = Blob { data: model_blob };
    let magic = blob.u32_at(HEADER_MAGIC_INDEX)?;
    if magic != EXPECTED_MAGIC {
        return Err(GpuModelError::InvalidModelBinary {
            message: format!("bad magic 0x{magic:08X} (expected 0x{EXPECTED_MAGIC:08X})"),
        });
    }
    let version = blob.u32_at(HEADER_VERSION_INDEX)?;
    if version != SUPPORTED_VERSION {
        return Err(GpuModelError::InvalidModelBinary {
            message: format!("unsupported blob version {version} (expected {SUPPORTED_VERSION})"),
        });
    }
    let full_size = blob.u32_at(HEADER_FULL_MODEL_SIZE_INDEX)?;
    if (full_size as usize) > model_blob.len() {
        return Err(GpuModelError::InvalidModelBinary {
            message: format!(
                "header declares {full_size} f32s but the blob slice holds {}",
                model_blob.len()
            ),
        });
    }

    let mut emitter = Emitter {
        blob,
        lanes,
        cb_size: blob.u32_at(HEADER_COMPUTE_BUFFER_SIZE_INDEX)?,
        full_size,
        weights_offset: blob.u32_at(HEADER_WEIGHTS_OFFSET_INDEX)?,
        params_offset: blob.u32_at(HEADER_PARAMS_OFFSET_INDEX)?,
        out: String::new(),
    };

    let instruction_count = blob.u32_at(HEADER_INSTRUCTION_COUNT_INDEX)? as usize;
    let instructions_offset = blob.u32_at(HEADER_INSTRUCTIONS_OFFSET_INDEX)? as usize;
    for idx in 0..instruction_count {
        let base = instructions_offset + idx * INSTRUCTION_FIELDS;
        let inst = Inst {
            opcode: emitter.blob.u32_at(base)?,
            input_ptr: emitter.blob.u32_at(base + 1)?,
            output_ptr: emitter.blob.u32_at(base + 2)?,
            data_size: emitter.blob.u32_at(base + 3)?,
            param0: emitter.blob.u32_at(base + 4)?,
            param1: emitter.blob.u32_at(base + 5)?,
            param2: emitter.blob.u32_at(base + 6)?,
        };
        emitter.emit_instruction(idx, &inst)?;
    }

    let cb_size = emitter.cb_size;
    let total = cb_size
        .checked_mul(lanes)
        .filter(|t| *t > 0)
        .ok_or_else(|| GpuModelError::InvalidModelBinary {
            message: format!("compute buffer {cb_size} f32s x {lanes} lanes overflows"),
        })?;
    Ok(format!(
        "// Specialized instmodel predict (lane-sliced shared-memory variant)\n\
         // Generated for a fixed topology: {instruction_count} instructions, \
         compute buffer {cb_size} f32s, {lanes} lanes\n\n\
         const MAX_COMPUTE_BUFFER: u32 = {cb_size}u;\n\
         const CB_LANES: u32 = {lanes}u;\n\
         var<workgroup> compute_buffers: array<f32, {total}u>;\n\n\
         fn cb_index(buffer_index: u32, lane: u32) -> u32 {{\n\
             return buffer_index * CB_LANES + lane;\n\
         }}\n\n\
         {activations}\n\n\
         {softmax}\n\n\
         fn predict(model_offset: u32, lane: u32) {{\n{body}}}\n",
        activations = include_str!("shaders/activations.wgsl"),
        softmax = LANE_SOFTMAX,
        body = emitter.out,
    ))
}

const LANE_SOFTMAX: &str = "\
// Numerically stable in-place softmax over one lane's buffer range.
fn apply_softmax(lane: u32, start: u32, size: u32) {
    var max_val: f32 = compute_buffers[start * CB_LANES + lane];
    for (var i: u32 = 1u; i < size; i = i + 1u) {
        max_val = max(max_val, compute_buffers[(start + i) * CB_LANES + lane]);
    }
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < size; i = i + 1u) {
        let e = exp(compute_buffers[(start + i) * CB_LANES + lane] - max_val);
        compute_buffers[(start + i) * CB_LANES + lane] = e;
        sum = sum + e;
    }
    let inv_sum = 1.0 / sum;
    for (var i: u32 = 0u; i < size; i = i + 1u) {
        compute_buffers[(start + i) * CB_LANES + lane] =
            compute_buffers[(start + i) * CB_LANES + lane] * inv_sum;
    }
}";

#[derive(Copy, Clone)]
struct Blob<'a> {
    data: &'a [f32],
}

impl Blob<'_> {
    fn u32_at(&self, index: usize) -> Result<u32, GpuModelError> {
        self.data
            .get(index)
            .map(|v| v.to_bits())
            .ok_or_else(|| GpuModelError::InvalidModelBinary {
                message: format!("blob read out of range at f32 index {index}"),
            })
    }

    fn f32_at(&self, index: usize) -> Result<f32, GpuModelError> {
        self.data
            .get(index)
            .copied()
            .ok_or_else(|| GpuModelError::InvalidModelBinary {
                message: format!("blob read out of range at f32 index {index}"),
            })
    }
}

struct Inst {
    opcode: u32,
    input_ptr: u32,
    output_ptr: u32,
    data_size: u32,
    param0: u32,
    param1: u32,
    param2: u32,
}

struct Emitter<'a> {
    blob: Blob<'a>,
    lanes: u32,
    cb_size: u32,
    full_size: u32,
    weights_offset: u32,
    params_offset: u32,
    out: String,
}

fn inst_err(instruction_index: usize, message: impl Into<String>) -> GpuModelError {
    GpuModelError::InvalidInstruction {
        instruction_index,
        message: message.into(),
    }
}

fn f32_literal(value: f32, idx: usize) -> Result<String, GpuModelError> {
    if !value.is_finite() {
        return Err(inst_err(
            idx,
            format!("non-finite constant parameter {value} cannot be inlined"),
        ));
    }
    Ok(format!("{value:?}"))
}

/// Combines terms with a binary operator as a balanced tree, so the compiled
/// dependency chain is `log2(n)` deep instead of `n`.
fn balanced_tree(mut terms: Vec<String>, op: &str) -> String {
    if terms.is_empty() {
        return "0.0".to_string();
    }
    while terms.len() > 1 {
        let mut next = Vec::with_capacity(terms.len().div_ceil(2));
        let mut iter = terms.chunks(2);
        for pair in &mut iter {
            match pair {
                [a, b] => next.push(format!("({a} {op} {b})")),
                [a] => next.push(a.clone()),
                _ => {}
            }
        }
        terms = next;
    }
    terms.swap_remove(0)
}

fn activation_expr(activation: u32, expr: String, idx: usize) -> Result<String, GpuModelError> {
    let function = match activation {
        activation_types::NONE => return Ok(expr),
        activation_types::RELU => "activation_relu",
        activation_types::SIGMOID => "activation_sigmoid",
        activation_types::TANH => "activation_tanh",
        activation_types::SQRT => "activation_sqrt",
        activation_types::LOG => "activation_log",
        activation_types::LOG10 => "activation_log10",
        activation_types::INVERSE => "activation_inverse",
        activation_types::GELU => "activation_gelu",
        activation_types::SOFTPLUS => "activation_softplus",
        activation_types::EXP => "activation_exp",
        activation_types::SIGN => "activation_sign",
        other => {
            return Err(inst_err(idx, format!("unknown activation type {other}")));
        }
    };
    Ok(format!("{function}({expr})"))
}

impl Emitter<'_> {
    /// Lane-sliced access to compute-buffer element `index` (a literal).
    fn cb(&self, index: u32, idx: usize) -> Result<String, GpuModelError> {
        if index >= self.cb_size {
            return Err(inst_err(
                idx,
                format!("buffer index {index} out of range ({} f32s)", self.cb_size),
            ));
        }
        Ok(format!("compute_buffers[{}u + lane]", index * self.lanes))
    }

    /// Weight read at a literal model-relative offset.
    fn weight(&self, offset: u32, idx: usize) -> Result<String, GpuModelError> {
        if offset >= self.full_size {
            return Err(inst_err(
                idx,
                format!(
                    "weight offset {offset} out of range ({} f32s)",
                    self.full_size
                ),
            ));
        }
        Ok(format!("model_data[model_offset + {offset}u]"))
    }

    /// Baked constant from the params region.
    fn param_f32(&self, relative: u32, idx: usize) -> Result<f32, GpuModelError> {
        let position = self.params_offset + relative;
        if position >= self.full_size {
            return Err(inst_err(
                idx,
                format!(
                    "params offset {position} out of range ({} f32s)",
                    self.full_size
                ),
            ));
        }
        self.blob.f32_at(position as usize)
    }

    fn param_u32(&self, relative: u32, idx: usize) -> Result<u32, GpuModelError> {
        Ok(self.param_f32(relative, idx)?.to_bits())
    }

    fn line(&mut self, text: &str) {
        self.out.push_str("    ");
        self.out.push_str(text);
        self.out.push('\n');
    }

    fn check_elementwise_size(&self, size: u32, idx: usize) -> Result<(), GpuModelError> {
        if size > ELEMENTWISE_UNROLL_LIMIT {
            return Err(inst_err(
                idx,
                format!(
                    "element-wise size {size} exceeds the specialization limit \
                     {ELEMENTWISE_UNROLL_LIMIT}; use get_instmodel_wgsl_lanes instead"
                ),
            ));
        }
        Ok(())
    }

    fn emit_instruction(&mut self, idx: usize, inst: &Inst) -> Result<(), GpuModelError> {
        self.line(&format!(
            "// [{idx}] opcode {} in={} out={} size={}",
            inst.opcode, inst.input_ptr, inst.output_ptr, inst.data_size
        ));
        match inst.opcode {
            opcodes::DOT => self.emit_dot(idx, inst),
            opcodes::ACTIVATION => self.emit_activation(idx, inst),
            opcodes::ELEM_WISE_ADD => self.emit_elem_wise_const(idx, inst, "+"),
            opcodes::ELEM_WISE_MUL => self.emit_elem_wise_const(idx, inst, "*"),
            opcodes::COPY => self.emit_copy(idx, inst),
            opcodes::COPY_MASKED => self.emit_copy_masked(idx, inst),
            opcodes::CLIP_ELEMENTWISE => self.emit_clip(idx, inst),
            opcodes::ELEM_WISE_BUFFERS_ADD => self.emit_elem_wise_buffers(idx, inst, "+"),
            opcodes::ELEM_WISE_BUFFERS_MUL => self.emit_elem_wise_buffers(idx, inst, "*"),
            opcodes::MULTIPLY_BUFFER_HEADS => self.emit_buffer_heads(idx, inst, "*"),
            opcodes::ADD_BUFFER_HEADS => self.emit_buffer_heads(idx, inst, "+"),
            opcodes::REDUCE_SUM => self.emit_reduce_sum(idx, inst),
            other => Err(inst_err(idx, format!("unknown opcode {other}"))),
        }
    }

    fn emit_dot(&mut self, idx: usize, inst: &Inst) -> Result<(), GpuModelError> {
        let output_size = inst.data_size;
        let input_size = inst.param1;
        let activation = inst.param2;
        let weights_base = self.weights_offset + inst.param0;
        let bias_base = weights_base + output_size * input_size;

        let terms = output_size.saturating_mul(input_size);
        if terms <= DOT_UNROLL_TERM_LIMIT {
            for out_idx in 0..output_size {
                let row = weights_base + out_idx * input_size;
                let mut products = Vec::with_capacity(input_size as usize);
                for i in 0..input_size {
                    products.push(format!(
                        "{} * {}",
                        self.weight(row + i, idx)?,
                        self.cb(inst.input_ptr + i, idx)?
                    ));
                }
                let sum = format!(
                    "{} + {}",
                    balanced_tree(products, "+"),
                    self.weight(bias_base + out_idx, idx)?
                );
                let value = if activation == activation_types::SOFTMAX {
                    sum
                } else {
                    activation_expr(activation, sum, idx)?
                };
                let store = self.cb(inst.output_ptr + out_idx, idx)?;
                self.line(&format!("{store} = {value};"));
            }
        } else {
            self.emit_dot_loop(idx, inst, weights_base, bias_base)?;
        }

        if activation == activation_types::SOFTMAX {
            self.line(&format!(
                "apply_softmax(lane, {}u, {}u);",
                inst.output_ptr, output_size
            ));
        }
        Ok(())
    }

    /// Literal-bound loop form for DOTs too large to unroll. Offsets and trip
    /// counts are still compile-time constants; only decode is avoided.
    fn emit_dot_loop(
        &mut self,
        idx: usize,
        inst: &Inst,
        weights_base: u32,
        bias_base: u32,
    ) -> Result<(), GpuModelError> {
        let output_size = inst.data_size;
        let input_size = inst.param1;
        let last_weight = bias_base + output_size.saturating_sub(1);
        // Validate the extremes once; the loop body stays within them.
        self.weight(last_weight, idx)?;
        self.cb(inst.input_ptr + input_size.saturating_sub(1), idx)?;
        self.cb(inst.output_ptr + output_size.saturating_sub(1), idx)?;

        let unroll_end = input_size & !3u32;
        let activation = inst.param2;
        let store_value = if activation == activation_types::SOFTMAX {
            "s".to_string()
        } else {
            activation_expr(activation, "s".to_string(), idx)?
        };
        let lanes = self.lanes;
        let input_ptr = inst.input_ptr;
        let output_ptr = inst.output_ptr;
        self.line(&format!(
            "for (var o: u32 = 0u; o < {output_size}u; o = o + 1u) {{"
        ));
        self.line(&format!(
            "    let row = model_offset + {weights_base}u + o * {input_size}u;"
        ));
        self.line("    var s0: f32 = 0.0;");
        self.line("    var s1: f32 = 0.0;");
        self.line("    var s2: f32 = 0.0;");
        self.line("    var s3: f32 = 0.0;");
        self.line("    var i: u32 = 0u;");
        self.line(&format!("    for (; i < {unroll_end}u; i = i + 4u) {{"));
        for k in 0..4u32 {
            self.line(&format!(
                "        s{k} = s{k} + model_data[row + i + {k}u] * \
                 compute_buffers[({input_ptr}u + i + {k}u) * {lanes}u + lane];"
            ));
        }
        self.line("    }");
        self.line("    var s: f32 = (s0 + s2) + (s1 + s3);");
        self.line(&format!("    for (; i < {input_size}u; i = i + 1u) {{"));
        self.line(&format!(
            "        s = s + model_data[row + i] * \
             compute_buffers[({input_ptr}u + i) * {lanes}u + lane];"
        ));
        self.line("    }");
        self.line(&format!(
            "    s = s + model_data[model_offset + {bias_base}u + o];"
        ));
        self.line(&format!(
            "    compute_buffers[({output_ptr}u + o) * {lanes}u + lane] = {store_value};"
        ));
        self.line("}");
        Ok(())
    }

    fn emit_activation(&mut self, idx: usize, inst: &Inst) -> Result<(), GpuModelError> {
        let activation = inst.param2;
        if activation == activation_types::NONE {
            return Ok(());
        }
        if activation == activation_types::SOFTMAX {
            self.line(&format!(
                "apply_softmax(lane, {}u, {}u);",
                inst.input_ptr, inst.data_size
            ));
            return Ok(());
        }
        self.check_elementwise_size(inst.data_size, idx)?;
        for i in 0..inst.data_size {
            let slot = self.cb(inst.input_ptr + i, idx)?;
            let value = activation_expr(activation, slot.clone(), idx)?;
            self.line(&format!("{slot} = {value};"));
        }
        Ok(())
    }

    fn emit_elem_wise_const(
        &mut self,
        idx: usize,
        inst: &Inst,
        op: &str,
    ) -> Result<(), GpuModelError> {
        self.check_elementwise_size(inst.data_size, idx)?;
        for i in 0..inst.data_size {
            let slot = self.cb(inst.input_ptr + i, idx)?;
            let constant = f32_literal(self.param_f32(inst.param0 + i, idx)?, idx)?;
            self.line(&format!("{slot} = {slot} {op} {constant};"));
        }
        Ok(())
    }

    fn emit_copy(&mut self, idx: usize, inst: &Inst) -> Result<(), GpuModelError> {
        self.check_elementwise_size(inst.data_size, idx)?;
        for i in 0..inst.data_size {
            let dst = self.cb(inst.output_ptr + i, idx)?;
            let src = self.cb(inst.input_ptr + i, idx)?;
            self.line(&format!("{dst} = {src};"));
        }
        Ok(())
    }

    fn emit_copy_masked(&mut self, idx: usize, inst: &Inst) -> Result<(), GpuModelError> {
        self.check_elementwise_size(inst.data_size, idx)?;
        for i in 0..inst.data_size {
            let pointer = self.param_u32(inst.param0 + i, idx)?;
            let dst = self.cb(inst.output_ptr + i, idx)?;
            let src = self.cb(pointer, idx)?;
            self.line(&format!("{dst} = {src};"));
        }
        Ok(())
    }

    fn emit_clip(&mut self, idx: usize, inst: &Inst) -> Result<(), GpuModelError> {
        self.check_elementwise_size(inst.data_size, idx)?;
        for i in 0..inst.data_size {
            let slot = self.cb(inst.input_ptr + i, idx)?;
            let lower = if inst.param0 == CLIP_BOUND_NONE {
                None
            } else {
                Some(f32_literal(self.param_f32(inst.param0 + i, idx)?, idx)?)
            };
            let upper = if inst.param1 == CLIP_BOUND_NONE {
                None
            } else {
                Some(f32_literal(self.param_f32(inst.param1 + i, idx)?, idx)?)
            };
            match (lower, upper) {
                (Some(lo), Some(hi)) => {
                    self.line(&format!("{slot} = clamp({slot}, {lo}, {hi});"));
                }
                (Some(lo), None) => self.line(&format!("{slot} = max({slot}, {lo});")),
                (None, Some(hi)) => self.line(&format!("{slot} = min({slot}, {hi});")),
                (None, None) => {}
            }
        }
        Ok(())
    }

    fn emit_elem_wise_buffers(
        &mut self,
        idx: usize,
        inst: &Inst,
        op: &str,
    ) -> Result<(), GpuModelError> {
        self.check_elementwise_size(inst.data_size, idx)?;
        let input_count = inst.param1;
        if input_count == 0 {
            return Err(inst_err(idx, "element-wise buffers op with zero inputs"));
        }
        let mut sources = Vec::with_capacity(input_count as usize);
        for j in 0..input_count {
            sources.push(self.param_u32(inst.param0 + j, idx)?);
        }
        for i in 0..inst.data_size {
            let mut terms = Vec::with_capacity(sources.len());
            for source in &sources {
                terms.push(self.cb(source + i, idx)?);
            }
            let dst = self.cb(inst.output_ptr + i, idx)?;
            self.line(&format!("{dst} = {};", balanced_tree(terms, op)));
        }
        Ok(())
    }

    fn emit_buffer_heads(
        &mut self,
        idx: usize,
        inst: &Inst,
        op: &str,
    ) -> Result<(), GpuModelError> {
        self.check_elementwise_size(inst.data_size, idx)?;
        let num_heads = inst.param1;
        if num_heads == 0 || !inst.data_size.is_multiple_of(num_heads) {
            return Err(inst_err(
                idx,
                format!(
                    "buffer-heads size {} is not divisible into {num_heads} heads",
                    inst.data_size
                ),
            ));
        }
        let head_dim = inst.data_size / num_heads;
        for i in 0..inst.data_size {
            let dst = self.cb(inst.output_ptr + i, idx)?;
            let data = self.cb(inst.input_ptr + i, idx)?;
            let head = self.cb(inst.param0 + i / head_dim, idx)?;
            self.line(&format!("{dst} = {data} {op} {head};"));
        }
        Ok(())
    }

    fn emit_reduce_sum(&mut self, idx: usize, inst: &Inst) -> Result<(), GpuModelError> {
        self.check_elementwise_size(inst.data_size, idx)?;
        let mut terms = Vec::with_capacity(inst.data_size as usize);
        for i in 0..inst.data_size {
            terms.push(self.cb(inst.input_ptr + i, idx)?);
        }
        let dst = self.cb(inst.output_ptr, idx)?;
        self.line(&format!("{dst} = {};", balanced_tree(terms, "+")));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::gpu::gpu_model::GpuModel;
    use crate::instruction_model_info::{
        DotInstructionInfo, ElemWiseAddInstructionInfo, InstructionInfo, InstructionModelInfo,
    };

    fn small_info() -> InstructionModelInfo {
        InstructionModelInfo {
            features: None,
            feature_size: Some(2),
            computation_buffer_sizes: vec![2, 2, 1],
            instructions: vec![
                InstructionInfo::Dot(DotInstructionInfo {
                    input: 0,
                    output: 1,
                    weights: 0,
                    activation: Some(Activation::Tanh),
                }),
                InstructionInfo::ElemWiseAdd(ElemWiseAddInstructionInfo {
                    input: 1,
                    parameters: 0,
                }),
                InstructionInfo::Dot(DotInstructionInfo {
                    input: 1,
                    output: 2,
                    weights: 1,
                    activation: None,
                }),
            ],
            weights: vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]], vec![vec![0.5, -0.5]]],
            bias: vec![vec![0.1, 0.2], vec![0.0]],
            parameters: Some(vec![vec![0.25, -1.5]]),
            maps: None,
            validation_data: None,
        }
    }

    #[test]
    fn specializes_small_model() {
        let model = GpuModel::from_info(&small_info()).unwrap();
        let wgsl = get_specialized_wgsl_lanes(model.as_f32_slice(), 4).unwrap();

        assert!(wgsl.contains("fn predict(model_offset: u32, lane: u32)"));
        assert!(wgsl.contains("var<workgroup> compute_buffers: array<f32, 20u>;"));
        // Buffer 1 starts at element 2; with 4 lanes that is offset 8.
        assert!(wgsl.contains("compute_buffers[8u + lane]"));
        // Weight reads stay relative to the runtime candidate offset.
        assert!(wgsl.contains("model_data[model_offset + "));
        // The dot activation resolves to a direct call, no dispatch switch.
        assert!(wgsl.contains("activation_tanh("));
        assert!(!wgsl.contains("read_instruction_u32"));
        // Constant params are inlined as literals.
        assert!(wgsl.contains("+ 0.25;"));
        assert!(wgsl.contains("+ -1.5;"));
    }

    #[test]
    fn rejects_bad_magic() {
        let model = GpuModel::from_info(&small_info()).unwrap();
        let mut blob = model.as_f32_slice().to_vec();
        blob[0] = f32::from_bits(0xDEADBEEF);
        let err = get_specialized_wgsl_lanes(&blob, 4).unwrap_err();
        assert!(matches!(err, GpuModelError::InvalidModelBinary { .. }));
    }

    #[test]
    fn rejects_truncated_blob() {
        let model = GpuModel::from_info(&small_info()).unwrap();
        let blob = &model.as_f32_slice()[..8];
        let err = get_specialized_wgsl_lanes(blob, 4).unwrap_err();
        assert!(matches!(err, GpuModelError::InvalidModelBinary { .. }));
    }
}
