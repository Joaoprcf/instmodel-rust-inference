//! WGSL shader sources for GPU neural network inference.

/// Maximum compute buffer size (in f32s) - can be overridden by user
pub const DEFAULT_MAX_COMPUTE_BUFFER: u32 = 65536;

/// The composed function bodies shared by both shader variants.
fn function_sources() -> String {
    format!(
        "{activations}\n\n{dot}\n\n{activation_inst}\n\n{elem_wise_add}\n\n\
         {elem_wise_mul}\n\n{copy}\n\n{copy_masked}\n\n{clip_elementwise}\n\n\
         {elem_wise_buffers}\n\n{buffer_heads}\n\n{reduce_sum}\n\n{instmodel}",
        activations = include_str!("activations.wgsl"),
        dot = include_str!("instructions/dot.wgsl"),
        activation_inst = include_str!("instructions/activation.wgsl"),
        elem_wise_add = include_str!("instructions/elem_wise_add.wgsl"),
        elem_wise_mul = include_str!("instructions/elem_wise_mul.wgsl"),
        copy = include_str!("instructions/copy.wgsl"),
        copy_masked = include_str!("instructions/copy_masked.wgsl"),
        clip_elementwise = include_str!("instructions/clip_elementwise.wgsl"),
        elem_wise_buffers = include_str!("instructions/elem_wise_buffers.wgsl"),
        buffer_heads = include_str!("instructions/buffer_heads.wgsl"),
        reduce_sum = include_str!("instructions/reduce_sum.wgsl"),
        instmodel = include_str!("instmodel.wgsl"),
    )
}

/// Get the composed WGSL shader source for instmodel inference.
///
/// This returns a complete shader that can be imported/included in other WGSL code.
/// The shader provides the `predict()` function and helper functions.
///
/// # Arguments
/// * `max_compute_buffer` - Maximum size of the compute buffer in f32s
pub fn get_instmodel_wgsl(max_compute_buffer: u32) -> String {
    format!(
        "// GPU Neural Network Inference Library\n\
         // Generated with max_compute_buffer = {max_compute_buffer}\n\n\
         const MAX_COMPUTE_BUFFER: u32 = {max_compute_buffer}u;\n\n\
         {functions}\n",
        functions = function_sources(),
    )
}

/// Get just the function definitions without any bindings.
/// This is useful when embedding the inference functions into an existing shader.
pub fn get_instmodel_functions_wgsl(max_compute_buffer: u32) -> String {
    get_instmodel_wgsl(max_compute_buffer)
}

/// Get the lane-sliced shared-memory WGSL shader variant, for kernels where
/// each thread runs a long sequential episode (many `predict` calls with a
/// live scratch buffer).
///
/// Instead of a caller-provided `ptr<function>` scratch buffer per thread —
/// which drivers place in DRAM-backed local memory — every function takes the
/// thread's `lane` index (`local_invocation_id.x`) and works on a disjoint
/// slice of one `var<workgroup>` block declared by this source. Element `i`
/// of lane `l`'s buffer lives at `compute_buffers[i * CB_LANES + l]`; the
/// interleaved layout puts warp-uniform buffer indexes on consecutive
/// addresses, so accesses are bank-conflict free. Slices are disjoint, so no
/// barriers are required.
///
/// Requirements for the embedding kernel:
/// * `@workgroup_size(CB_LANES)` with `lanes` threads per workgroup and
///   `lane = local_invocation_id.x` (the `CB_LANES` constant is emitted by
///   this source).
/// * The device must allow `lanes * max_compute_buffer * 4` bytes of
///   workgroup storage (`wgpu::Limits::max_compute_workgroup_storage_size`).
/// * Not combinable with [`get_instmodel_wgsl`] in the same module: both
///   variants define `predict` and the same helper names.
pub fn get_instmodel_wgsl_lanes(max_compute_buffer: u32, lanes: u32) -> String {
    format!(
        "// GPU Neural Network Inference Library (lane-sliced shared-memory variant)\n\
         // Generated with max_compute_buffer = {max_compute_buffer}, lanes = {lanes}\n\n\
         const MAX_COMPUTE_BUFFER: u32 = {max_compute_buffer}u;\n\
         const CB_LANES: u32 = {lanes}u;\n\
         var<workgroup> compute_buffers: array<f32, {total}u>;\n\n\
         fn cb_index(buffer_index: u32, lane: u32) -> u32 {{\n\
             return buffer_index * CB_LANES + lane;\n\
         }}\n\n\
         {functions}\n",
        total = max_compute_buffer * lanes,
        functions = rewrite_to_lane_form(&function_sources()),
    )
}

/// Rewrites the per-thread shader sources into the lane-sliced form:
/// `compute_buffer` pointer parameters become a `lane: u32` parameter and
/// every `(*compute_buffer)[expr]` access becomes
/// `compute_buffers[cb_index(expr, lane)]`.
fn rewrite_to_lane_form(source: &str) -> String {
    const ACCESS: &str = "(*compute_buffer)[";
    let source = source.replace(
        "compute_buffer: ptr<function, array<f32, MAX_COMPUTE_BUFFER>>",
        "lane: u32",
    );

    let mut out = String::with_capacity(source.len());
    let mut rest = source.as_str();
    while let Some(pos) = rest.find(ACCESS) {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + ACCESS.len()..];
        let mut depth = 1usize;
        let mut close = None;
        for (i, c) in after.char_indices() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        match close {
            Some(end) => {
                out.push_str("__CB_ARRAY__[cb_index(");
                out.push_str(&after[..end]);
                out.push_str(", lane)]");
                rest = &after[end + 1..];
            }
            None => {
                // Unbalanced brackets cannot happen with the bundled sources;
                // keep the remainder untouched rather than dropping it.
                out.push_str(&rest[pos..]);
                rest = "";
            }
        }
    }
    out.push_str(rest);

    // Remaining `compute_buffer` occurrences are call-site arguments (and
    // comments); the placeholder keeps them from matching the array name.
    out.replace("compute_buffer", "lane")
        .replace("__CB_ARRAY__", "compute_buffers")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_instmodel_wgsl_compiles() {
        let wgsl = get_instmodel_wgsl(4096);
        assert!(wgsl.contains("fn predict("));
        assert!(wgsl.contains("fn get_feature_size("));
        assert!(wgsl.contains("fn get_output_start("));
        assert!(wgsl.contains("fn get_full_model_size("));
        assert!(wgsl.contains("MAX_COMPUTE_BUFFER"));
    }

    #[test]
    fn test_max_compute_buffer_substitution() {
        let wgsl_small = get_instmodel_wgsl(1024);
        let wgsl_large = get_instmodel_wgsl(65536);

        assert!(wgsl_small.contains("const MAX_COMPUTE_BUFFER: u32 = 1024u;"));
        assert!(wgsl_large.contains("const MAX_COMPUTE_BUFFER: u32 = 65536u;"));
    }

    #[test]
    fn test_lanes_variant_shape() {
        let wgsl = get_instmodel_wgsl_lanes(128, 32);
        assert!(wgsl.contains("const MAX_COMPUTE_BUFFER: u32 = 128u;"));
        assert!(wgsl.contains("const CB_LANES: u32 = 32u;"));
        assert!(wgsl.contains("var<workgroup> compute_buffers: array<f32, 4096u>;"));
        assert!(wgsl.contains("fn predict(\n    model_offset: u32,\n    lane: u32\n)"));
    }

    #[test]
    fn test_lanes_variant_has_no_private_buffer_form() {
        let wgsl = get_instmodel_wgsl_lanes(64, 16);
        assert!(!wgsl.contains("ptr<function"));
        assert!(!wgsl.contains("(*compute_buffer)"));
        assert!(!wgsl.contains("(*lane)"));
        assert!(!wgsl.contains("__CB_ARRAY__"));
    }

    #[test]
    fn test_lanes_variant_rewrites_accesses_and_calls() {
        let wgsl = get_instmodel_wgsl_lanes(64, 16);
        // Dot-product read: index expression wrapped, lane threaded through.
        assert!(wgsl.contains("compute_buffers[cb_index(input_ptr + i, lane)]"));
        // Write side of an instruction.
        assert!(wgsl.contains("compute_buffers[cb_index(output_ptr + out_idx, lane)] = sum;"));
        // Buffer pointer argument at call sites becomes the lane index.
        assert!(wgsl.contains("apply_softmax(lane, output_ptr, output_size);"));
    }
}
