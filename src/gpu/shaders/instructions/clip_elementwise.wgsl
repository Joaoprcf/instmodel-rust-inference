// clip_elementwise.wgsl - In-place clamp against per-element bound vectors
// in the params region: lower bound first, then upper (matches CPU order).
// min_rel/max_rel are params-relative offsets; CLIP_BOUND_NONE marks an
// absent bound and must be checked before resolving the offset.

const CLIP_BOUND_NONE: u32 = 0xffffffffu;

fn execute_clip_elementwise(
    model_offset: u32,
    compute_buffer: ptr<function, array<f32, MAX_COMPUTE_BUFFER>>,
    ptr_start: u32,
    size: u32,
    params_offset: u32,
    min_rel: u32,
    max_rel: u32
) {
    for (var i: u32 = 0u; i < size; i = i + 1u) {
        var value = (*compute_buffer)[ptr_start + i];
        if min_rel != CLIP_BOUND_NONE {
            value = max(value, model_data[model_offset + params_offset + min_rel + i]);
        }
        if max_rel != CLIP_BOUND_NONE {
            value = min(value, model_data[model_offset + params_offset + max_rel + i]);
        }
        (*compute_buffer)[ptr_start + i] = value;
    }
}
