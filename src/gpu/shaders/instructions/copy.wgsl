// copy.wgsl - Direct buffer copy

fn execute_copy(
    compute_buffer: ptr<function, array<f32, MAX_COMPUTE_BUFFER>>,
    src_ptr: u32,
    dst_ptr: u32,
    size: u32
) {
    // Handle overlapping regions by copying in appropriate direction
    if dst_ptr > src_ptr && dst_ptr < src_ptr + size {
        // Copy backwards to avoid overwriting source
        for (var i: u32 = size; i > 0u; i = i - 1u) {
            (*compute_buffer)[dst_ptr + i - 1u] = (*compute_buffer)[src_ptr + i - 1u];
        }
    } else {
        // Copy forwards
        for (var i: u32 = 0u; i < size; i = i + 1u) {
            (*compute_buffer)[dst_ptr + i] = (*compute_buffer)[src_ptr + i];
        }
    }
}
