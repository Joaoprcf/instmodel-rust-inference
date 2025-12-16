// activation.wgsl - Standalone activation on buffer range

fn execute_activation(
    compute_buffer: ptr<function, array<f32, MAX_COMPUTE_BUFFER>>,
    ptr_start: u32,
    size: u32,
    activation_type: u32
) {
    if activation_type == ACTIVATION_SOFTMAX {
        apply_softmax(compute_buffer, ptr_start, size);
    } else if activation_type != ACTIVATION_NONE {
        for (var i: u32 = 0u; i < size; i = i + 1u) {
            let idx = ptr_start + i;
            (*compute_buffer)[idx] = apply_activation_single((*compute_buffer)[idx], activation_type);
        }
    }
}
