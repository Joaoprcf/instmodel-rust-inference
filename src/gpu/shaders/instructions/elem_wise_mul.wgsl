// elem_wise_mul.wgsl - buffer[i] *= params[i]
// Accesses global model_data buffer using model_offset

fn execute_elem_wise_mul(
    model_offset: u32,
    compute_buffer: ptr<function, array<f32, MAX_COMPUTE_BUFFER>>,
    ptr_start: u32,
    size: u32,
    params_offset: u32
) {
    for (var i: u32 = 0u; i < size; i = i + 1u) {
        (*compute_buffer)[ptr_start + i] = (*compute_buffer)[ptr_start + i] * model_data[model_offset + params_offset + i];
    }
}
