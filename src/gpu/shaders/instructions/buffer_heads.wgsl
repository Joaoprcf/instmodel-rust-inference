// buffer_heads.wgsl - Broadcast a heads buffer across equal segments of a
// data buffer: out[i] = data[i] (op) heads[i / head_dim] with
// head_dim = data_size / num_heads (divisibility validated at build time).

fn execute_multiply_buffer_heads(
    compute_buffer: ptr<function, array<f32, MAX_COMPUTE_BUFFER>>,
    data_ptr: u32,
    output_ptr: u32,
    data_size: u32,
    heads_ptr: u32,
    num_heads: u32
) {
    let head_dim = data_size / num_heads;
    for (var i: u32 = 0u; i < data_size; i = i + 1u) {
        (*compute_buffer)[output_ptr + i] =
            (*compute_buffer)[data_ptr + i] * (*compute_buffer)[heads_ptr + i / head_dim];
    }
}

fn execute_add_buffer_heads(
    compute_buffer: ptr<function, array<f32, MAX_COMPUTE_BUFFER>>,
    data_ptr: u32,
    output_ptr: u32,
    data_size: u32,
    heads_ptr: u32,
    num_heads: u32
) {
    let head_dim = data_size / num_heads;
    for (var i: u32 = 0u; i < data_size; i = i + 1u) {
        (*compute_buffer)[output_ptr + i] =
            (*compute_buffer)[data_ptr + i] + (*compute_buffer)[heads_ptr + i / head_dim];
    }
}
