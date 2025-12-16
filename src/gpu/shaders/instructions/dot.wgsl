// dot.wgsl - Matrix-vector multiply + bias + optional activation
// Accesses global model_data buffer using model_offset

fn execute_dot(
    model_offset: u32,
    compute_buffer: ptr<function, array<f32, MAX_COMPUTE_BUFFER>>,
    input_ptr: u32,
    output_ptr: u32,
    output_size: u32,
    weights_offset: u32,
    input_size: u32,
    activation_type: u32
) {
    // Weights layout: [weights_flattened_row_major][bias]
    // weights[output_idx * input_size + input_idx]
    // bias at weights_offset + output_size * input_size + output_idx

    let bias_offset = weights_offset + output_size * input_size;

    for (var out_idx: u32 = 0u; out_idx < output_size; out_idx = out_idx + 1u) {
        var sum: f32 = 0.0;
        let row_offset = weights_offset + out_idx * input_size;

        // 4-way unrolled dot product for better performance
        var i: u32 = 0u;
        let unroll_end = input_size & ~3u;  // Round down to multiple of 4

        for (; i < unroll_end; i = i + 4u) {
            sum = sum + model_data[model_offset + row_offset + i] * (*compute_buffer)[input_ptr + i];
            sum = sum + model_data[model_offset + row_offset + i + 1u] * (*compute_buffer)[input_ptr + i + 1u];
            sum = sum + model_data[model_offset + row_offset + i + 2u] * (*compute_buffer)[input_ptr + i + 2u];
            sum = sum + model_data[model_offset + row_offset + i + 3u] * (*compute_buffer)[input_ptr + i + 3u];
        }

        // Handle remainder
        for (; i < input_size; i = i + 1u) {
            sum = sum + model_data[model_offset + row_offset + i] * (*compute_buffer)[input_ptr + i];
        }

        // Add bias
        sum = sum + model_data[model_offset + bias_offset + out_idx];

        (*compute_buffer)[output_ptr + out_idx] = sum;
    }

    // Apply activation if specified (not softmax - handled separately)
    if activation_type != ACTIVATION_NONE && activation_type != ACTIVATION_SOFTMAX {
        for (var i: u32 = 0u; i < output_size; i = i + 1u) {
            let idx = output_ptr + i;
            (*compute_buffer)[idx] = apply_activation_single((*compute_buffer)[idx], activation_type);
        }
    } else if activation_type == ACTIVATION_SOFTMAX {
        // Apply softmax in-place
        apply_softmax(compute_buffer, output_ptr, output_size);
    }
}
