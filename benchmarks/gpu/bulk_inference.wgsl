// Bulk inference benchmark shader
// Each thread processes multiple inputs sequentially

@group(0) @binding(0) var<storage, read> model_data: array<f32>;
@group(0) @binding(1) var<storage, read> all_inputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> all_outputs: array<f32>;
@group(0) @binding(3) var<uniform> params: BenchmarkParams;

struct BenchmarkParams {
    total_inputs: u32,
    feature_size: u32,
    output_size: u32,
    inputs_per_thread: u32,
}

// INSTMODEL_WGSL_PLACEHOLDER

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let thread_id = global_id.x;
    // Calculate total threads for strided access (workgroup_size is 256)
    let total_threads = num_workgroups.x * 256u;

    // Allocate compute buffer in function scope
    // COMPUTE_BUFFER_DECLARATION

    let model_offset: u32 = 0u;
    let feature_size = params.feature_size;
    let output_size = params.output_size;
    let output_start = get_output_start(model_offset);

    // Pre-calculate unroll boundaries
    let feature_unroll_end = feature_size & ~3u;  // Round down to multiple of 4
    let output_unroll_end = output_size & ~3u;

    // Use strided loop for better memory locality and load balancing
    for (var input_idx: u32 = thread_id; input_idx < params.total_inputs; input_idx = input_idx + total_threads) {
        let input_offset = input_idx * feature_size;
        let output_offset = input_idx * output_size;

        // Copy input to compute buffer - 4-way unrolled
        var i: u32 = 0u;
        for (; i < feature_unroll_end; i = i + 4u) {
            compute_buffer[i] = all_inputs[input_offset + i];
            compute_buffer[i + 1u] = all_inputs[input_offset + i + 1u];
            compute_buffer[i + 2u] = all_inputs[input_offset + i + 2u];
            compute_buffer[i + 3u] = all_inputs[input_offset + i + 3u];
        }
        // Handle remainder
        for (; i < feature_size; i = i + 1u) {
            compute_buffer[i] = all_inputs[input_offset + i];
        }

        // Run inference
        predict(model_offset, &compute_buffer);

        // Copy output - 4-way unrolled
        var j: u32 = 0u;
        for (; j < output_unroll_end; j = j + 4u) {
            all_outputs[output_offset + j] = compute_buffer[output_start + j];
            all_outputs[output_offset + j + 1u] = compute_buffer[output_start + j + 1u];
            all_outputs[output_offset + j + 2u] = compute_buffer[output_start + j + 2u];
            all_outputs[output_offset + j + 3u] = compute_buffer[output_start + j + 3u];
        }
        // Handle remainder
        for (; j < output_size; j = j + 1u) {
            all_outputs[output_offset + j] = compute_buffer[output_start + j];
        }
    }
}
