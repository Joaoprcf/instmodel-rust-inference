// Test shader for GPU inference tests
// This file is included via include_str! for proper IDE highlighting

// Buffer bindings
@group(0) @binding(0) var<storage, read> model_data: array<f32>;
@group(0) @binding(1) var<storage, read> input_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>;

// INSTMODEL_WGSL_PLACEHOLDER - replaced at runtime with actual instmodel code

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Allocate compute buffer in function scope
    // COMPUTE_BUFFER_DECLARATION - replaced at runtime with actual size

    // Model offset (for single model, it's 0)
    let model_offset: u32 = 0u;

    // Copy input to compute buffer
    let feature_size = get_feature_size(model_offset);
    for (var i: u32 = 0u; i < feature_size; i = i + 1u) {
        compute_buffer[i] = input_data[i];
    }

    // Execute model
    predict(model_offset, &compute_buffer);

    // Copy output
    let output_start = get_output_start(model_offset);
    let out_size = get_output_size(model_offset);
    for (var i: u32 = 0u; i < out_size; i = i + 1u) {
        output_data[i] = compute_buffer[output_start + i];
    }
}
