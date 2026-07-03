// population_entry.wgsl - Population evaluation entry point.
// One thread per (candidate, sample): thread idx maps to candidate
// idx / BATCH_SIZE and sample idx % BATCH_SIZE, model data is read at
// candidate * MODEL_STRIDE, and outputs land candidate-major at
// (candidate * BATCH_SIZE + sample) * out_size.

@group(0) @binding(0) var<storage, read> model_data: array<f32>;
@group(0) @binding(1) var<storage, read> input_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>;

// POPULATION_DIMS - replaced at build time with N_CANDIDATES/BATCH_SIZE/MODEL_STRIDE constants

// INSTMODEL_FUNCTIONS - replaced at build time with the generated inference library

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= N_CANDIDATES * BATCH_SIZE {
        return;
    }
    let candidate = idx / BATCH_SIZE;
    let sample = idx - candidate * BATCH_SIZE;
    let model_offset = candidate * MODEL_STRIDE;

    var compute_buffer: array<f32, MAX_COMPUTE_BUFFER>;
    let feature_size = get_feature_size(model_offset);
    let input_base = sample * feature_size;
    for (var i: u32 = 0u; i < feature_size; i = i + 1u) {
        compute_buffer[i] = input_data[input_base + i];
    }

    predict(model_offset, &compute_buffer);

    let out_size = get_output_size(model_offset);
    let output_start = get_output_start(model_offset);
    let out_base = (candidate * BATCH_SIZE + sample) * out_size;
    for (var i: u32 = 0u; i < out_size; i = i + 1u) {
        output_data[out_base + i] = compute_buffer[output_start + i];
    }
}
