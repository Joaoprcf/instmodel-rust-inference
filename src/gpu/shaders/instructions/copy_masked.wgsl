// copy_masked.wgsl - Gather: output[i] = compute_buffer[pointers[i]]
// The pointer list lives in the params region as bitcast-u32 absolute
// compute-buffer indexes; pointers_offset is model-relative.

fn execute_copy_masked(
    model_offset: u32,
    compute_buffer: ptr<function, array<f32, MAX_COMPUTE_BUFFER>>,
    output_ptr: u32,
    count: u32,
    pointers_offset: u32
) {
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let src = bitcast<u32>(model_data[model_offset + pointers_offset + i]);
        (*compute_buffer)[output_ptr + i] = (*compute_buffer)[src];
    }
}
