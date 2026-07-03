// elem_wise_buffers.wgsl - N-ary element-wise combine across buffers.
// The input pointer list lives in the params region as bitcast-u32 absolute
// compute-buffer indexes; pointers_offset is model-relative.

fn execute_elem_wise_buffers_add(
    model_offset: u32,
    compute_buffer: ptr<function, array<f32, MAX_COMPUTE_BUFFER>>,
    output_ptr: u32,
    size: u32,
    pointers_offset: u32,
    input_count: u32
) {
    for (var i: u32 = 0u; i < size; i = i + 1u) {
        var acc: f32 = 0.0;
        for (var j: u32 = 0u; j < input_count; j = j + 1u) {
            let src = bitcast<u32>(model_data[model_offset + pointers_offset + j]);
            acc = acc + (*compute_buffer)[src + i];
        }
        (*compute_buffer)[output_ptr + i] = acc;
    }
}

fn execute_elem_wise_buffers_mul(
    model_offset: u32,
    compute_buffer: ptr<function, array<f32, MAX_COMPUTE_BUFFER>>,
    output_ptr: u32,
    size: u32,
    pointers_offset: u32,
    input_count: u32
) {
    for (var i: u32 = 0u; i < size; i = i + 1u) {
        var acc: f32 = 1.0;
        for (var j: u32 = 0u; j < input_count; j = j + 1u) {
            let src = bitcast<u32>(model_data[model_offset + pointers_offset + j]);
            acc = acc * (*compute_buffer)[src + i];
        }
        (*compute_buffer)[output_ptr + i] = acc;
    }
}
