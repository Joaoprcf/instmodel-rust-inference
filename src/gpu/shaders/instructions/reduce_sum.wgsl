// reduce_sum.wgsl - Sum a buffer segment into a single scalar.

fn execute_reduce_sum(
    compute_buffer: ptr<function, array<f32, MAX_COMPUTE_BUFFER>>,
    input_ptr: u32,
    output_ptr: u32,
    size: u32
) {
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < size; i = i + 1u) {
        sum = sum + (*compute_buffer)[input_ptr + i];
    }
    (*compute_buffer)[output_ptr] = sum;
}
