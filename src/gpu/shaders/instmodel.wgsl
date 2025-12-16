// instmodel.wgsl - GPU-embeddable neural network inference
// Main entry point with predict() function
//
// IMPORTANT: This shader requires a global model_data binding to be defined:
//   @group(0) @binding(0) var<storage, read> model_data: array<f32>;
//
// Functions use model_offset parameter to index into the global buffer,
// allowing multiple models to be packed together.

// Header field indices (in f32/u32 units)
const HEADER_MAGIC: u32 = 0u;
const HEADER_VERSION: u32 = 1u;
const HEADER_FEATURE_SIZE: u32 = 2u;
const HEADER_OUTPUT_SIZE: u32 = 3u;
const HEADER_COMPUTE_BUFFER_SIZE: u32 = 4u;
const HEADER_INSTRUCTION_COUNT: u32 = 5u;
const HEADER_INSTRUCTIONS_OFFSET: u32 = 6u;
const HEADER_WEIGHTS_OFFSET: u32 = 7u;
const HEADER_WEIGHTS_COUNT: u32 = 8u;
const HEADER_PARAMS_OFFSET: u32 = 9u;
const HEADER_PARAMS_COUNT: u32 = 10u;
const HEADER_OUTPUT_START: u32 = 11u;
const HEADER_FULL_MODEL_SIZE: u32 = 12u;

// Instruction field offsets (within 8-f32 instruction block)
const INST_OPCODE: u32 = 0u;
const INST_INPUT_PTR: u32 = 1u;
const INST_OUTPUT_PTR: u32 = 2u;
const INST_DATA_SIZE: u32 = 3u;
const INST_PARAM0: u32 = 4u;
const INST_PARAM1: u32 = 5u;
const INST_PARAM2: u32 = 6u;

// Opcode constants
const OPCODE_DOT: u32 = 1u;
const OPCODE_ACTIVATION: u32 = 2u;
const OPCODE_ELEM_WISE_ADD: u32 = 3u;
const OPCODE_ELEM_WISE_MUL: u32 = 4u;
const OPCODE_COPY: u32 = 5u;

// Read u32 value from f32 array at model_offset + field_index (bitcast)
fn read_header_u32(model_offset: u32, field_index: u32) -> u32 {
    return bitcast<u32>(model_data[model_offset + field_index]);
}

// Read instruction field as u32
fn read_instruction_u32(model_offset: u32, inst_base: u32, field_offset: u32) -> u32 {
    return bitcast<u32>(model_data[model_offset + inst_base + field_offset]);
}

// Apply softmax in-place (numerically stable)
fn apply_softmax(
    compute_buffer: ptr<function, array<f32, MAX_COMPUTE_BUFFER>>,
    start: u32,
    size: u32
) {
    // Find max for numerical stability
    var max_val: f32 = (*compute_buffer)[start];
    for (var i: u32 = 1u; i < size; i = i + 1u) {
        max_val = max(max_val, (*compute_buffer)[start + i]);
    }

    // Compute exp(x - max) and sum
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < size; i = i + 1u) {
        let exp_val = exp((*compute_buffer)[start + i] - max_val);
        (*compute_buffer)[start + i] = exp_val;
        sum = sum + exp_val;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for (var i: u32 = 0u; i < size; i = i + 1u) {
        (*compute_buffer)[start + i] = (*compute_buffer)[start + i] * inv_sum;
    }
}

// Main predict function
// model_offset: offset into global model_data where this model starts
// compute_buffer: caller-provided computation buffer (function-local)
fn predict(
    model_offset: u32,
    compute_buffer: ptr<function, array<f32, MAX_COMPUTE_BUFFER>>
) {
    // Read header
    let instruction_count = read_header_u32(model_offset, HEADER_INSTRUCTION_COUNT);
    let instructions_offset = read_header_u32(model_offset, HEADER_INSTRUCTIONS_OFFSET);
    let weights_offset = read_header_u32(model_offset, HEADER_WEIGHTS_OFFSET);
    let params_offset = read_header_u32(model_offset, HEADER_PARAMS_OFFSET);

    // Execute instructions sequentially
    for (var inst_idx: u32 = 0u; inst_idx < instruction_count; inst_idx = inst_idx + 1u) {
        let inst_base = instructions_offset + inst_idx * 8u;

        let opcode = read_instruction_u32(model_offset, inst_base, INST_OPCODE);
        let input_ptr = read_instruction_u32(model_offset, inst_base, INST_INPUT_PTR);
        let output_ptr = read_instruction_u32(model_offset, inst_base, INST_OUTPUT_PTR);
        let data_size = read_instruction_u32(model_offset, inst_base, INST_DATA_SIZE);
        let param0 = read_instruction_u32(model_offset, inst_base, INST_PARAM0);
        let param1 = read_instruction_u32(model_offset, inst_base, INST_PARAM1);
        let param2 = read_instruction_u32(model_offset, inst_base, INST_PARAM2);

        switch opcode {
            case OPCODE_DOT: {
                // param0 = weights offset (relative), param1 = input_size, param2 = activation
                execute_dot(
                    model_offset,
                    compute_buffer,
                    input_ptr,
                    output_ptr,
                    data_size,
                    weights_offset + param0,
                    param1,
                    param2
                );
            }
            case OPCODE_ACTIVATION: {
                // param2 = activation type
                execute_activation(
                    compute_buffer,
                    input_ptr,
                    data_size,
                    param2
                );
            }
            case OPCODE_ELEM_WISE_ADD: {
                // param0 = params offset (relative)
                execute_elem_wise_add(
                    model_offset,
                    compute_buffer,
                    input_ptr,
                    data_size,
                    params_offset + param0
                );
            }
            case OPCODE_ELEM_WISE_MUL: {
                // param0 = params offset (relative)
                execute_elem_wise_mul(
                    model_offset,
                    compute_buffer,
                    input_ptr,
                    data_size,
                    params_offset + param0
                );
            }
            case OPCODE_COPY: {
                execute_copy(
                    compute_buffer,
                    input_ptr,
                    output_ptr,
                    data_size
                );
            }
            default: {
                // Unknown opcode - skip
            }
        }
    }
}

// Helper functions to read model metadata
// model_offset: offset into global model_data where this model starts

fn get_feature_size(model_offset: u32) -> u32 {
    return read_header_u32(model_offset, HEADER_FEATURE_SIZE);
}

fn get_output_size(model_offset: u32) -> u32 {
    return read_header_u32(model_offset, HEADER_OUTPUT_SIZE);
}

fn get_compute_buffer_size(model_offset: u32) -> u32 {
    return read_header_u32(model_offset, HEADER_COMPUTE_BUFFER_SIZE);
}

fn get_output_start(model_offset: u32) -> u32 {
    return read_header_u32(model_offset, HEADER_OUTPUT_START);
}

fn get_full_model_size(model_offset: u32) -> u32 {
    return read_header_u32(model_offset, HEADER_FULL_MODEL_SIZE);
}
