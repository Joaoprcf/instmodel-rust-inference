//! GPU instruction encoding for neural network inference.

use crate::activation::Activation;

/// Opcode constants matching WGSL definitions.
pub mod opcodes {
    pub const DOT: u32 = 0x01;
    pub const ACTIVATION: u32 = 0x02;
    pub const ELEM_WISE_ADD: u32 = 0x03;
    pub const ELEM_WISE_MUL: u32 = 0x04;
    pub const COPY: u32 = 0x05;
    pub const COPY_MASKED: u32 = 0x06;
    pub const CLIP_ELEMENTWISE: u32 = 0x07;
    pub const ELEM_WISE_BUFFERS_ADD: u32 = 0x08;
    pub const ELEM_WISE_BUFFERS_MUL: u32 = 0x09;
    pub const MULTIPLY_BUFFER_HEADS: u32 = 0x0A;
    pub const ADD_BUFFER_HEADS: u32 = 0x0B;
    pub const REDUCE_SUM: u32 = 0x0C;
}

/// Sentinel for an absent bound in CLIP_ELEMENTWISE `param0`/`param1`.
///
/// Params-region offsets start at 0, so absence needs an out-of-band marker;
/// the WGSL interpreter checks against this value before touching the offset.
pub const CLIP_BOUND_NONE: u32 = u32::MAX;

/// Activation type constants matching WGSL definitions.
pub mod activation_types {
    pub const NONE: u32 = 0x00;
    pub const RELU: u32 = 0x01;
    pub const SIGMOID: u32 = 0x02;
    pub const SOFTMAX: u32 = 0x03;
    pub const TANH: u32 = 0x04;
    pub const SQRT: u32 = 0x05;
    pub const LOG: u32 = 0x06;
    pub const LOG10: u32 = 0x07;
    pub const INVERSE: u32 = 0x08;
    pub const GELU: u32 = 0x09;
    pub const SOFTPLUS: u32 = 0x0A;
    pub const EXP: u32 = 0x0B;
    pub const SIGN: u32 = 0x0C;
}

/// Convert Activation enum to GPU activation type.
pub fn activation_to_gpu(activation: Option<Activation>) -> u32 {
    match activation {
        None => activation_types::NONE,
        Some(Activation::Relu) => activation_types::RELU,
        Some(Activation::Sigmoid) => activation_types::SIGMOID,
        Some(Activation::Softmax) => activation_types::SOFTMAX,
        Some(Activation::Tanh) => activation_types::TANH,
        Some(Activation::Sqrt) => activation_types::SQRT,
        Some(Activation::Log) => activation_types::LOG,
        Some(Activation::Log10) => activation_types::LOG10,
        Some(Activation::Inverse) => activation_types::INVERSE,
        Some(Activation::Gelu) => activation_types::GELU,
        Some(Activation::Softplus) => activation_types::SOFTPLUS,
        Some(Activation::Exp) => activation_types::EXP,
        Some(Activation::Sign) => activation_types::SIGN,
    }
}

/// Encoded GPU instruction (32 bytes / 8 u32s).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuInstruction {
    pub opcode: u32,
    pub input_ptr: u32,
    pub output_ptr: u32,
    pub data_size: u32,
    pub param0: u32,
    pub param1: u32,
    pub param2: u32,
    pub reserved: u32,
}

impl GpuInstruction {
    pub const SIZE_BYTES: usize = 32;
    pub const SIZE_U32S: usize = 8;

    /// Create a DOT instruction.
    pub fn dot(
        input_ptr: u32,
        output_ptr: u32,
        output_size: u32,
        weights_offset: u32,
        input_size: u32,
        activation: Option<Activation>,
    ) -> Self {
        Self {
            opcode: opcodes::DOT,
            input_ptr,
            output_ptr,
            data_size: output_size,
            param0: weights_offset,
            param1: input_size,
            param2: activation_to_gpu(activation),
            reserved: 0,
        }
    }

    /// Create an ACTIVATION instruction.
    pub fn activation(ptr: u32, size: u32, activation: Activation) -> Self {
        Self {
            opcode: opcodes::ACTIVATION,
            input_ptr: ptr,
            output_ptr: ptr,
            data_size: size,
            param0: 0,
            param1: 0,
            param2: activation_to_gpu(Some(activation)),
            reserved: 0,
        }
    }

    /// Create an ELEM_WISE_ADD instruction.
    pub fn elem_wise_add(ptr: u32, size: u32, params_offset: u32) -> Self {
        Self {
            opcode: opcodes::ELEM_WISE_ADD,
            input_ptr: ptr,
            output_ptr: ptr,
            data_size: size,
            param0: params_offset,
            param1: 0,
            param2: 0,
            reserved: 0,
        }
    }

    /// Create an ELEM_WISE_MUL instruction.
    pub fn elem_wise_mul(ptr: u32, size: u32, params_offset: u32) -> Self {
        Self {
            opcode: opcodes::ELEM_WISE_MUL,
            input_ptr: ptr,
            output_ptr: ptr,
            data_size: size,
            param0: params_offset,
            param1: 0,
            param2: 0,
            reserved: 0,
        }
    }

    /// Create a COPY instruction.
    pub fn copy(src_ptr: u32, dst_ptr: u32, size: u32) -> Self {
        Self {
            opcode: opcodes::COPY,
            input_ptr: src_ptr,
            output_ptr: dst_ptr,
            data_size: size,
            param0: 0,
            param1: 0,
            param2: 0,
            reserved: 0,
        }
    }

    /// Create a COPY_MASKED instruction.
    ///
    /// `pointers_offset` locates a list of `count` bitcast-u32 absolute
    /// compute-buffer indexes in the params region; the interpreter gathers
    /// `output[i] = compute_buffer[pointers[i]]`.
    pub fn copy_masked(pointers_offset: u32, count: u32, output_ptr: u32) -> Self {
        Self {
            opcode: opcodes::COPY_MASKED,
            input_ptr: 0,
            output_ptr,
            data_size: count,
            param0: pointers_offset,
            param1: 0,
            param2: 0,
            reserved: 0,
        }
    }

    /// Create a CLIP_ELEMENTWISE instruction (in place).
    ///
    /// `min_offset`/`max_offset` are params-region offsets of the per-element
    /// bound vectors; `None` encodes as [`CLIP_BOUND_NONE`].
    pub fn clip_elementwise(
        ptr: u32,
        size: u32,
        min_offset: Option<u32>,
        max_offset: Option<u32>,
    ) -> Self {
        Self {
            opcode: opcodes::CLIP_ELEMENTWISE,
            input_ptr: ptr,
            output_ptr: ptr,
            data_size: size,
            param0: min_offset.unwrap_or(CLIP_BOUND_NONE),
            param1: max_offset.unwrap_or(CLIP_BOUND_NONE),
            param2: 0,
            reserved: 0,
        }
    }

    /// Create an ELEM_WISE_BUFFERS_ADD instruction.
    ///
    /// `pointers_offset` locates a list of `input_count` bitcast-u32 absolute
    /// compute-buffer indexes in the params region — one per input buffer.
    pub fn elem_wise_buffers_add(
        output_ptr: u32,
        size: u32,
        pointers_offset: u32,
        input_count: u32,
    ) -> Self {
        Self {
            opcode: opcodes::ELEM_WISE_BUFFERS_ADD,
            input_ptr: 0,
            output_ptr,
            data_size: size,
            param0: pointers_offset,
            param1: input_count,
            param2: 0,
            reserved: 0,
        }
    }

    /// Create an ELEM_WISE_BUFFERS_MUL instruction (same encoding as the
    /// add variant).
    pub fn elem_wise_buffers_mul(
        output_ptr: u32,
        size: u32,
        pointers_offset: u32,
        input_count: u32,
    ) -> Self {
        Self {
            opcode: opcodes::ELEM_WISE_BUFFERS_MUL,
            input_ptr: 0,
            output_ptr,
            data_size: size,
            param0: pointers_offset,
            param1: input_count,
            param2: 0,
            reserved: 0,
        }
    }

    /// Create a MULTIPLY_BUFFER_HEADS instruction.
    ///
    /// `out[i] = data[i] * heads[i / (data_size / num_heads)]`.
    pub fn multiply_buffer_heads(
        data_ptr: u32,
        output_ptr: u32,
        data_size: u32,
        heads_ptr: u32,
        num_heads: u32,
    ) -> Self {
        Self {
            opcode: opcodes::MULTIPLY_BUFFER_HEADS,
            input_ptr: data_ptr,
            output_ptr,
            data_size,
            param0: heads_ptr,
            param1: num_heads,
            param2: 0,
            reserved: 0,
        }
    }

    /// Create an ADD_BUFFER_HEADS instruction (same encoding as multiply).
    pub fn add_buffer_heads(
        data_ptr: u32,
        output_ptr: u32,
        data_size: u32,
        heads_ptr: u32,
        num_heads: u32,
    ) -> Self {
        Self {
            opcode: opcodes::ADD_BUFFER_HEADS,
            input_ptr: data_ptr,
            output_ptr,
            data_size,
            param0: heads_ptr,
            param1: num_heads,
            param2: 0,
            reserved: 0,
        }
    }

    /// Create a REDUCE_SUM instruction: sums `input_size` elements at
    /// `input_ptr` into the single scalar at `output_ptr`.
    pub fn reduce_sum(input_ptr: u32, output_ptr: u32, input_size: u32) -> Self {
        Self {
            opcode: opcodes::REDUCE_SUM,
            input_ptr,
            output_ptr,
            data_size: input_size,
            param0: 0,
            param1: 0,
            param2: 0,
            reserved: 0,
        }
    }

    /// Convert to f32 array (for packing into single f32 buffer).
    /// Each u32 field is bitcast to f32.
    pub fn to_f32_array(&self) -> [f32; Self::SIZE_U32S] {
        [
            f32::from_bits(self.opcode),
            f32::from_bits(self.input_ptr),
            f32::from_bits(self.output_ptr),
            f32::from_bits(self.data_size),
            f32::from_bits(self.param0),
            f32::from_bits(self.param1),
            f32::from_bits(self.param2),
            f32::from_bits(self.reserved),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_instruction_size() {
        assert_eq!(
            std::mem::size_of::<GpuInstruction>(),
            GpuInstruction::SIZE_BYTES
        );
    }

    #[test]
    fn test_dot_instruction() {
        let inst = GpuInstruction::dot(0, 10, 5, 100, 8, Some(Activation::Relu));
        assert_eq!(inst.opcode, opcodes::DOT);
        assert_eq!(inst.input_ptr, 0);
        assert_eq!(inst.output_ptr, 10);
        assert_eq!(inst.data_size, 5);
        assert_eq!(inst.param0, 100);
        assert_eq!(inst.param1, 8);
        assert_eq!(inst.param2, activation_types::RELU);
    }

    #[test]
    fn test_activation_to_gpu() {
        assert_eq!(activation_to_gpu(None), activation_types::NONE);
        assert_eq!(
            activation_to_gpu(Some(Activation::Relu)),
            activation_types::RELU
        );
        assert_eq!(
            activation_to_gpu(Some(Activation::Sigmoid)),
            activation_types::SIGMOID
        );
        assert_eq!(
            activation_to_gpu(Some(Activation::Softmax)),
            activation_types::SOFTMAX
        );
        assert_eq!(
            activation_to_gpu(Some(Activation::Tanh)),
            activation_types::TANH
        );
        assert_eq!(
            activation_to_gpu(Some(Activation::Sqrt)),
            activation_types::SQRT
        );
        assert_eq!(
            activation_to_gpu(Some(Activation::Log)),
            activation_types::LOG
        );
        assert_eq!(
            activation_to_gpu(Some(Activation::Log10)),
            activation_types::LOG10
        );
        assert_eq!(
            activation_to_gpu(Some(Activation::Inverse)),
            activation_types::INVERSE
        );
        assert_eq!(
            activation_to_gpu(Some(Activation::Gelu)),
            activation_types::GELU
        );
        assert_eq!(
            activation_to_gpu(Some(Activation::Softplus)),
            activation_types::SOFTPLUS
        );
    }

    #[test]
    fn test_copy_masked_instruction() {
        let inst = GpuInstruction::copy_masked(7, 5, 12);
        assert_eq!(inst.opcode, opcodes::COPY_MASKED);
        assert_eq!(inst.output_ptr, 12);
        assert_eq!(inst.data_size, 5);
        assert_eq!(inst.param0, 7);
    }

    #[test]
    fn test_clip_elementwise_bound_encoding() {
        let both = GpuInstruction::clip_elementwise(3, 4, Some(0), Some(4));
        assert_eq!(both.opcode, opcodes::CLIP_ELEMENTWISE);
        assert_eq!(both.input_ptr, 3);
        assert_eq!(both.output_ptr, 3);
        assert_eq!(both.param0, 0);
        assert_eq!(both.param1, 4);

        let min_only = GpuInstruction::clip_elementwise(3, 4, Some(2), None);
        assert_eq!(min_only.param0, 2);
        assert_eq!(min_only.param1, CLIP_BOUND_NONE);

        let max_only = GpuInstruction::clip_elementwise(3, 4, None, Some(2));
        assert_eq!(max_only.param0, CLIP_BOUND_NONE);
        assert_eq!(max_only.param1, 2);
    }

    #[test]
    fn test_elem_wise_buffers_instructions() {
        let add = GpuInstruction::elem_wise_buffers_add(9, 3, 6, 3);
        assert_eq!(add.opcode, opcodes::ELEM_WISE_BUFFERS_ADD);
        assert_eq!(add.output_ptr, 9);
        assert_eq!(add.data_size, 3);
        assert_eq!(add.param0, 6);
        assert_eq!(add.param1, 3);

        let mul = GpuInstruction::elem_wise_buffers_mul(9, 3, 6, 3);
        assert_eq!(mul.opcode, opcodes::ELEM_WISE_BUFFERS_MUL);
        assert_eq!(mul.param1, 3);
    }

    #[test]
    fn test_buffer_heads_instructions() {
        let mul = GpuInstruction::multiply_buffer_heads(0, 8, 6, 6, 2);
        assert_eq!(mul.opcode, opcodes::MULTIPLY_BUFFER_HEADS);
        assert_eq!(mul.input_ptr, 0);
        assert_eq!(mul.output_ptr, 8);
        assert_eq!(mul.data_size, 6);
        assert_eq!(mul.param0, 6);
        assert_eq!(mul.param1, 2);

        let add = GpuInstruction::add_buffer_heads(0, 8, 6, 6, 2);
        assert_eq!(add.opcode, opcodes::ADD_BUFFER_HEADS);
    }

    #[test]
    fn test_reduce_sum_instruction() {
        let inst = GpuInstruction::reduce_sum(2, 10, 5);
        assert_eq!(inst.opcode, opcodes::REDUCE_SUM);
        assert_eq!(inst.input_ptr, 2);
        assert_eq!(inst.output_ptr, 10);
        assert_eq!(inst.data_size, 5);
    }

    #[test]
    fn test_to_f32_array_roundtrip() {
        let inst = GpuInstruction::dot(42, 100, 16, 500, 32, Some(Activation::Sigmoid));
        let f32_array = inst.to_f32_array();

        assert_eq!(f32_array[0].to_bits(), opcodes::DOT);
        assert_eq!(f32_array[1].to_bits(), 42);
        assert_eq!(f32_array[2].to_bits(), 100);
        assert_eq!(f32_array[3].to_bits(), 16);
        assert_eq!(f32_array[4].to_bits(), 500);
        assert_eq!(f32_array[5].to_bits(), 32);
        assert_eq!(f32_array[6].to_bits(), activation_types::SIGMOID);
        assert_eq!(f32_array[7].to_bits(), 0);
    }
}
