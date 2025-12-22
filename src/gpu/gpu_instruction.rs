//! GPU instruction encoding for neural network inference.

use crate::activation::Activation;

/// Opcode constants matching WGSL definitions.
pub mod opcodes {
    pub const DOT: u32 = 0x01;
    pub const ACTIVATION: u32 = 0x02;
    pub const ELEM_WISE_ADD: u32 = 0x03;
    pub const ELEM_WISE_MUL: u32 = 0x04;
    pub const COPY: u32 = 0x05;
}

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
