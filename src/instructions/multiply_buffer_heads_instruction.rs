//! MULTIPLY_BUFFER_HEADS.

super::buffer_heads_instruction!(
    /// `out[h * head_dim + j] = data[h * head_dim + j] * heads[h]`.
    MultiplyBufferHeadsInstruction, *
);
