//! MUL_ELEMENTWISE.

super::param_elementwise_instruction!(
    /// In-place `x[i] *= parameters[i]`.
    ElemWiseMulInstruction, *=
);
