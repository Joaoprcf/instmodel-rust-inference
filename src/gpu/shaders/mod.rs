//! WGSL shader sources for GPU neural network inference.

/// Maximum compute buffer size (in f32s) - can be overridden by user
pub const DEFAULT_MAX_COMPUTE_BUFFER: u32 = 65536;

/// Get the composed WGSL shader source for instmodel inference.
///
/// This returns a complete shader that can be imported/included in other WGSL code.
/// The shader provides the `predict()` function and helper functions.
///
/// # Arguments
/// * `max_compute_buffer` - Maximum size of the compute buffer in f32s
pub fn get_instmodel_wgsl(max_compute_buffer: u32) -> String {
    format!(
        r#"// GPU Neural Network Inference Library
// Generated with max_compute_buffer = {max_compute_buffer}

const MAX_COMPUTE_BUFFER: u32 = {max_compute_buffer}u;

{activations}

{dot}

{activation_inst}

{elem_wise_add}

{elem_wise_mul}

{copy}

{instmodel}
"#,
        max_compute_buffer = max_compute_buffer,
        activations = include_str!("activations.wgsl"),
        dot = include_str!("instructions/dot.wgsl"),
        activation_inst = include_str!("instructions/activation.wgsl"),
        elem_wise_add = include_str!("instructions/elem_wise_add.wgsl"),
        elem_wise_mul = include_str!("instructions/elem_wise_mul.wgsl"),
        copy = include_str!("instructions/copy.wgsl"),
        instmodel = include_str!("instmodel.wgsl"),
    )
}

/// Get just the function definitions without any bindings.
/// This is useful when embedding the inference functions into an existing shader.
pub fn get_instmodel_functions_wgsl(max_compute_buffer: u32) -> String {
    get_instmodel_wgsl(max_compute_buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_instmodel_wgsl_compiles() {
        let wgsl = get_instmodel_wgsl(4096);
        assert!(wgsl.contains("fn predict("));
        assert!(wgsl.contains("fn get_feature_size("));
        assert!(wgsl.contains("fn get_output_start("));
        assert!(wgsl.contains("fn get_full_model_size("));
        assert!(wgsl.contains("MAX_COMPUTE_BUFFER"));
    }

    #[test]
    fn test_max_compute_buffer_substitution() {
        let wgsl_small = get_instmodel_wgsl(1024);
        let wgsl_large = get_instmodel_wgsl(65536);

        assert!(wgsl_small.contains("const MAX_COMPUTE_BUFFER: u32 = 1024u;"));
        assert!(wgsl_large.contains("const MAX_COMPUTE_BUFFER: u32 = 65536u;"));
    }
}
