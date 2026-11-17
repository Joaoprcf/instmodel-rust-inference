//! Graph → runnable model: θ scattering and [`InstructionModelInfo`] assembly.

use crate::errors::GraphError;
use crate::instruction_model_info::InstructionModelInfo;
use crate::params::ParamLayout;

use super::ModelGraph;
use super::plan::{Plan, plan};

/// The graph's [`ParamLayout`], derived from structure alone.
pub fn weights_map(graph: &ModelGraph) -> Result<ParamLayout, GraphError> {
    Ok(plan(graph)?.layout)
}

/// Compiles `graph` with `theta` (length = layout `total`) into a runnable [`InstructionModelInfo`].
pub fn compile(graph: &ModelGraph, theta: &[f32]) -> Result<InstructionModelInfo, GraphError> {
    assemble(plan(graph)?, theta)
}

/// [`compile`] with a zero θ, for workflows that overwrite θ in place afterwards.
pub fn compile_zeroed(graph: &ModelGraph) -> Result<InstructionModelInfo, GraphError> {
    let plan = plan(graph)?;
    let theta = vec![0.0f32; plan.layout.total];
    assemble(plan, &theta)
}

fn assemble(plan: Plan, theta: &[f32]) -> Result<InstructionModelInfo, GraphError> {
    if theta.len() != plan.layout.total {
        return Err(GraphError::ThetaLenMismatch {
            expected: plan.layout.total,
            got: theta.len(),
        });
    }
    let jagged = plan.layout.scatter_to_jagged(theta)?;
    Ok(InstructionModelInfo {
        features: plan.features,
        feature_size: Some(plan.feature_size),
        computation_buffer_sizes: plan.buffer_sizes,
        instructions: plan.instructions,
        weights: jagged.weights,
        bias: jagged.bias,
        parameters: if plan.parameters.is_empty() {
            None
        } else {
            Some(plan.parameters)
        },
        maps: if plan.maps.is_empty() {
            None
        } else {
            Some(plan.maps)
        },
        validation_data: None,
        format_version: None,
        sequence: None,
        validation_sequences: None,
    })
}
