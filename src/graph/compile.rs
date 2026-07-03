//! Graph → runnable model: θ scattering and [`InstructionModelInfo`] assembly.

use crate::errors::GraphError;
use crate::instruction_model_info::InstructionModelInfo;
use crate::params::ParamLayout;

use super::ModelGraph;
use super::plan::{Plan, plan};

/// Returns the canonical flat-θ ↔ model offset table for `graph` without
/// needing any weight data.
pub fn weights_map(graph: &ModelGraph) -> Result<ParamLayout, GraphError> {
    Ok(plan(graph)?.layout)
}

/// Compiles `graph` with a flat parameter vector into a runnable
/// [`InstructionModelInfo`].
///
/// `theta` is scattered through the graph's [`ParamLayout`] into the served
/// jagged `weights`/`bias`; `theta.len()` must equal the layout's `total`.
pub fn compile(graph: &ModelGraph, theta: &[f32]) -> Result<InstructionModelInfo, GraphError> {
    assemble(plan(graph)?, theta)
}

/// Compiles `graph` with all parameters zeroed — a template for workflows that
/// immediately overwrite θ in place
/// ([`InstructionModel::apply_theta`](crate::InstructionModel::apply_theta)).
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
    })
}
