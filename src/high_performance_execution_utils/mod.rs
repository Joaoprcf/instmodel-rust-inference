//! Instruction dependency graph derived from buffer read/write patterns.

use std::collections::HashSet;

use crate::errors::UnusedComputationError;
use crate::instruction_model_info::InstructionInfo;

/// A node in the execution graph representing a single instruction.
#[derive(Debug, Clone)]
pub struct ExecutionNode {
    /// Index of this instruction in the instructions vector.
    pub instruction_index: usize,
    /// Number of parent nodes that must complete before this node can execute.
    pub dependency_count: usize,
    /// Indices of child nodes that depend on this node's output.
    pub children: Vec<usize>,
}

/// The parallel execution graph built from instruction dependencies.
#[derive(Debug, Clone)]
pub struct ParallelExecutionGraph {
    /// All nodes in the graph, one per instruction.
    pub nodes: Vec<ExecutionNode>,
    /// Indices of root nodes (nodes with no dependencies on other instructions).
    pub root_indices: Vec<usize>,
    /// Whether the graph has parallelization opportunities.
    pub is_parallelizable: bool,
    /// Per buffer, the nodes that last wrote to it.
    pub buffer_last_nodes: Vec<Vec<usize>>,
}

/// Step-by-step [`ParallelExecutionGraph`] construction (exposes intermediate state).
#[derive(Debug, Clone)]
pub struct ParallelExecutionGraphBuilder<'a> {
    instructions: &'a [InstructionInfo],
    buffer_last_nodes: Vec<Vec<usize>>,
    nodes: Vec<ExecutionNode>,
    root_indices: Vec<usize>,
    current_index: usize,
}

impl<'a> ParallelExecutionGraphBuilder<'a> {
    /// Creates a new builder for the given instructions and buffer count.
    pub fn new(instructions: &'a [InstructionInfo], num_buffers: usize) -> Self {
        Self {
            instructions,
            buffer_last_nodes: vec![Vec::new(); num_buffers],
            nodes: Vec::with_capacity(instructions.len()),
            root_indices: Vec::new(),
            current_index: 0,
        }
    }

    /// Returns the current buffer_last_nodes state.
    pub fn buffer_last_nodes(&self) -> &[Vec<usize>] {
        &self.buffer_last_nodes
    }

    /// Returns the current nodes.
    pub fn nodes(&self) -> &[ExecutionNode] {
        &self.nodes
    }

    /// Returns the current root indices.
    pub fn root_indices(&self) -> &[usize] {
        &self.root_indices
    }

    /// Returns the current instruction index (how many have been processed).
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// Returns true if all instructions have been processed.
    pub fn is_complete(&self) -> bool {
        self.current_index >= self.instructions.len()
    }

    /// Processes the next instruction; `Ok(false)` once all are processed.
    pub fn step(&mut self) -> Result<bool, UnusedComputationError> {
        if self.is_complete() {
            return Ok(false);
        }

        let inst_index = self.current_index;
        let instruction_info = &self.instructions[inst_index];
        let output_index = instruction_info.output();
        let parents: HashSet<usize> = instruction_info
            .get_inputs()
            .into_iter()
            .flat_map(|input| self.buffer_last_nodes[input].iter().copied())
            .collect();

        let new_node_idx = self.nodes.len();
        self.nodes.push(ExecutionNode {
            instruction_index: inst_index,
            // Roots count as having one dependency so execution can treat all nodes alike.
            dependency_count: parents.len().max(1),
            children: Vec::new(),
        });
        if parents.is_empty() {
            self.root_indices.push(new_node_idx);
        }
        for &parent_idx in &parents {
            self.nodes[parent_idx].children.push(new_node_idx);
        }

        if instruction_info.supports_partial_write() {
            self.buffer_last_nodes[output_index].push(new_node_idx);
        } else {
            if let Some(&prev) = self.buffer_last_nodes[output_index]
                .iter()
                .find(|&&prev| self.nodes[prev].children.is_empty())
            {
                return Err(UnusedComputationError {
                    instruction_index: prev,
                    buffer_index: output_index,
                    overwritten_by: inst_index,
                });
            }
            self.buffer_last_nodes[output_index] = vec![new_node_idx];
        }

        self.current_index += 1;
        Ok(true)
    }

    /// Processes all remaining instructions and builds the final graph.
    pub fn build(mut self) -> Result<ParallelExecutionGraph, UnusedComputationError> {
        while self.step()? {}

        let is_parallelizable =
            self.root_indices.len() > 1 || self.nodes.iter().any(|n| n.children.len() > 1);

        Ok(ParallelExecutionGraph {
            nodes: self.nodes,
            root_indices: self.root_indices,
            is_parallelizable,
            buffer_last_nodes: self.buffer_last_nodes,
        })
    }
}

impl ParallelExecutionGraph {
    /// Builds the dependency graph; partial writers (COPY, COPY_MASKED, MAP_TRANSFORM) accumulate
    /// as co-owners of their output buffer. Errors when a full write overwrites an unread result.
    pub fn build(
        instructions: &[InstructionInfo],
        num_buffers: usize,
    ) -> Result<Self, UnusedComputationError> {
        ParallelExecutionGraphBuilder::new(instructions, num_buffers).build()
    }

    /// Returns true if the graph can benefit from parallel execution.
    pub fn is_parallelizable(&self) -> bool {
        self.is_parallelizable
    }

    /// Returns the number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction_model_info::{
        ActivationInstructionInfo, CopyInstructionInfo, DotInstructionInfo,
        ElemWiseBuffersAddInstructionInfo,
    };

    fn dot(input: usize, output: usize) -> InstructionInfo {
        InstructionInfo::Dot(DotInstructionInfo {
            input,
            output,
            weights: 0,
            activation: None,
        })
    }

    fn copy(input: usize, output: usize) -> InstructionInfo {
        InstructionInfo::Copy(CopyInstructionInfo {
            input,
            output,
            internal_index: 0,
        })
    }

    fn relu(buffer: usize) -> InstructionInfo {
        InstructionInfo::Activation(ActivationInstructionInfo {
            input: buffer,
            activation: crate::activation::Activation::Relu,
        })
    }

    fn add(input: Vec<usize>, output: usize) -> InstructionInfo {
        InstructionInfo::ElemWiseBuffersAdd(ElemWiseBuffersAddInstructionInfo { input, output })
    }

    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort();
        v
    }

    struct Case {
        instructions: Vec<InstructionInfo>,
        buffers: usize,
        roots: Vec<usize>,
        dependency_counts: Vec<usize>,
        children: Vec<Vec<usize>>,
        last_writers: Vec<Vec<usize>>,
        parallel: bool,
    }

    #[test]
    fn builds_dependency_graphs() {
        let cases = [
            Case {
                instructions: vec![],
                buffers: 1,
                roots: vec![],
                dependency_counts: vec![],
                children: vec![],
                last_writers: vec![vec![]],
                parallel: false,
            },
            Case {
                instructions: vec![dot(0, 1), dot(1, 2)],
                buffers: 3,
                roots: vec![0],
                dependency_counts: vec![1, 1],
                children: vec![vec![1], vec![]],
                last_writers: vec![vec![], vec![0], vec![1]],
                parallel: false,
            },
            Case {
                instructions: vec![relu(0), dot(0, 1)],
                buffers: 2,
                roots: vec![0],
                dependency_counts: vec![1, 1],
                children: vec![vec![1], vec![]],
                last_writers: vec![vec![0], vec![1]],
                parallel: false,
            },
            Case {
                instructions: vec![dot(0, 1), copy(1, 0), copy(1, 2)],
                buffers: 3,
                roots: vec![0],
                dependency_counts: vec![1, 1, 1],
                children: vec![vec![1, 2], vec![], vec![]],
                last_writers: vec![vec![1], vec![0], vec![2]],
                parallel: true,
            },
            Case {
                instructions: vec![dot(0, 1), dot(0, 2), add(vec![1, 2], 3)],
                buffers: 4,
                roots: vec![0, 1],
                dependency_counts: vec![1, 1, 2],
                children: vec![vec![2], vec![2], vec![]],
                last_writers: vec![vec![], vec![0], vec![1], vec![2]],
                parallel: true,
            },
            // Partial writers (COPY) into buffer 3 both feed the final DOT.
            Case {
                instructions: vec![
                    dot(0, 1),
                    dot(0, 2),
                    add(vec![0, 1, 2], 0),
                    copy(0, 3),
                    copy(1, 3),
                    dot(3, 4),
                ],
                buffers: 5,
                roots: vec![0, 1],
                dependency_counts: vec![1, 1, 2, 1, 1, 2],
                children: vec![vec![2, 4], vec![2], vec![3], vec![5], vec![5], vec![]],
                last_writers: vec![vec![2], vec![0], vec![1], vec![3, 4], vec![5]],
                parallel: true,
            },
            Case {
                instructions: vec![copy(0, 1), copy(0, 1)],
                buffers: 2,
                roots: vec![0, 1],
                dependency_counts: vec![1, 1],
                children: vec![vec![], vec![]],
                last_writers: vec![vec![], vec![0, 1]],
                parallel: true,
            },
        ];
        for (i, case) in cases.into_iter().enumerate() {
            let graph = ParallelExecutionGraph::build(&case.instructions, case.buffers).unwrap();
            assert_eq!(graph.len(), case.instructions.len(), "case {i}");
            assert_eq!(sorted(graph.root_indices.clone()), case.roots, "case {i}");
            let deps: Vec<usize> = graph.nodes.iter().map(|n| n.dependency_count).collect();
            assert_eq!(deps, case.dependency_counts, "case {i}");
            let children: Vec<Vec<usize>> = graph
                .nodes
                .iter()
                .map(|n| sorted(n.children.clone()))
                .collect();
            assert_eq!(children, case.children, "case {i}");
            assert_eq!(graph.buffer_last_nodes, case.last_writers, "case {i}");
            assert_eq!(graph.is_parallelizable(), case.parallel, "case {i}");
        }
    }

    #[test]
    fn overwrite_before_read_is_unused_computation() {
        let err = ParallelExecutionGraph::build(&[dot(0, 1), dot(0, 1)], 2).unwrap_err();
        assert_eq!(
            (err.instruction_index, err.buffer_index, err.overwritten_by),
            (0, 1, 1)
        );
    }

    #[test]
    fn builder_exposes_intermediate_state() {
        let instructions = [dot(0, 1), dot(0, 2), add(vec![0, 1, 2], 0), copy(0, 3)];
        let mut builder = ParallelExecutionGraphBuilder::new(&instructions, 4);
        assert_eq!(builder.current_index(), 0);
        let expected: [&[Vec<usize>]; 4] = [
            &[vec![], vec![0], vec![], vec![]],
            &[vec![], vec![0], vec![1], vec![]],
            &[vec![2], vec![0], vec![1], vec![]],
            &[vec![2], vec![0], vec![1], vec![3]],
        ];
        for (step, writers) in expected.iter().enumerate() {
            assert!(builder.step().unwrap());
            assert_eq!(builder.buffer_last_nodes(), *writers, "step {step}");
            assert_eq!(builder.nodes().len(), step + 1);
        }
        assert_eq!(builder.root_indices(), &[0, 1]);
        assert!(builder.is_complete());
        assert!(!builder.step().unwrap());
        assert!(builder.build().unwrap().is_parallelizable);
    }
}
