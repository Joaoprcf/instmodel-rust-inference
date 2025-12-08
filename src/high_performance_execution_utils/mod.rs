//! Execution graph structures for parallel instruction execution.
//!
//! This module provides the dependency graph representation needed for
//! parallel execution of instructions. The graph tracks dependencies
//! between instructions based on buffer read/write patterns.

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
    /// Final state of buffer_last_nodes after processing all instructions.
    /// Each buffer index maps to a list of node indices that last wrote to it.
    pub buffer_last_nodes: Vec<Vec<usize>>,
}

/// Builder for constructing a ParallelExecutionGraph step by step.
/// Useful for testing intermediate states during graph construction.
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

    /// Processes the next instruction and returns Ok(true) if successful,
    /// Ok(false) if already complete, or an error if validation fails.
    pub fn step(&mut self) -> Result<bool, UnusedComputationError> {
        if self.is_complete() {
            return Ok(false);
        }

        let inst_index = self.current_index;
        let instruction_info = &self.instructions[inst_index];
        let input_indices = instruction_info.get_inputs();
        let output_index = instruction_info.output();

        // Collect unique parent node indices from all input buffers
        let mut parent_node_indices: HashSet<usize> = HashSet::new();
        for input_idx in input_indices {
            for &node_idx in &self.buffer_last_nodes[input_idx] {
                parent_node_indices.insert(node_idx);
            }
        }

        // Calculate dependency count
        let dependency_count = if parent_node_indices.is_empty() {
            1 // Root nodes have dependency_count=1 for execution simplicity
        } else {
            parent_node_indices.len()
        };

        let new_node_idx = self.nodes.len();
        self.nodes.push(ExecutionNode {
            instruction_index: inst_index,
            dependency_count,
            children: Vec::new(),
        });

        // If no parents, this is a root node
        if parent_node_indices.is_empty() {
            self.root_indices.push(new_node_idx);
        } else {
            // Add as child to all parent nodes
            for &parent_idx in &parent_node_indices {
                self.nodes[parent_idx].children.push(new_node_idx);
            }
        }

        // Update buffer_last_nodes for the output buffer
        if instruction_info.supports_partial_write() {
            // Partial write: append to the list (for concatenation patterns)
            self.buffer_last_nodes[output_index].push(new_node_idx);
        } else {
            // Full write: check for unused computation, then replace the list
            for &prev_node_idx in &self.buffer_last_nodes[output_index] {
                if self.nodes[prev_node_idx].children.is_empty() {
                    return Err(UnusedComputationError {
                        instruction_index: prev_node_idx,
                        buffer_index: output_index,
                        overwritten_by: inst_index,
                    });
                }
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
    /// Builds a parallel execution graph from instruction information.
    ///
    /// The graph is built by tracking which nodes last wrote to each buffer.
    /// Multiple nodes can be tracked per buffer to support partial writes (e.g., Copy
    /// with internal_index for concatenation).
    ///
    /// # Arguments
    /// * `instructions` - The instruction information from the model.
    /// * `num_buffers` - The number of computation buffers in the model.
    ///
    /// # Returns
    /// A `ParallelExecutionGraph` if successful, or an error if an instruction
    /// writes to a buffer that was previously written but never read (unused computation).
    pub fn build(
        instructions: &[InstructionInfo],
        num_buffers: usize,
    ) -> Result<Self, UnusedComputationError> {
        // Track multiple last nodes per buffer to support partial writes
        let mut buffer_last_nodes: Vec<Vec<usize>> = vec![Vec::new(); num_buffers];
        let mut nodes: Vec<ExecutionNode> = Vec::with_capacity(instructions.len());
        let mut root_indices: Vec<usize> = Vec::new();

        for (inst_index, instruction_info) in instructions.iter().enumerate() {
            let input_indices = instruction_info.get_inputs();
            let output_index = instruction_info.output();

            // Collect unique parent node indices from all input buffers
            let mut parent_node_indices: HashSet<usize> = HashSet::new();
            for input_idx in input_indices {
                for &node_idx in &buffer_last_nodes[input_idx] {
                    parent_node_indices.insert(node_idx);
                }
            }

            // Calculate dependency count
            let dependency_count = if parent_node_indices.is_empty() {
                1 // Root nodes have dependency_count=1 for execution simplicity
            } else {
                parent_node_indices.len()
            };

            let new_node_idx = nodes.len();
            nodes.push(ExecutionNode {
                instruction_index: inst_index,
                dependency_count,
                children: Vec::new(),
            });

            // If no parents, this is a root node
            if parent_node_indices.is_empty() {
                root_indices.push(new_node_idx);
            } else {
                // Add as child to all parent nodes
                for &parent_idx in &parent_node_indices {
                    nodes[parent_idx].children.push(new_node_idx);
                }
            }

            // Update buffer_last_nodes for the output buffer
            if instruction_info.supports_partial_write() {
                // Partial write: append to the list (for concatenation patterns)
                buffer_last_nodes[output_index].push(new_node_idx);
            } else {
                // Full write: check for unused computation, then replace the list
                for &prev_node_idx in &buffer_last_nodes[output_index] {
                    if nodes[prev_node_idx].children.is_empty() {
                        return Err(UnusedComputationError {
                            instruction_index: prev_node_idx,
                            buffer_index: output_index,
                            overwritten_by: inst_index,
                        });
                    }
                }
                buffer_last_nodes[output_index] = vec![new_node_idx];
            }
        }

        // Determine if parallelization is beneficial
        let is_parallelizable =
            root_indices.len() > 1 || nodes.iter().any(|n| n.children.len() > 1);

        Ok(ParallelExecutionGraph {
            nodes,
            root_indices,
            is_parallelizable,
            buffer_last_nodes,
        })
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

    /// Helper to create a simple DOT instruction info for testing.
    fn dot_inst(input: usize, output: usize) -> InstructionInfo {
        InstructionInfo::Dot(DotInstructionInfo {
            input,
            output,
            weights: 0,
            activation: None,
        })
    }

    /// Helper to create a COPY instruction info for testing.
    fn copy_inst(input: usize, output: usize) -> InstructionInfo {
        InstructionInfo::Copy(CopyInstructionInfo {
            input,
            output,
            internal_index: 0,
        })
    }

    /// Helper to create an ACTIVATION instruction info for testing (in-place).
    fn activation_inst(buffer: usize) -> InstructionInfo {
        InstructionInfo::Activation(ActivationInstructionInfo {
            input: buffer,
            activation: crate::activation::Activation::Relu,
        })
    }

    /// Helper to create a multi-input instruction for testing.
    fn buffers_add_inst(inputs: Vec<usize>, output: usize) -> InstructionInfo {
        InstructionInfo::ElemWiseBuffersAdd(ElemWiseBuffersAddInstructionInfo {
            input: inputs,
            output,
        })
    }

    #[test]
    fn test_model_1_sequential_chain() {
        // inst0: INPUT [0] → OUTPUT [1]
        // inst1: INPUT [1] → OUTPUT [2]
        let instructions = vec![dot_inst(0, 1), dot_inst(1, 2)];

        let graph = ParallelExecutionGraph::build(&instructions, 3).unwrap();

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.root_indices, vec![0]);
        assert!(!graph.is_parallelizable);

        // Node 0: root, one child (Node 1)
        assert_eq!(graph.nodes[0].instruction_index, 0);
        assert_eq!(graph.nodes[0].dependency_count, 1);
        assert_eq!(graph.nodes[0].children, vec![1]);

        // Node 1: depends on Node 0, no children
        assert_eq!(graph.nodes[1].instruction_index, 1);
        assert_eq!(graph.nodes[1].dependency_count, 1);
        assert!(graph.nodes[1].children.is_empty());
    }

    #[test]
    fn test_model_2_in_place_first() {
        // inst0: INPUT [0] → OUTPUT [0] (in-place!)
        // inst1: INPUT [0] → OUTPUT [1]
        let instructions = vec![activation_inst(0), dot_inst(0, 1)];

        let graph = ParallelExecutionGraph::build(&instructions, 2).unwrap();

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.root_indices, vec![0]);
        assert!(!graph.is_parallelizable);

        // Node 0: root, one child (Node 1)
        assert_eq!(graph.nodes[0].children, vec![1]);

        // Node 1: depends on Node 0
        assert_eq!(graph.nodes[1].dependency_count, 1);
    }

    #[test]
    fn test_model_3_fan_out() {
        // inst0: INPUT [0] → OUTPUT [1]
        // inst1: INPUT [1] → OUTPUT [0]
        // inst2: INPUT [1] → OUTPUT [2]
        let instructions = vec![dot_inst(0, 1), copy_inst(1, 0), copy_inst(1, 2)];

        let graph = ParallelExecutionGraph::build(&instructions, 3).unwrap();

        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.root_indices, vec![0]);
        assert!(graph.is_parallelizable); // Node 0 has 2 children

        // Node 0: root, two children (fan-out)
        assert_eq!(graph.nodes[0].children.len(), 2);
        assert!(graph.nodes[0].children.contains(&1));
        assert!(graph.nodes[0].children.contains(&2));

        // Node 1 and Node 2: both depend only on Node 0
        assert_eq!(graph.nodes[1].dependency_count, 1);
        assert_eq!(graph.nodes[2].dependency_count, 1);
    }

    #[test]
    fn test_model_5_two_roots() {
        // inst0: INPUT [0] → OUTPUT [1]
        // inst1: INPUT [0] → OUTPUT [2]
        // inst2: INPUT [1,2] → OUTPUT [3]
        let instructions = vec![
            dot_inst(0, 1),
            dot_inst(0, 2),
            buffers_add_inst(vec![1, 2], 3),
        ];

        let graph = ParallelExecutionGraph::build(&instructions, 4).unwrap();

        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.root_indices.len(), 2);
        assert!(graph.root_indices.contains(&0));
        assert!(graph.root_indices.contains(&1));
        assert!(graph.is_parallelizable); // Two roots

        // Node 0 and Node 1 are roots, each points to Node 2
        assert_eq!(graph.nodes[0].children, vec![2]);
        assert_eq!(graph.nodes[1].children, vec![2]);

        // Node 2: depends on both Node 0 and Node 1
        assert_eq!(graph.nodes[2].dependency_count, 2);
    }

    #[test]
    fn test_model_4_complex() {
        // inst0: INPUT [0] → OUTPUT [1]
        // inst1: INPUT [0] → OUTPUT [2]
        // inst2: INPUT [0,1,2] → OUTPUT [0]
        // inst3: INPUT [0] → OUTPUT [3]
        let instructions = vec![
            dot_inst(0, 1),
            dot_inst(0, 2),
            buffers_add_inst(vec![0, 1, 2], 0),
            dot_inst(0, 3),
        ];

        let graph = ParallelExecutionGraph::build(&instructions, 4).unwrap();

        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(graph.root_indices.len(), 2);
        assert!(graph.is_parallelizable); // Two roots

        // Node 0 and Node 1 are roots
        assert!(graph.root_indices.contains(&0));
        assert!(graph.root_indices.contains(&1));

        // Node 2 depends on Node 0 and Node 1 (buffer[0] is None, but [1] and [2] have nodes)
        assert_eq!(graph.nodes[2].dependency_count, 2);

        // Node 2's children should include Node 3
        assert!(graph.nodes[2].children.contains(&3));

        // Node 3 depends on Node 2 (via buffer[0])
        assert_eq!(graph.nodes[3].dependency_count, 1);
    }

    #[test]
    fn test_model_6_unused_computation_error() {
        // inst0: INPUT [0] → OUTPUT [1]
        // inst1: INPUT [0] → OUTPUT [1] ← Overwrites buffer[1] before anyone reads it!
        let instructions = vec![dot_inst(0, 1), dot_inst(0, 1)];

        let result = ParallelExecutionGraph::build(&instructions, 2);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.instruction_index, 0);
        assert_eq!(err.buffer_index, 1);
        assert_eq!(err.overwritten_by, 1);
    }

    #[test]
    fn test_empty_instructions() {
        let instructions: Vec<InstructionInfo> = vec![];
        let graph = ParallelExecutionGraph::build(&instructions, 1).unwrap();

        assert!(graph.is_empty());
        assert!(graph.root_indices.is_empty());
        assert!(!graph.is_parallelizable);
    }

    #[test]
    fn test_single_instruction() {
        let instructions = vec![dot_inst(0, 1)];
        let graph = ParallelExecutionGraph::build(&instructions, 2).unwrap();

        assert_eq!(graph.len(), 1);
        assert_eq!(graph.root_indices, vec![0]);
        assert!(!graph.is_parallelizable);
    }

    #[test]
    fn test_model_6_concatenation_pattern() {
        // inst0: INPUT [0]     → OUTPUT [1]
        // inst1: INPUT [0]     → OUTPUT [2]
        // inst2: INPUT [0,1,2] → OUTPUT [0]
        // inst3: INPUT [0]     → OUTPUT [3] (Copy for concatenation)
        // inst4: INPUT [1]     → OUTPUT [3] (Copy for concatenation)
        // inst5: INPUT [3]     → OUTPUT [4]
        let instructions = vec![
            dot_inst(0, 1),
            dot_inst(0, 2),
            buffers_add_inst(vec![0, 1, 2], 0),
            copy_inst(0, 3), // First part of concatenation
            copy_inst(1, 3), // Second part of concatenation
            dot_inst(3, 4),
        ];

        let graph = ParallelExecutionGraph::build(&instructions, 5).unwrap();

        assert_eq!(graph.nodes.len(), 6);
        assert_eq!(graph.root_indices.len(), 2);
        assert!(graph.is_parallelizable);

        // Node 0 and Node 1 are roots
        assert!(graph.root_indices.contains(&0));
        assert!(graph.root_indices.contains(&1));

        // Node 2 depends on Node 0 and Node 1
        assert_eq!(graph.nodes[2].dependency_count, 2);

        // Node 3 depends on Node 2 (via buffer[0])
        assert_eq!(graph.nodes[3].dependency_count, 1);

        // Node 4 depends on Node 0 (via buffer[1])
        assert_eq!(graph.nodes[4].dependency_count, 1);

        // Node 5 depends on BOTH Node 3 and Node 4 (via buffer[3] which has both)
        assert_eq!(graph.nodes[5].dependency_count, 2);

        // Both Node 3 and Node 4 should have Node 5 as child
        assert!(graph.nodes[3].children.contains(&5));
        assert!(graph.nodes[4].children.contains(&5));
    }

    #[test]
    fn test_partial_write_no_unused_error() {
        // Multiple Copy instructions to the same buffer should NOT trigger unused error
        // because Copy supports partial writes
        let instructions = vec![
            copy_inst(0, 1), // First write to buffer 1
            copy_inst(0, 1), // Second write to buffer 1 (appends, doesn't overwrite)
        ];

        // This should succeed - partial writes don't trigger unused computation error
        let graph = ParallelExecutionGraph::build(&instructions, 2).unwrap();
        assert_eq!(graph.nodes.len(), 2);
    }

    #[test]
    fn test_final_buffer_last_nodes() {
        // Test that buffer_last_nodes is correctly stored in the final graph
        let instructions = vec![
            dot_inst(0, 1),
            dot_inst(0, 2),
            buffers_add_inst(vec![1, 2], 3),
        ];

        let graph = ParallelExecutionGraph::build(&instructions, 4).unwrap();

        // Final state: buffer[0] = [], buffer[1] = [0], buffer[2] = [1], buffer[3] = [2]
        assert_eq!(graph.buffer_last_nodes[0], Vec::<usize>::new());
        assert_eq!(graph.buffer_last_nodes[1], vec![0]);
        assert_eq!(graph.buffer_last_nodes[2], vec![1]);
        assert_eq!(graph.buffer_last_nodes[3], vec![2]);
    }

    #[test]
    fn test_builder_step_by_step_model_4() {
        // Model 4 from the examples:
        // inst0: INPUT [0]     → OUTPUT [1]
        // inst1: INPUT [0]     → OUTPUT [2]
        // inst2: INPUT [0,1,2] → OUTPUT [0]
        // inst3: INPUT [0]     → OUTPUT [3]
        let instructions = vec![
            dot_inst(0, 1),
            dot_inst(0, 2),
            buffers_add_inst(vec![0, 1, 2], 0),
            dot_inst(0, 3),
        ];

        let mut builder = ParallelExecutionGraphBuilder::new(&instructions, 4);

        // Initial state
        let empty: Vec<usize> = vec![];
        assert_eq!(
            builder.buffer_last_nodes(),
            &[empty.clone(), empty.clone(), empty.clone(), empty.clone()]
        );
        assert!(builder.root_indices().is_empty());
        assert_eq!(builder.current_index(), 0);

        // Step 1: inst0 (INPUT [0] → OUTPUT [1])
        assert!(builder.step().unwrap());
        assert_eq!(
            builder.buffer_last_nodes(),
            &[vec![], vec![0], vec![], vec![]]
        );
        assert_eq!(builder.root_indices(), &[0]);

        // Step 2: inst1 (INPUT [0] → OUTPUT [2])
        assert!(builder.step().unwrap());
        assert_eq!(
            builder.buffer_last_nodes(),
            &[vec![], vec![0], vec![1], vec![]]
        );
        assert_eq!(builder.root_indices(), &[0, 1]);

        // Step 3: inst2 (INPUT [0,1,2] → OUTPUT [0])
        assert!(builder.step().unwrap());
        assert_eq!(
            builder.buffer_last_nodes(),
            &[vec![2], vec![0], vec![1], vec![]]
        );
        // Root indices unchanged (inst2 has parents)
        assert_eq!(builder.root_indices(), &[0, 1]);
        // Node 2 should have dependency_count = 2 (depends on Node 0 and Node 1)
        assert_eq!(builder.nodes()[2].dependency_count, 2);

        // Step 4: inst3 (INPUT [0] → OUTPUT [3])
        assert!(builder.step().unwrap());
        assert_eq!(
            builder.buffer_last_nodes(),
            &[vec![2], vec![0], vec![1], vec![3]]
        );
        // Node 3 depends on Node 2 (via buffer[0])
        assert_eq!(builder.nodes()[3].dependency_count, 1);
        // Node 2 should now have Node 3 as a child
        assert!(builder.nodes()[2].children.contains(&3));

        // No more steps
        assert!(!builder.step().unwrap());
        assert!(builder.is_complete());

        // Build final graph
        let graph = builder.build().unwrap();
        assert!(graph.is_parallelizable);
    }

    #[test]
    fn test_builder_step_by_step_concatenation() {
        // Model 6 with concatenation:
        // inst0: INPUT [0]     → OUTPUT [1]
        // inst1: INPUT [0]     → OUTPUT [2]
        // inst2: INPUT [0,1,2] → OUTPUT [0]
        // inst3: INPUT [0]     → OUTPUT [3] (Copy)
        // inst4: INPUT [1]     → OUTPUT [3] (Copy)
        // inst5: INPUT [3]     → OUTPUT [4]
        let instructions = vec![
            dot_inst(0, 1),
            dot_inst(0, 2),
            buffers_add_inst(vec![0, 1, 2], 0),
            copy_inst(0, 3),
            copy_inst(1, 3),
            dot_inst(3, 4),
        ];

        let mut builder = ParallelExecutionGraphBuilder::new(&instructions, 5);

        // Process first 4 instructions
        for _ in 0..4 {
            builder.step().unwrap();
        }

        // After inst3 (Copy to buffer[3])
        assert_eq!(
            builder.buffer_last_nodes(),
            &[vec![2], vec![0], vec![1], vec![3], vec![]]
        );

        // Step 5: inst4 (Copy from buffer[1] to buffer[3])
        builder.step().unwrap();
        // buffer[3] should now have BOTH Node 3 and Node 4 (partial writes append)
        assert_eq!(
            builder.buffer_last_nodes(),
            &[vec![2], vec![0], vec![1], vec![3, 4], vec![]]
        );

        // Step 6: inst5 (INPUT [3] → OUTPUT [4])
        builder.step().unwrap();
        // Node 5 should depend on both Node 3 and Node 4
        assert_eq!(builder.nodes()[5].dependency_count, 2);
        // Final buffer_last_nodes
        assert_eq!(
            builder.buffer_last_nodes(),
            &[vec![2], vec![0], vec![1], vec![3, 4], vec![5]]
        );

        let graph = builder.build().unwrap();
        assert_eq!(
            graph.buffer_last_nodes,
            vec![vec![2], vec![0], vec![1], vec![3, 4], vec![5]]
        );
    }
}
