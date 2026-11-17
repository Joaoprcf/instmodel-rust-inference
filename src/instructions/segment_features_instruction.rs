//! SEGMENT_FEATURES.

use crate::errors::InstructionModelError;
use crate::instructions::Instruction;
use crate::sequence::{ScannerState, StepState};

/// Writes `[ordinal, index]` of the current token's run of member tokens, advancing one
/// feature scanner of the sequence state.
///
/// A token is in a run when it is a member, or an opener while the previous token is not a
/// member (a sign before a digit). `ordinal` counts run starts so far and `index` is the
/// one-based place in the current run (0 outside a run); both outputs clamp to `rows - 1`,
/// while the scanner keeps the unclamped counts.
#[derive(Clone)]
pub struct SegmentFeaturesInstruction {
    input_ptr: usize,
    output_ptr: usize,
    data_size: usize,
    members: Vec<u32>,
    openers: Vec<u32>,
    limit: u32,
    scanner: usize,
}

impl SegmentFeaturesInstruction {
    /// `rows` must be at least 1.
    pub fn new(
        input_ptr: usize,
        output_ptr: usize,
        members: &[u32],
        openers: &[u32],
        rows: usize,
        scanner: usize,
    ) -> Self {
        let sorted = |ids: &[u32]| {
            let mut ids = ids.to_vec();
            ids.sort_unstable();
            ids.dedup();
            ids
        };
        Self {
            input_ptr,
            output_ptr,
            data_size: 2,
            members: sorted(members),
            openers: sorted(openers),
            limit: u32::try_from(rows.saturating_sub(1)).unwrap_or(u32::MAX),
            scanner,
        }
    }

    /// The scanner state after `token`, given the state after the previous token.
    fn advance(&self, previous: ScannerState, token: f32) -> ScannerState {
        let token = token.round_ties_even();
        // NaN and negative values are not token ids, so they belong to no set.
        let id = (token >= 0.0).then_some(token as u32);
        let contains = |ids: &[u32]| id.is_some_and(|id| ids.binary_search(&id).is_ok());
        let member = contains(&self.members);
        let run = member || (contains(&self.openers) && !previous.prev_member);
        let start = run && !previous.prev_run;
        ScannerState {
            prev_member: member,
            prev_run: run,
            ordinal: previous.ordinal.saturating_add(u32::from(start)),
            index: if start {
                1
            } else if run {
                previous.index.saturating_add(1)
            } else {
                0
            },
        }
    }
}

impl Instruction for SegmentFeaturesInstruction {
    super::instruction_common!();

    fn apply(&self, _buffer: &mut [f32]) -> Result<(), InstructionModelError> {
        Err(InstructionModelError::MissingSequenceState {
            instruction_type: "SEGMENT_FEATURES".to_string(),
        })
    }

    fn apply_with_state(
        &self,
        buffer: &mut [f32],
        state: &mut StepState,
    ) -> Result<(), InstructionModelError> {
        let position = state.position;
        let history = state.scanners.get_mut(self.scanner).ok_or(
            InstructionModelError::ScannerIndexOutOfBounds {
                index: self.scanner,
            },
        )?;
        // history[t] is the scanner after t tokens; this token writes history[position + 1].
        if position + 1 >= history.len() {
            return Err(InstructionModelError::SequenceFull {
                max_positions: history.len().saturating_sub(1),
            });
        }
        let next = self.advance(history[position], buffer[self.input_ptr]);
        history[position + 1] = next;
        buffer[self.output_ptr] = next.ordinal.min(self.limit) as f32;
        buffer[self.output_ptr + 1] = next.index.min(self.limit) as f32;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeds `tokens` and returns the `(ordinal, index)` written after each.
    fn scan(instruction: &SegmentFeaturesInstruction, tokens: &[f32]) -> Vec<(f32, f32)> {
        let mut state = StepState::for_test(tokens.len(), 1);
        let mut out = Vec::new();
        for &token in tokens {
            let mut buffer = [token, 0.0, 0.0];
            instruction
                .apply_with_state(&mut buffer, &mut state)
                .unwrap();
            state.position += 1;
            out.push((buffer[1], buffer[2]));
        }
        out
    }

    #[test]
    fn follows_signed_runs() {
        // instmodel tests/test_sequence.py::test_segment_features_follow_signed_runs:
        // "-12--3=45678-=9" with digits 0..9 as members, '-' (11) as opener, 4 rows.
        let digits: Vec<u32> = (0..10).collect();
        let instruction = SegmentFeaturesInstruction::new(0, 1, &digits, &[11], 4, 0);
        let text = [11, 1, 2, 11, 11, 3, 12, 4, 5, 6, 7, 8, 11, 12, 9].map(|t| t as f32);
        let (ordinal, index): (Vec<f32>, Vec<f32>) = scan(&instruction, &text).into_iter().unzip();
        assert_eq!(
            index,
            [1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 3, 3, 0, 0, 1].map(|v| v as f32)
        );
        assert_eq!(
            ordinal,
            [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3].map(|v| v as f32)
        );
    }

    #[test]
    fn clamps_outputs_but_not_counts() {
        // Runs "11", "1111", "1": index clamps at 1 while counting on; ordinal passes the limit.
        let instruction = SegmentFeaturesInstruction::new(0, 1, &[1], &[], 2, 0);
        let tokens = [1, 1, 0, 1, 1, 1, 1, 0, 1].map(|t| t as f32);
        let out = scan(&instruction, &tokens);
        let expected = [
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        ];
        assert_eq!(out, expected);
        let mut state = StepState::for_test(tokens.len(), 1);
        for (position, &token) in tokens.iter().enumerate() {
            state.position = position;
            instruction
                .apply_with_state(&mut [token, 0.0, 0.0], &mut state)
                .unwrap();
        }
        let last = state.scanners[0][tokens.len()];
        assert_eq!((last.ordinal, last.index), (3, 1));
        assert_eq!(state.scanners[0][7].index, 4);
        assert_eq!(state.scanners[0][8].index, 0);
    }

    #[test]
    fn non_ids_are_outside_every_run() {
        let instruction = SegmentFeaturesInstruction::new(0, 1, &[0], &[1], 8, 0);
        let out = scan(&instruction, &[0.0, f32::NAN, -0.2, -3.0, 1.0, 0.0]);
        let expected = [
            (1.0, 1.0),
            (1.0, 0.0),
            (2.0, 1.0),
            (2.0, 0.0),
            (3.0, 1.0),
            (3.0, 2.0),
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn needs_state_and_room() {
        let instruction = SegmentFeaturesInstruction::new(0, 1, &[0], &[], 4, 0);
        assert!(matches!(
            instruction.apply(&mut [0.0; 3]),
            Err(InstructionModelError::MissingSequenceState { .. })
        ));
        let mut state = StepState::for_test(1, 1);
        state.position = 1;
        assert!(matches!(
            instruction.apply_with_state(&mut [0.0; 3], &mut state),
            Err(InstructionModelError::SequenceFull { max_positions: 1 })
        ));
        let other = SegmentFeaturesInstruction::new(0, 1, &[0], &[], 4, 1);
        assert!(matches!(
            other.apply_with_state(&mut [0.0; 3], &mut StepState::for_test(1, 1)),
            Err(InstructionModelError::ScannerIndexOutOfBounds { index: 1 })
        ));
    }
}
