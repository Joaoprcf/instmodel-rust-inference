//! [`SequenceModel`]: a validated format-2 step program.

use std::ops::Range;

use crate::errors::{InstructionModelError, InstructionModelResult};
use crate::instruction_model::build_instructions;
use crate::instruction_model_info::{InstructionModelInfo, KvSlotInfo, ValidationSequence};
use crate::instruction_model_validation as validation;
use crate::instructions::Instruction;
use crate::sequence::first_max;
use crate::sequence::state::{SequenceState, StepState};

/// Logits may differ from a model's `validation_sequences` by this much (absolute).
pub const SEQUENCE_VALIDATION_TOLERANCE: f32 = 1e-3;

/// A causal sequence model run one token at a time (JSON `format_version` 2).
///
/// The instruction list processes one token: buffer 0 receives the token id and the last
/// buffer holds the next-token logits. Everything that carries over between tokens
/// (attention key/value caches, feature scanners) lives in a caller-owned
/// [`SequenceState`], so one model serves any number of sequences, from any number of
/// threads. The model is immutable after construction.
///
/// ```
/// use instmodel_inference::{InstructionModelInfo, SequenceModel, argmax_allowed};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sequence/plain.json");
/// let info: InstructionModelInfo = serde_json::from_str(&std::fs::read_to_string(path)?)?;
/// let model = SequenceModel::new(info)?; // validates, replays validation_sequences
/// let mut state = model.new_state();
/// let mut token = 3;
/// for _ in 0..10 {
///     let logits = model.step(&mut state, token)?; // next-token logits
///     token = argmax_allowed(logits, &[0, 1, 2]).unwrap_or(0);
/// }
/// let saved = state.to_bytes();
/// let restored = model.state_from_bytes(&saved)?; // continues bit-identically
/// assert_eq!(restored.position(), 10);
/// # Ok(())
/// # }
/// ```
pub struct SequenceModel {
    instructions: Vec<Box<dyn Instruction>>,
    buffer_len: usize,
    logits: Range<usize>,
    vocabulary: Vec<String>,
    max_positions: usize,
    kv_slots: Vec<KvSlotInfo>,
    scanners: usize,
    config: serde_json::Value,
    metadata: serde_json::Value,
    fingerprint: u64,
}

impl std::fmt::Debug for SequenceModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequenceModel")
            .field("instructions", &self.instructions.len())
            .field("vocab_size", &self.vocabulary.len())
            .field("max_positions", &self.max_positions)
            .field("kv_slots", &self.kv_slots)
            .field("scanners", &self.scanners)
            .field("fingerprint", &format_args!("{:#018x}", self.fingerprint))
            .finish()
    }
}

impl SequenceModel {
    /// Validates a format-2 model and replays its `validation_sequences`: every logit must
    /// be within [`SEQUENCE_VALIDATION_TOLERANCE`] of the stored one and every argmax equal.
    pub fn new(info: InstructionModelInfo) -> InstructionModelResult<Self> {
        let not_sequence = |reason: &str| InstructionModelError::NotSequenceModel {
            reason: reason.to_string(),
        };
        if info.format_version != Some(2) {
            return Err(not_sequence("format_version must be 2"));
        }
        let sequence = info
            .sequence
            .as_ref()
            .ok_or_else(|| not_sequence("the sequence block is missing"))?;
        validation::validate_inputs(&info)?;
        let sizes = &info.computation_buffer_sizes;
        validation::validate_sequence_info(sequence, sizes)?;
        let mut indexes = Vec::new();
        let buffer_len = validation::calculate_computation_buffer_indexes(sizes, &mut indexes)?;
        validation::validate_required_memory(buffer_len)?;
        let (instructions, _) = build_instructions(&info, &indexes, sizes, Some(sequence))?;

        let model = Self {
            instructions,
            buffer_len,
            logits: buffer_len - sizes[sizes.len() - 1]..buffer_len,
            vocabulary: sequence.vocabulary.clone(),
            max_positions: sequence.max_positions,
            kv_slots: sequence.kv_slots.clone(),
            scanners: sequence.scanners,
            config: sequence.config.clone(),
            metadata: sequence.metadata.clone(),
            fingerprint: fingerprint(&info),
        };
        for (index, case) in info.validation_sequences.iter().flatten().enumerate() {
            model.replay(index, case)?;
        }
        Ok(model)
    }

    /// Token names, indexed by token id.
    pub fn vocabulary(&self) -> &[String] {
        &self.vocabulary
    }

    /// Number of tokens, which is also the number of logits.
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Tokens one [`SequenceState`] can hold.
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// The `sequence.config` JSON (the training configuration), `null` when absent.
    pub fn config(&self) -> &serde_json::Value {
        &self.config
    }

    /// The `sequence.metadata` JSON, `null` when absent.
    pub fn metadata(&self) -> &serde_json::Value {
        &self.metadata
    }

    /// An empty state with every cache allocated for [`Self::max_positions`] tokens.
    pub fn new_state(&self) -> SequenceState {
        SequenceState {
            fingerprint: self.fingerprint,
            buffer: vec![0.0; self.buffer_len],
            step: StepState::new(self.max_positions, &self.kv_slots, self.scanners),
        }
    }

    /// Feeds `token` and returns the next-token logits, which live in `state`.
    ///
    /// Does not allocate. On error the position and caches are unchanged, so the state
    /// continues as before; the logits of the previous step may be overwritten.
    pub fn step<'s>(
        &self,
        state: &'s mut SequenceState,
        token: u32,
    ) -> InstructionModelResult<&'s [f32]> {
        if state.fingerprint != self.fingerprint {
            return Err(InstructionModelError::SequenceStateModelMismatch {
                expected: self.fingerprint,
                found: state.fingerprint,
            });
        }
        if token as usize >= self.vocabulary.len() {
            return Err(InstructionModelError::TokenOutOfVocabulary {
                token,
                vocab_size: self.vocabulary.len(),
            });
        }
        if state.step.position >= self.max_positions {
            return Err(InstructionModelError::SequenceFull {
                max_positions: self.max_positions,
            });
        }
        let buffer = state.buffer.as_mut_slice();
        // Instructions that write part of a buffer rely on the rest being zero.
        buffer.fill(0.0);
        // Exact: the vocabulary is smaller than the buffer limit, far below 2^24.
        buffer[0] = token as f32;
        for instruction in &self.instructions {
            instruction.apply_with_state(buffer, &mut state.step)?;
        }
        state.step.position += 1;
        Ok(&state.buffer[self.logits.clone()])
    }

    /// Restores a state saved by [`SequenceState::to_bytes`] from this model; bytes from
    /// another model fail with [`InstructionModelError::SequenceStateModelMismatch`].
    pub fn state_from_bytes(&self, bytes: &[u8]) -> InstructionModelResult<SequenceState> {
        let mut state = self.new_state();
        state.restore(bytes)?;
        Ok(state)
    }

    fn replay(&self, index: usize, case: &ValidationSequence) -> InstructionModelResult<()> {
        let malformed = |reason: String| InstructionModelError::ValidationSequenceShape {
            sequence: index,
            reason,
        };
        if case.tokens.len() != case.logits.len() {
            return Err(malformed(format!(
                "{} tokens but {} logit rows",
                case.tokens.len(),
                case.logits.len()
            )));
        }
        if case.tokens.len() > self.max_positions {
            return Err(malformed(format!(
                "{} tokens exceed max_positions {}",
                case.tokens.len(),
                self.max_positions
            )));
        }
        let mut state = self.new_state();
        for (position, (&token, expected)) in case.tokens.iter().zip(&case.logits).enumerate() {
            if expected.len() != self.vocab_size() {
                return Err(malformed(format!(
                    "logit row {position} has {} values, not {}",
                    expected.len(),
                    self.vocab_size()
                )));
            }
            let logits = self.step(&mut state, token)?;
            let max_error = logits
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, |worst, error| {
                    if error > worst || error.is_nan() {
                        error
                    } else {
                        worst
                    }
                });
            let (computed_argmax, expected_argmax) = (argmax(logits), argmax(expected));
            if max_error.is_nan()
                || max_error > SEQUENCE_VALIDATION_TOLERANCE
                || computed_argmax != expected_argmax
            {
                return Err(InstructionModelError::SequenceValidationMismatch {
                    sequence: index,
                    position,
                    max_error,
                    expected_argmax,
                    computed_argmax,
                });
            }
        }
        Ok(())
    }
}

/// Index of the first maximum (NumPy's `argmax`, including its preference for NaN).
fn argmax(values: &[f32]) -> usize {
    first_max(values.iter().copied().enumerate()).unwrap_or(0)
}

/// FNV-1a 64 over everything that affects a step: buffer layout, instructions, weights,
/// bias, parameters, maps and the sequence block's shapes and vocabulary.
fn fingerprint(info: &InstructionModelInfo) -> u64 {
    let mut hash = Fnv::default();
    hash.bytes(b"instmodel-sequence-v1");
    hash.usize(info.computation_buffer_sizes.len());
    for &size in &info.computation_buffer_sizes {
        hash.usize(size);
    }
    // Instruction structs serialize deterministically (fixed field order, no maps).
    match serde_json::to_vec(&info.instructions) {
        Ok(json) => hash.bytes(&json),
        Err(_) => hash.bytes(b"?"),
    }
    let parameters = info.parameters.as_deref().unwrap_or(&[]);
    for group in [info.bias.as_slice(), parameters] {
        hash.usize(group.len());
        for vector in group {
            hash.floats(vector);
        }
    }
    hash.usize(info.weights.len());
    for matrix in &info.weights {
        hash.usize(matrix.len());
        for row in matrix {
            hash.floats(row);
        }
    }
    let maps = info.maps.as_deref().unwrap_or(&[]);
    hash.usize(maps.len());
    for map in maps {
        let mut entries: Vec<_> = map.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        hash.usize(entries.len());
        for (key, values) in entries {
            hash.str(key);
            hash.floats(values);
        }
    }
    if let Some(sequence) = &info.sequence {
        hash.usize(sequence.max_positions);
        hash.usize(sequence.scanners);
        hash.usize(sequence.kv_slots.len());
        for slot in &sequence.kv_slots {
            hash.usize(slot.heads);
            hash.usize(slot.kv_heads);
            hash.usize(slot.head_dim);
        }
        hash.usize(sequence.vocabulary.len());
        for token in &sequence.vocabulary {
            hash.str(token);
        }
    }
    hash.0
}

struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv {
    fn bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn usize(&mut self, value: usize) {
        self.bytes(&(value as u64).to_le_bytes());
    }

    fn str(&mut self, value: &str) {
        self.usize(value.len());
        self.bytes(value.as_bytes());
    }

    fn floats(&mut self, values: &[f32]) {
        self.usize(values.len());
        for value in values {
            self.bytes(&value.to_le_bytes());
        }
    }
}
