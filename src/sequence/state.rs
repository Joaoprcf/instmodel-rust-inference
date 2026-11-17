//! Per-sequence state: key/value caches, scanner histories and the step buffer.

use crate::errors::InstructionModelError;
use crate::instruction_model_info::KvSlotInfo;

/// One feature scanner after some number of tokens.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ScannerState {
    pub(crate) prev_member: bool,
    pub(crate) prev_run: bool,
    pub(crate) ordinal: u32,
    pub(crate) index: u32,
}

/// Keys and values of one attention instruction, `[position][kv_heads][head_dim]`, with
/// room for `max_positions` tokens.
#[derive(Debug)]
pub(crate) struct KvCache {
    /// `kv_heads * head_dim`: values per position.
    pub(crate) row: usize,
    pub(crate) keys: Vec<f32>,
    pub(crate) values: Vec<f32>,
}

/// The caches stateful instructions read and update during
/// [`crate::SequenceModel::step`]: one key/value slot per CAUSAL_ATTENTION, one scanner
/// history per SEGMENT_FEATURES, the number of tokens consumed and score scratch space.
///
/// Every entry is indexed by position, so dropping the tail (truncation, or a failed step
/// that wrote past the current position) needs no undo.
#[derive(Debug)]
pub struct StepState {
    pub(crate) position: usize,
    pub(crate) slots: Vec<KvCache>,
    /// `scanners[s][t]` is scanner `s` after `t` tokens; `[s][0]` is the empty scanner.
    pub(crate) scanners: Vec<Vec<ScannerState>>,
    /// One attention score per cached position.
    pub(crate) scores: Vec<f32>,
}

impl StepState {
    /// Tokens consumed before the current step.
    pub fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn new(max_positions: usize, kv_slots: &[KvSlotInfo], scanners: usize) -> Self {
        Self {
            position: 0,
            slots: kv_slots
                .iter()
                .map(|slot| {
                    let row = slot.kv_heads * slot.head_dim;
                    KvCache {
                        row,
                        keys: vec![0.0; row * max_positions],
                        values: vec![0.0; row * max_positions],
                    }
                })
                .collect(),
            scanners: vec![vec![ScannerState::default(); max_positions + 1]; scanners],
            scores: vec![0.0; max_positions],
        }
    }

    /// Tokens the state can hold.
    pub(crate) fn max_positions(&self) -> usize {
        self.scores.len()
    }

    #[cfg(test)]
    pub(crate) fn for_test(max_positions: usize, scanners: usize) -> Self {
        Self::new(max_positions, &[], scanners)
    }

    /// Slots given as `(kv_heads, head_dim)`.
    #[cfg(test)]
    pub(crate) fn for_test_slots(max_positions: usize, slots: &[(usize, usize)]) -> Self {
        let slots: Vec<KvSlotInfo> = slots
            .iter()
            .map(|&(kv_heads, head_dim)| KvSlotInfo {
                heads: kv_heads,
                kv_heads,
                head_dim,
            })
            .collect();
        Self::new(max_positions, &slots, 0)
    }
}

impl Clone for StepState {
    /// Copies only the used prefix of each cache; the rest stays zeroed capacity.
    fn clone(&self) -> Self {
        let max_positions = self.max_positions();
        let used = |values: &[f32], row: usize| {
            let mut copy = vec![0.0; row * max_positions];
            let end = self.position * row;
            copy[..end].copy_from_slice(&values[..end]);
            copy
        };
        Self {
            position: self.position,
            slots: self
                .slots
                .iter()
                .map(|cache| KvCache {
                    row: cache.row,
                    keys: used(&cache.keys, cache.row),
                    values: used(&cache.values, cache.row),
                })
                .collect(),
            scanners: self
                .scanners
                .iter()
                .map(|history| {
                    let mut copy = vec![ScannerState::default(); history.len()];
                    copy[..=self.position].copy_from_slice(&history[..=self.position]);
                    copy
                })
                .collect(),
            scores: vec![0.0; max_positions],
        }
    }
}

/// The caller-owned state of one sequence for a [`crate::SequenceModel`].
///
/// It holds the key/value caches, the feature scanner history and the step buffer
/// whose last part holds the logits returned by [`crate::SequenceModel::step`]. All
/// storage is sized for `max_positions` when the state is created, so stepping does
/// not allocate. Clones are independent and copy only the used prefix of the caches.
#[derive(Debug, Clone)]
pub struct SequenceState {
    pub(crate) fingerprint: u64,
    pub(crate) buffer: Vec<f32>,
    pub(crate) step: StepState,
}

const MAGIC: &[u8; 4] = b"IMSQ";
const LAYOUT_VERSION: u32 = 1;
const HEADER_LEN: usize = 4 + 4 + 8 + 8 + 4 + 4;
const SCANNER_ENTRY_LEN: usize = 1 + 4 + 4;

impl SequenceState {
    /// Tokens consumed so far.
    pub fn position(&self) -> usize {
        self.step.position
    }

    /// Tokens the state can hold.
    pub fn max_positions(&self) -> usize {
        self.step.max_positions()
    }

    /// Drops every token after the first `position`, so the state continues exactly as
    /// if only those tokens had been fed.
    pub fn truncate(&mut self, position: usize) -> Result<(), InstructionModelError> {
        if position > self.step.position {
            return Err(InstructionModelError::TruncatePastPosition {
                requested: position,
                position: self.step.position,
            });
        }
        self.step.position = position;
        Ok(())
    }

    /// Forgets every token, as a new state would.
    pub fn reset(&mut self) {
        self.step.position = 0;
    }

    /// Serializes the used part of the state (layout version 1, little-endian).
    ///
    /// Layout: `b"IMSQ"`, `u32` layout version, `u64` model fingerprint, `u64` position,
    /// `u32` slot count, `u32` scanner count; per slot the `position` cached keys then
    /// values as `f32` (`[position][kv_heads][head_dim]`); per scanner `position + 1`
    /// entries of `u8` flags (bit 0 previous token was a member, bit 1 it was in a run),
    /// `u32` ordinal and `u32` index. Restore with [`crate::SequenceModel::state_from_bytes`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let position = self.step.position;
        let cached: usize = self.step.slots.iter().map(|cache| cache.row).sum();
        let mut bytes = Vec::with_capacity(
            HEADER_LEN
                + 2 * 4 * position * cached
                + self.step.scanners.len() * (position + 1) * SCANNER_ENTRY_LEN,
        );
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&LAYOUT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.fingerprint.to_le_bytes());
        bytes.extend_from_slice(&(position as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.step.slots.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.step.scanners.len() as u32).to_le_bytes());
        for cache in &self.step.slots {
            for values in [&cache.keys, &cache.values] {
                for value in &values[..position * cache.row] {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        for history in &self.step.scanners {
            for entry in &history[..=position] {
                bytes.push(u8::from(entry.prev_member) | (u8::from(entry.prev_run) << 1));
                bytes.extend_from_slice(&entry.ordinal.to_le_bytes());
                bytes.extend_from_slice(&entry.index.to_le_bytes());
            }
        }
        bytes
    }

    /// Fills `self`, a fresh state of the model with `fingerprint`, from [`Self::to_bytes`].
    pub(crate) fn restore(&mut self, bytes: &[u8]) -> Result<(), InstructionModelError> {
        let invalid = |reason: &str| InstructionModelError::InvalidSequenceState {
            reason: reason.to_string(),
        };
        let mut reader = Reader { bytes };
        if reader.take(4)? != MAGIC {
            return Err(invalid("missing IMSQ header"));
        }
        let version = reader.u32()?;
        if version != LAYOUT_VERSION {
            return Err(InstructionModelError::InvalidSequenceState {
                reason: format!("layout version {version} is not {LAYOUT_VERSION}"),
            });
        }
        let fingerprint = reader.u64()?;
        if fingerprint != self.fingerprint {
            return Err(InstructionModelError::SequenceStateModelMismatch {
                expected: self.fingerprint,
                found: fingerprint,
            });
        }
        let position = usize::try_from(reader.u64()?)
            .ok()
            .filter(|&position| position <= self.step.max_positions())
            .ok_or_else(|| invalid("position exceeds max_positions"))?;
        if reader.u32()? as usize != self.step.slots.len()
            || reader.u32()? as usize != self.step.scanners.len()
        {
            return Err(invalid("slot or scanner count differs from the model"));
        }
        for cache in &mut self.step.slots {
            for values in [&mut cache.keys, &mut cache.values] {
                let raw = reader.take(4 * position * cache.row)?;
                for (value, chunk) in values.iter_mut().zip(raw.as_chunks::<4>().0) {
                    *value = f32::from_le_bytes(*chunk);
                }
            }
        }
        for history in &mut self.step.scanners {
            for entry in &mut history[..=position] {
                let flags = reader.take(1)?[0];
                if flags > 0b11 {
                    return Err(invalid("unknown scanner flags"));
                }
                *entry = ScannerState {
                    prev_member: flags & 1 != 0,
                    prev_run: flags & 2 != 0,
                    ordinal: reader.u32()?,
                    index: reader.u32()?,
                };
            }
        }
        if !reader.bytes.is_empty() {
            return Err(invalid("trailing bytes"));
        }
        self.step.position = position;
        Ok(())
    }
}

/// Little-endian reader that fails with a typed error on truncated input.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], InstructionModelError> {
        if len > self.bytes.len() {
            return Err(InstructionModelError::InvalidSequenceState {
                reason: "truncated".to_string(),
            });
        }
        let (head, tail) = self.bytes.split_at(len);
        self.bytes = tail;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32, InstructionModelError> {
        let raw = self.take(4)?;
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn u64(&mut self) -> Result<u64, InstructionModelError> {
        let raw = self.take(8)?;
        let mut array = [0u8; 8];
        array.copy_from_slice(raw);
        Ok(u64::from_le_bytes(array))
    }
}
