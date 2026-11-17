//! Token-by-token inference for causal sequence models (JSON `format_version` 2) with a
//! caller-owned key/value cache: [`SequenceModel`], [`SequenceState`], [`argmax_allowed`].

mod model;
mod state;

pub use model::{SEQUENCE_VALIDATION_TOLERANCE, SequenceModel};
pub(crate) use state::ScannerState;
pub use state::{SequenceState, StepState};

/// The allowed token with the largest logit, taking the first maximum in the order of
/// `allowed` (a NaN logit counts as the maximum, as in NumPy and PyTorch). Ids outside
/// `logits` are skipped; `None` when no allowed id is inside.
pub fn argmax_allowed(logits: &[f32], allowed: &[u32]) -> Option<u32> {
    let candidates = allowed
        .iter()
        .filter_map(|&id| logits.get(id as usize).map(|&logit| (id, logit)));
    first_max(candidates)
}

/// First `(id, value)` with the largest value; the first NaN wins outright.
pub(crate) fn first_max<I: Copy>(candidates: impl Iterator<Item = (I, f32)>) -> Option<I> {
    let mut best: Option<(I, f32)> = None;
    for (id, value) in candidates {
        match best {
            Some((_, top)) if top.is_nan() => break,
            Some((_, top)) if !(value.is_nan() || value > top) => {}
            _ => best = Some((id, value)),
        }
    }
    best.map(|(id, _)| id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_allowed_takes_the_first_maximum_in_allowed_order() {
        let logits = [0.5, 2.0, -1.0, 2.0, 1.0];
        assert_eq!(argmax_allowed(&logits, &[0, 1, 2, 3, 4]), Some(1));
        assert_eq!(argmax_allowed(&logits, &[3, 1]), Some(3));
        assert_eq!(argmax_allowed(&logits, &[4, 0, 2]), Some(4));
        assert_eq!(argmax_allowed(&logits, &[2]), Some(2));
        assert_eq!(argmax_allowed(&logits, &[9, 0]), Some(0));
        assert_eq!(argmax_allowed(&logits, &[]), None);
        assert_eq!(argmax_allowed(&logits, &[5, 6]), None);
        let infinite = [f32::NEG_INFINITY; 3];
        assert_eq!(argmax_allowed(&infinite, &[2, 0]), Some(2));
        let nan = [1.0, f32::NAN, 3.0, f32::NAN];
        assert_eq!(argmax_allowed(&nan, &[0, 2, 3, 1]), Some(3));
        assert_eq!(argmax_allowed(&nan, &[0, 2]), Some(2));
    }
}
