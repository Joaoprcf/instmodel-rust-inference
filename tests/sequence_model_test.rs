//! Format-2 sequence models: fixture replay, state handling and errors.
//!
//! The fixtures under `tests/fixtures/sequence` come from instmodel
//! (`tests/make_sequence_fixtures.py`) and binary-es (the calculator controllers); each
//! carries reference logits per token computed by instmodel's NumPy reference.

use std::path::Path;

use instmodel_inference::errors::InstructionModelError as E;
use instmodel_inference::{
    InstructionModel, InstructionModelInfo, SequenceModel, SequenceState, argmax_allowed,
};
use serde_json::{Value, json};

const FIXTURES: [&str; 8] = [
    "plain",
    "features",
    "grouped",
    "untied",
    "looped",
    "stacked_loops",
    "graph-w20",
    "looped-w28",
];

fn fixture_json(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sequence")
        .join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    serde_json::from_str(&text).unwrap()
}

fn info(model: Value) -> InstructionModelInfo {
    serde_json::from_value(model).unwrap()
}

fn load(name: &str) -> SequenceModel {
    SequenceModel::new(info(fixture_json(name))).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn first_tokens(name: &str) -> Vec<u32> {
    let model = fixture_json(name);
    serde_json::from_value(model["validation_sequences"][0]["tokens"].clone()).unwrap()
}

/// Logits after every token, from a fresh state.
fn run(model: &SequenceModel, tokens: &[u32]) -> Vec<Vec<f32>> {
    let mut state = model.new_state();
    tokens
        .iter()
        .map(|&token| model.step(&mut state, token).unwrap().to_vec())
        .collect()
}

fn feed(model: &SequenceModel, state: &mut SequenceState, tokens: &[u32]) -> Vec<Vec<f32>> {
    tokens
        .iter()
        .map(|&token| model.step(state, token).unwrap().to_vec())
        .collect()
}

fn first_argmax(values: &[f32]) -> usize {
    let ids: Vec<u32> = (0..values.len() as u32).collect();
    argmax_allowed(values, &ids).unwrap() as usize
}

#[test]
fn every_fixture_matches_its_reference_logits() {
    for name in FIXTURES {
        let json = fixture_json(name);
        let model = load(name);
        let sequences = json["validation_sequences"].as_array().unwrap();
        assert!(!sequences.is_empty(), "{name}");
        let mut worst = 0.0f32;
        let mut tokens_checked = 0;
        for case in sequences {
            let tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
            let expected: Vec<Vec<f32>> = serde_json::from_value(case["logits"].clone()).unwrap();
            for (position, (logits, reference)) in
                run(&model, &tokens).iter().zip(&expected).enumerate()
            {
                assert_eq!(logits.len(), model.vocab_size());
                for (a, b) in logits.iter().zip(reference) {
                    worst = worst.max((a - b).abs());
                }
                assert_eq!(
                    first_argmax(logits),
                    first_argmax(reference),
                    "{name} position {position}"
                );
                tokens_checked += 1;
            }
        }
        assert!(worst <= 1e-3, "{name}: worst logit error {worst}");
        println!("{name}: {tokens_checked} tokens, worst logit error {worst:.3e}");
    }
}

#[test]
fn accessors_expose_the_sequence_block() {
    let model = load("graph-w20");
    assert_eq!(model.vocab_size(), 28);
    assert_eq!(model.vocabulary()[10], "+");
    assert_eq!(model.vocabulary()[27], "NULL");
    assert_eq!(model.max_positions(), 192);
    assert_eq!(model.metadata()["protocol"]["shortcut"], 27);
    assert_eq!(model.config()["width"], 20);
    let state = model.new_state();
    assert_eq!((state.position(), state.max_positions()), (0, 192));
}

#[test]
fn cloned_states_diverge_independently() {
    let model = load("graph-w20");
    let tokens = first_tokens("graph-w20");
    let whole = run(&model, &tokens);
    let mut state = model.new_state();
    feed(&model, &mut state, &tokens[..30]);
    let kept = state.clone();
    // A discarded continuation must not leak into the clone.
    feed(&model, &mut state, &[3, 3, 12, 7, 13]);
    let mut resumed = kept.clone();
    assert_eq!(feed(&model, &mut resumed, &tokens[30..]), whole[30..]);
    assert_eq!(kept.position(), 30);
    assert_eq!(state.position(), 35);
}

#[test]
fn truncation_equals_an_uninterrupted_run() {
    for name in ["graph-w20", "looped", "features"] {
        let model = load(name);
        let tokens = first_tokens(name);
        let whole = run(&model, &tokens);
        let mut state = model.new_state();
        feed(&model, &mut state, &tokens);
        // Rewind into the middle of a digit run, so the scanner history matters.
        for cut in [tokens.len() - 1, 9, 3, 0] {
            state.truncate(cut).unwrap();
            assert_eq!(state.position(), cut);
            assert_eq!(
                feed(&model, &mut state, &tokens[cut..]),
                whole[cut..],
                "{name} at {cut}"
            );
        }
        assert!(matches!(
            state.truncate(tokens.len() + 1),
            Err(E::TruncatePastPosition { .. })
        ));
    }
}

#[test]
fn reset_starts_over() {
    let model = load("features");
    let tokens = first_tokens("features");
    let mut state = model.new_state();
    feed(&model, &mut state, &tokens);
    state.reset();
    assert_eq!(state.position(), 0);
    assert_eq!(feed(&model, &mut state, &tokens), run(&model, &tokens));
    state.reset();
    assert_eq!(state.to_bytes(), model.new_state().to_bytes());
}

#[test]
fn bytes_round_trip_mid_sequence() {
    for name in ["graph-w20", "looped-w28", "untied"] {
        let model = load(name);
        let tokens = first_tokens(name);
        let whole = run(&model, &tokens);
        let mut state = model.new_state();
        feed(&model, &mut state, &tokens[..11]);
        let bytes = state.to_bytes();
        let mut restored = model.state_from_bytes(&bytes).unwrap();
        assert_eq!(restored.position(), 11);
        assert_eq!(restored.to_bytes(), bytes);
        // Bit-identical continuation.
        assert_eq!(
            feed(&model, &mut restored, &tokens[11..]),
            whole[11..],
            "{name}"
        );
        // An empty state round-trips as well.
        let empty = model
            .state_from_bytes(&model.new_state().to_bytes())
            .unwrap();
        assert_eq!(empty.position(), 0);
    }
}

#[test]
fn bytes_are_bound_to_their_model() {
    let graph = load("graph-w20");
    let looped = load("looped-w28");
    let mut state = graph.new_state();
    feed(&graph, &mut state, &first_tokens("graph-w20")[..5]);
    let bytes = state.to_bytes();
    assert!(matches!(
        looped.state_from_bytes(&bytes),
        Err(E::SequenceStateModelMismatch { .. })
    ));
    // Same structure, different weights: the fingerprint still differs.
    let mut changed = fixture_json("graph-w20");
    changed["weights"][3][0][0] = json!(changed["weights"][3][0][0].as_f64().unwrap() + 0.5);
    changed["validation_sequences"] = json!([]);
    let changed = SequenceModel::new(info(changed)).unwrap();
    assert!(matches!(
        changed.state_from_bytes(&bytes),
        Err(E::SequenceStateModelMismatch { .. })
    ));
    assert!(matches!(
        looped.step(&mut state, 0),
        Err(E::SequenceStateModelMismatch { .. })
    ));
}

#[test]
fn malformed_bytes_are_rejected() {
    let model = load("features");
    let mut state = model.new_state();
    feed(&model, &mut state, &[1, 2, 11, 3]);
    let bytes = state.to_bytes();
    let mut cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        ("truncated", bytes[..bytes.len() - 1].to_vec()),
        ("trailing", [bytes.as_slice(), &[0]].concat()),
    ];
    let mut magic = bytes.clone();
    magic[0] = b'X';
    cases.push(("magic", magic));
    let mut version = bytes.clone();
    version[4] = 9;
    cases.push(("version", version));
    let mut position = bytes.clone();
    position[16..24].copy_from_slice(&1000u64.to_le_bytes());
    cases.push(("position", position));
    let mut flags = bytes.clone();
    let flags_at = bytes.len() - 9;
    flags[flags_at] = 7;
    cases.push(("flags", flags));
    for (label, bytes) in cases {
        assert!(
            matches!(
                model.state_from_bytes(&bytes),
                Err(E::InvalidSequenceState { .. })
            ),
            "{label}"
        );
    }
}

#[test]
fn failed_steps_leave_the_state_unchanged() {
    let model = load("plain");
    let tokens = first_tokens("plain");
    let whole = run(&model, &tokens);
    let mut state = model.new_state();
    feed(&model, &mut state, &tokens[..4]);
    let before = state.to_bytes();
    assert!(matches!(
        model.step(&mut state, 14),
        Err(E::TokenOutOfVocabulary {
            token: 14,
            vocab_size: 14
        })
    ));
    assert!(matches!(
        model.step(&mut state, u32::MAX),
        Err(E::TokenOutOfVocabulary { .. })
    ));
    assert_eq!(state.to_bytes(), before);
    assert_eq!(feed(&model, &mut state, &tokens[4..]), whole[4..]);

    // max_positions is 24 for the small fixtures.
    let mut full = model.new_state();
    let long: Vec<u32> = (0..24).map(|t| t % 14).collect();
    feed(&model, &mut full, &long);
    let before = full.to_bytes();
    assert!(matches!(
        model.step(&mut full, 1),
        Err(E::SequenceFull { max_positions: 24 })
    ));
    assert_eq!(full.to_bytes(), before);
    full.truncate(23).unwrap();
    assert!(model.step(&mut full, 1).is_ok());
}

/// Token table of 2 rows behind a vocabulary of 3, so token 2 fails inside the program.
fn short_table_model() -> Value {
    json!({
        "format_version": 2, "features": ["token"], "buffer_sizes": [1, 2, 3, 1, 3],
        "instructions": [
            {"type": "EMBEDDING", "input": 0, "output": 1, "weights": 0},
            {"type": "DOT", "input": 1, "output": 2, "weights": 1},
            {"type": "CAUSAL_ATTENTION", "input": 2, "output": 3, "heads": 1, "kv_heads": 1, "slot": 0},
            {"type": "DOT", "input": 3, "output": 4, "weights": 2}],
        "weights": [[[1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                    [[1.0], [-1.0], [2.0]]],
        "bias": [[0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
        "sequence": {"vocabulary": ["a", "b", "c"], "max_positions": 4,
                     "kv_slots": [{"heads": 1, "kv_heads": 1, "head_dim": 1}], "scanners": 0}
    })
}

#[test]
fn hand_built_model_steps_and_reports_bad_ids() {
    let model = SequenceModel::new(info(short_table_model())).unwrap();
    let mut state = model.new_state();
    // Token 0: e = [1, 0], qkv = [1, 0, 1]; one cached position, so attention = v = 1.
    assert_eq!(model.step(&mut state, 0).unwrap(), [1.0, -0.5, 2.0]);
    // Token 1: e = [0, 1], qkv = [0, 1, 1]; q = 0 weights both positions equally: (1 + 1) / 2.
    assert_eq!(model.step(&mut state, 1).unwrap(), [1.0, -0.5, 2.0]);
    let before = state.to_bytes();
    assert!(matches!(
        model.step(&mut state, 2),
        Err(E::EmbeddingIdOutOfRange { rows: 2, .. })
    ));
    assert_eq!(state.to_bytes(), before);
    assert_eq!(state.position(), 2);
}

#[test]
fn construction_errors() {
    type Edit = fn(&mut Value);
    type IsExpected = fn(&E) -> bool;
    let edits: Vec<(&str, Edit, IsExpected)> = vec![
        (
            "bad slot",
            |m| m["instructions"][2]["slot"] = json!(1),
            |e| matches!(e, E::KvSlotIndexOutOfBounds { index: 1 }),
        ),
        (
            "heads differ from slot",
            |m| m["instructions"][2]["heads"] = json!(2),
            |e| matches!(e, E::AttentionHeadsMismatch { .. }),
        ),
        (
            "attention input size",
            |m| m["sequence"]["kv_slots"][0]["head_dim"] = json!(2),
            |e| matches!(e, E::InputBufferSizeMismatch { index: 2, .. }),
        ),
        (
            "unused slot",
            |m| {
                m["sequence"]["kv_slots"] = json!([{"heads": 1, "kv_heads": 1, "head_dim": 1},
                                                   {"heads": 1, "kv_heads": 1, "head_dim": 1}])
            },
            |e| matches!(e, E::UnusedSequenceState { index: 1, .. }),
        ),
        (
            "kv_heads must divide heads",
            |m| m["sequence"]["kv_slots"][0]["kv_heads"] = json!(2),
            |e| matches!(e, E::InvalidSequenceInfo { .. }),
        ),
        (
            "vocabulary size",
            |m| m["sequence"]["vocabulary"] = json!(["a", "b"]),
            |e| matches!(e, E::InvalidSequenceInfo { .. }),
        ),
        (
            "max_positions",
            |m| m["sequence"]["max_positions"] = json!(0),
            |e| matches!(e, E::InvalidSequenceInfo { .. }),
        ),
        (
            "format version",
            |m| m["format_version"] = json!(1),
            |e| matches!(e, E::NotSequenceModel { .. }),
        ),
        (
            "embedding width",
            |m| m["weights"][0] = json!([[1.0], [0.0]]),
            |e| matches!(e, E::OutputBufferSizeMismatch { index: 1, .. }),
        ),
        (
            "validation logits",
            |m| m["validation_sequences"] = json!([{"tokens": [0], "logits": [[1.0, -0.5, 2.1]]}]),
            |e| matches!(e, E::SequenceValidationMismatch { position: 0, .. }),
        ),
        (
            "validation shape",
            |m| {
                m["validation_sequences"] =
                    json!([{"tokens": [0, 1], "logits": [[1.0, -0.5, 2.0]]}])
            },
            |e| matches!(e, E::ValidationSequenceShape { sequence: 0, .. }),
        ),
    ];
    for (label, edit, check) in edits {
        let mut model = short_table_model();
        edit(&mut model);
        let error = SequenceModel::new(info(model)).expect_err(label);
        assert!(check(&error), "{label}: {error}");
    }

    let mut scanner = fixture_json("features");
    scanner["instructions"][1]["scanner"] = json!(3);
    assert!(matches!(
        SequenceModel::new(info(scanner)),
        Err(E::ScannerIndexOutOfBounds { index: 3 })
    ));
    assert!(matches!(
        SequenceModel::new(info(json!({"feature_size": 1, "buffer_sizes": [1, 1],
            "instructions": [{"type": "COPY", "input": 0, "output": 1, "internal_index": 0}],
            "weights": [], "bias": []}))),
        Err(E::NotSequenceModel { .. })
    ));
    // Within tolerance and with the same argmax, the replay passes.
    let mut close = short_table_model();
    close["validation_sequences"] =
        json!([{"tokens": [0, 1], "logits": [[1.0005, -0.5, 2.0], [1.0, -0.5, 2.0009]]}]);
    assert!(SequenceModel::new(info(close)).is_ok());
}

#[test]
fn stateless_predictor_rejects_sequence_models() {
    for name in ["plain", "graph-w20"] {
        assert!(matches!(
            InstructionModel::new(info(fixture_json(name))),
            Err(E::StatefulInstruction {
                instruction_type,
                ..
            }) if instruction_type == "CAUSAL_ATTENTION" || instruction_type == "SEGMENT_FEATURES"
        ));
    }
    let mut future = fixture_json("plain");
    future["format_version"] = json!(3);
    assert!(matches!(
        InstructionModel::new(info(future)),
        Err(E::UnsupportedFormatVersion { version: 3 })
    ));
}

#[test]
fn one_model_steps_many_sequences_across_threads() {
    let model = load("graph-w20");
    let json = fixture_json("graph-w20");
    let sequences: Vec<Vec<u32>> = json["validation_sequences"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| serde_json::from_value(case["tokens"].clone()).unwrap())
        .collect();
    let serial: Vec<Vec<Vec<f32>>> = sequences.iter().map(|t| run(&model, t)).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|worker| {
                let (model, tokens) = (&model, &sequences[worker % sequences.len()]);
                scope.spawn(move || run(model, tokens))
            })
            .collect();
        for (worker, handle) in handles.into_iter().enumerate() {
            assert_eq!(handle.join().unwrap(), serial[worker % sequences.len()]);
        }
    });
}

#[test]
fn model_and_state_are_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<SequenceModel>();
    check::<SequenceState>();
}
