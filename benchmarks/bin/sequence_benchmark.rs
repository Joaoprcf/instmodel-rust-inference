//! Token-by-token sequence model benchmark: microseconds per `SequenceModel::step` over
//! a 60-token sequence for the calculator controllers in `tests/fixtures/sequence`, plus
//! the cost of cloning and serializing a state at that length.
//!
//! Run with: `cargo run --release --bin sequence_benchmark [MODEL.json ...]`

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use instmodel_inference::errors::InstructionModelError;
use instmodel_inference::{InstructionModelInfo, SequenceModel};
use log::{error, info};
use thiserror::Error;

const TOKENS: usize = 60;
const SEQUENCES: u32 = 2_000;

#[derive(Error, Debug)]
enum BenchError {
    #[error("{path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}: {source}")]
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("{path}: the first validation sequence has fewer than {TOKENS} tokens")]
    ShortSequence { path: PathBuf },
    #[error(transparent)]
    Model(#[from] InstructionModelError),
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sequence");
    let mut paths: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if paths.is_empty() {
        paths = vec![
            fixtures.join("graph-w20.json"),
            fixtures.join("looped-w28.json"),
        ];
    }
    for path in paths {
        if let Err(e) = bench(&path) {
            error!("sequence_benchmark failed: {e}");
            std::process::exit(1);
        }
    }
}

fn bench(path: &Path) -> Result<(), BenchError> {
    let text = std::fs::read_to_string(path).map_err(|source| BenchError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let info: InstructionModelInfo =
        serde_json::from_str(&text).map_err(|source| BenchError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
    // The stored validation sequence is a real controller episode.
    let tokens: Vec<u32> = info
        .validation_sequences
        .as_ref()
        .and_then(|sequences| sequences.first())
        .map(|case| case.tokens.clone())
        .filter(|tokens| tokens.len() >= TOKENS)
        .ok_or_else(|| BenchError::ShortSequence {
            path: path.to_path_buf(),
        })?;
    let tokens = &tokens[..TOKENS];

    let start = Instant::now();
    let model = SequenceModel::new(info)?;
    let load_ms = start.elapsed().as_secs_f64() * 1e3;

    let mut state = model.new_state();
    for _ in 0..SEQUENCES / 10 {
        state.reset();
        for &token in tokens {
            black_box(model.step(&mut state, token)?);
        }
    }
    let start = Instant::now();
    for _ in 0..SEQUENCES {
        state.reset();
        for &token in tokens {
            black_box(model.step(&mut state, token)?);
        }
    }
    let per_token_us = start.elapsed().as_secs_f64() * 1e6 / (f64::from(SEQUENCES) * TOKENS as f64);

    let start = Instant::now();
    for _ in 0..SEQUENCES {
        black_box(state.clone());
    }
    let clone_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(SEQUENCES);
    let start = Instant::now();
    let mut bytes = 0;
    for _ in 0..SEQUENCES {
        bytes = black_box(state.to_bytes()).len();
    }
    let to_bytes_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(SEQUENCES);

    let name = path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    info!(
        "== {name}: {} tokens, max_positions {} ==",
        model.vocab_size(),
        model.max_positions()
    );
    info!("load + validate:            {load_ms:>9.2} ms");
    info!("step ({TOKENS}-token sequence): {per_token_us:>9.3} µs/token");
    info!("clone at {TOKENS} tokens:       {clone_us:>9.3} µs");
    info!("to_bytes at {TOKENS} tokens:    {to_bytes_us:>9.3} µs ({bytes} bytes)");
    Ok(())
}
