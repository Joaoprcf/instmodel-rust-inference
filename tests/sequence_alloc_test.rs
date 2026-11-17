//! `SequenceModel::step` must not allocate once the state exists. This binary counts the
//! allocations made by the test thread through a wrapping global allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::Path;

use instmodel_inference::{InstructionModelInfo, SequenceModel};

struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: forwards every call to the system allocator unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn stepping_does_not_allocate() {
    for name in ["graph-w20", "looped-w28"] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/sequence")
            .join(format!("{name}.json"));
        let text = std::fs::read_to_string(path).unwrap();
        let info: InstructionModelInfo = serde_json::from_str(&text).unwrap();
        let tokens = info.validation_sequences.as_ref().unwrap()[0]
            .tokens
            .clone();
        let model = SequenceModel::new(info).unwrap();
        let mut state = model.new_state();

        let before = ALLOCATIONS.with(Cell::get);
        assert!(
            before > 0,
            "the counter sees the allocations made while loading"
        );
        let mut checksum = 0.0f32;
        for &token in &tokens {
            checksum += model.step(&mut state, token).unwrap()[0];
        }
        state.truncate(10).unwrap();
        for &token in &tokens[10..] {
            checksum += model.step(&mut state, token).unwrap()[0];
        }
        let allocations = ALLOCATIONS.with(Cell::get) - before;
        assert_eq!(allocations, 0, "{name}: {} tokens", tokens.len());
        assert!(checksum.is_finite());
    }
}
