//! Multi-threaded batch prediction.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use crate::errors::{ParallelPredictError, ParallelPredictResult};
use crate::instruction_model::InstructionModel;

/// Output pointer shared by worker threads; each sample index is claimed by exactly
/// one thread, so writes never overlap.
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);

impl SendPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

// SAFETY: see the type docs — threads write disjoint ranges of a buffer that outlives them.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

#[derive(Debug, Clone, Default)]
pub struct PredictConfig {
    threads: Option<usize>,
    slice_result_buffer: Option<(usize, usize)>,
}

impl PredictConfig {
    pub fn new() -> Self {
        Self {
            threads: None,
            slice_result_buffer: None,
        }
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = Some(threads);
        self
    }

    pub fn with_slice_result_buffer(mut self, start: usize, end: usize) -> Self {
        self.slice_result_buffer = Some((start, end));
        self
    }

    pub fn get_threads(&self) -> usize {
        self.threads.unwrap_or_else(|| {
            thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
    }

    pub fn get_slice_range(&self, model: &InstructionModel) -> (usize, usize) {
        self.slice_result_buffer
            .unwrap_or_else(|| (model.get_output_index_start(), model.required_memory()))
    }
}

pub struct ParallelPredictOutput {
    buffer: Vec<f32>,
    num_samples: usize,
    slice_size: usize,
}

impl ParallelPredictOutput {
    fn new(buffer: Vec<f32>, num_samples: usize, slice_size: usize) -> Self {
        Self {
            buffer,
            num_samples,
            slice_size,
        }
    }

    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    pub fn slice_size(&self) -> usize {
        self.slice_size
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.buffer
    }

    pub fn copy_results(&self, dest: &mut [f32]) -> ParallelPredictResult<()> {
        if dest.len() != self.buffer.len() {
            return Err(ParallelPredictError::DestinationBufferSizeMismatch {
                expected: self.buffer.len(),
                actual: dest.len(),
            });
        }
        dest.copy_from_slice(&self.buffer);
        Ok(())
    }

    pub fn copy_results_to_vec(&self) -> Vec<Vec<f32>> {
        (0..self.num_samples)
            .map(|i| {
                let start = i * self.slice_size;
                let end = start + self.slice_size;
                self.buffer[start..end].to_vec()
            })
            .collect()
    }

    pub fn get_result(&self, index: usize) -> ParallelPredictResult<&[f32]> {
        if index >= self.num_samples {
            return Err(ParallelPredictError::ResultIndexOutOfBounds {
                index,
                num_samples: self.num_samples,
            });
        }
        let start = index * self.slice_size;
        let end = start + self.slice_size;
        Ok(&self.buffer[start..end])
    }

    pub fn into_buffer(self) -> Vec<f32> {
        self.buffer
    }
}

pub fn execute_parallel_predict(
    model: &InstructionModel,
    inputs: &[f32],
    config: &PredictConfig,
) -> ParallelPredictResult<ParallelPredictOutput> {
    let feature_size = model.get_feature_size();
    let required_memory = model.required_memory();

    if !inputs.len().is_multiple_of(feature_size) {
        let num_samples = inputs.len() / feature_size;
        let expected = (num_samples + 1) * feature_size;
        return Err(ParallelPredictError::InputBufferSizeMismatch {
            expected,
            actual: inputs.len(),
            num_samples,
            feature_size,
        });
    }

    let num_samples = inputs.len() / feature_size;
    if num_samples == 0 {
        return Ok(ParallelPredictOutput::new(Vec::new(), 0, 0));
    }

    let num_threads = config.get_threads();
    if num_threads == 0 {
        return Err(ParallelPredictError::InvalidThreadCount { count: 0 });
    }

    let (slice_start, slice_end) = config.get_slice_range(model);
    if slice_start >= slice_end {
        return Err(ParallelPredictError::InvalidSliceRange {
            start: slice_start,
            end: slice_end,
        });
    }
    if slice_end > required_memory {
        return Err(ParallelPredictError::SliceRangeOutOfBounds {
            start: slice_start,
            end: slice_end,
            buffer_size: required_memory,
        });
    }

    let slice_size = slice_end - slice_start;
    let mut output_buffer = vec![0.0f32; num_samples * slice_size];
    let mut computation_buffers = vec![0.0f32; num_threads.min(num_samples) * required_memory];
    let sample_counter = AtomicUsize::new(0);
    let output_ptr = SendPtr(output_buffer.as_mut_ptr());

    thread::scope(|scope| {
        let handles: Vec<_> = computation_buffers
            .chunks_mut(required_memory)
            .map(|thread_buffer| {
                let sample_counter = &sample_counter;
                scope.spawn(move || -> ParallelPredictResult<()> {
                    loop {
                        let sample_index = sample_counter.fetch_add(1, Ordering::Relaxed);
                        if sample_index >= num_samples {
                            return Ok(());
                        }
                        let input_start = sample_index * feature_size;
                        thread_buffer[..feature_size]
                            .copy_from_slice(&inputs[input_start..input_start + feature_size]);
                        model.predict_with_buffer(thread_buffer).map_err(|e| {
                            ParallelPredictError::PredictionFailed {
                                sample_index,
                                message: e.to_string(),
                            }
                        })?;
                        let result = &thread_buffer[slice_start..slice_end];
                        // SAFETY: `sample_index < num_samples` was claimed by this thread only,
                        // so this `slice_size` range of `output_buffer` is in bounds and unaliased.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                result.as_ptr(),
                                output_ptr.get().add(sample_index * slice_size),
                                slice_size,
                            );
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle
                .join()
                .map_err(|_| ParallelPredictError::ThreadPanicked)??;
        }
        Ok(())
    })?;

    Ok(ParallelPredictOutput::new(
        output_buffer,
        num_samples,
        slice_size,
    ))
}
