use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use crate::errors::{ParallelPredictError, ParallelPredictResult};
use crate::instruction_model::InstructionModel;

#[derive(Clone, Copy)]
struct SendPtr {
    ptr: *mut f32,
}

impl SendPtr {
    fn new(ptr: *mut f32) -> Self {
        Self { ptr }
    }

    unsafe fn add(self, offset: usize) -> *mut f32 {
        unsafe { self.ptr.add(offset) }
    }

    unsafe fn as_slice_mut(self, offset: usize, len: usize) -> &'static mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(offset), len) }
    }
}

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
    let mut computation_buffers = vec![0.0f32; num_threads * required_memory];

    let sample_counter = AtomicUsize::new(0);

    thread::scope(|scope| {
        let sample_counter_ref = &sample_counter;
        let output_buffer_ptr = SendPtr::new(output_buffer.as_mut_ptr());
        let computation_buffers_ptr = SendPtr::new(computation_buffers.as_mut_ptr());

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                scope.spawn(move || -> ParallelPredictResult<()> {
                    let thread_buffer_start = thread_id * required_memory;
                    let thread_buffer = unsafe {
                        computation_buffers_ptr.as_slice_mut(thread_buffer_start, required_memory)
                    };

                    loop {
                        let sample_index = sample_counter_ref.fetch_add(1, Ordering::Relaxed);
                        if sample_index >= num_samples {
                            break;
                        }

                        let input_start = sample_index * feature_size;
                        let input_end = input_start + feature_size;
                        thread_buffer[..feature_size]
                            .copy_from_slice(&inputs[input_start..input_end]);

                        model.predict_with_buffer(thread_buffer).map_err(|e| {
                            ParallelPredictError::PredictionFailed {
                                sample_index,
                                message: e.to_string(),
                            }
                        })?;

                        let output_start = sample_index * slice_size;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                thread_buffer.as_ptr().add(slice_start),
                                output_buffer_ptr.add(output_start),
                                slice_size,
                            );
                        }
                    }

                    Ok(())
                })
            })
            .collect();

        for handle in handles {
            match handle.join() {
                Ok(result) => result?,
                Err(_) => return Err(ParallelPredictError::ThreadPanicked),
            }
        }

        Ok(())
    })?;

    Ok(ParallelPredictOutput::new(
        output_buffer,
        num_samples,
        slice_size,
    ))
}
