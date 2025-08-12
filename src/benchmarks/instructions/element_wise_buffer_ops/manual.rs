use crate::benchmarks::benchmark_types::ElementWiseOpsConfig;

pub struct ManualElementWiseOps {
    buffer_sizes: Vec<usize>,
    input_size: usize,
}

impl ManualElementWiseOps {
    pub fn new(config: &ElementWiseOpsConfig) -> Self {
        Self {
            buffer_sizes: config.buffer_sizes.clone(),
            input_size: config.input_size,
        }
    }

    pub fn required_buffer_len(&self) -> usize {
        self.buffer_sizes.iter().sum()
    }

    pub fn compute_with_buffer<'a>(
        &'a self,
        input: &[f32],
        unified_buffer: &'a mut [f32],
    ) -> (&'a [f32], &'a [f32]) {
        assert_eq!(input.len(), self.input_size);
        assert!(unified_buffer.len() >= self.required_buffer_len());

        unified_buffer[..input.len()].copy_from_slice(input);

        let buffer0_size = self.buffer_sizes[0];
        let buffer1_size = self.buffer_sizes[1];
        let buffer2_size = self.buffer_sizes[2];
        let buffer3_size = self.buffer_sizes[3];

        let buffer0_start = 0;
        let buffer1_start = buffer0_start + buffer0_size;
        let buffer2_start = buffer1_start + buffer1_size;
        let buffer3_start = buffer2_start + buffer2_size;

        let add_limit = buffer2_size.min(buffer0_size).min(buffer1_size);
        for i in 0..add_limit {
            unified_buffer[buffer2_start + i] =
                unified_buffer[buffer0_start + i] + unified_buffer[buffer1_start + i];
        }

        let mul_limit = buffer3_size.min(buffer0_size).min(buffer1_size);
        for i in 0..mul_limit {
            unified_buffer[buffer3_start + i] =
                unified_buffer[buffer0_start + i] * unified_buffer[buffer1_start + i];
        }

        (
            &unified_buffer[buffer2_start..buffer2_start + buffer2_size],
            &unified_buffer[buffer3_start..buffer3_start + buffer3_size],
        )
    }
}

pub fn create_element_wise_test_data(config: &ElementWiseOpsConfig) -> Vec<f32> {
    let mut input = Vec::with_capacity(config.input_size);
    for i in 0..config.input_size {
        input.push((i as f32 * 0.001) % 2.0 - 1.0);
    }
    input
}
