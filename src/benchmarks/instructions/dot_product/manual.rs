use crate::benchmarks::benchmark_types::DotProductConfig;

/// Manual implementation of the dot product neural network benchmark.
pub struct ManualDotProduct {
    weights_layer1: Vec<Vec<f32>>,
    bias_layer1: Vec<f32>,
    weights_layer2: Vec<Vec<f32>>,
    bias_layer2: Vec<f32>,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
}

impl ManualDotProduct {
    pub fn new(
        weights_layer1: Vec<Vec<f32>>,
        bias_layer1: Vec<f32>,
        weights_layer2: Vec<Vec<f32>>,
        bias_layer2: Vec<f32>,
    ) -> Self {
        let input_size = weights_layer1.get(0).map_or(0, |row| row.len());
        let hidden_size = weights_layer1.len();
        let output_size = weights_layer2.len();

        Self {
            weights_layer1,
            bias_layer1,
            weights_layer2,
            bias_layer2,
            input_size,
            hidden_size,
            output_size,
        }
    }

    pub fn required_buffer_len(&self) -> usize {
        self.input_size + self.hidden_size + self.output_size
    }

    /// Executes the forward pass using a pre-allocated unified buffer.
    ///
    /// Layout: | input (input_size) | hidden (hidden_size) | output (output_size) |
    pub fn compute_with_buffer<'a>(
        &'a self,
        input: &[f32],
        unified_buffer: &'a mut [f32],
    ) -> &'a [f32] {
        assert_eq!(input.len(), self.input_size);
        assert!(unified_buffer.len() >= self.required_buffer_len());

        let (input_region, tail) = unified_buffer.split_at_mut(self.input_size);
        input_region.copy_from_slice(input);
        let (hidden_region, output_region) = tail.split_at_mut(self.hidden_size);

        // Layer 1: Dense + ReLU
        for (i, weights_row) in self.weights_layer1.iter().enumerate() {
            let mut sum = self.bias_layer1[i];
            for (j, &weight) in weights_row.iter().enumerate() {
                sum += weight * input_region[j];
            }
            hidden_region[i] = if sum > 0.0 { sum } else { 0.0 };
        }

        // Layer 2: Dense + Sigmoid
        for (i, weights_row) in self.weights_layer2.iter().enumerate() {
            let mut sum = self.bias_layer2[i];
            for (j, &weight) in weights_row.iter().enumerate() {
                sum += weight * hidden_region[j];
            }
            output_region[i] = 1.0 / (1.0 + (-sum).exp());
        }

        output_region
    }
}

/// Creates deterministic test data for the dot product benchmark.
pub fn create_dot_product_test_data(
    config: &DotProductConfig,
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
    let input_size = config.network_config.input_size;
    let hidden_size = config.network_config.hidden_size;
    let output_size = config.network_config.output_size;

    let weights_layer1: Vec<Vec<f32>> = (0..hidden_size)
        .map(|i| {
            (0..input_size)
                .map(|j| 0.1 + (i as f32 + j as f32) * 0.0001)
                .collect()
        })
        .collect();

    let bias_layer1: Vec<f32> = (0..hidden_size)
        .map(|i| 0.01 + (i as f32) * 0.00001)
        .collect();

    let weights_layer2: Vec<Vec<f32>> = (0..output_size)
        .map(|i| {
            (0..hidden_size)
                .map(|j| 0.001 + (i as f32 + j as f32) * 0.000001)
                .collect()
        })
        .collect();

    let bias_layer2: Vec<f32> = (0..output_size).map(|i| 0.1 + (i as f32) * 0.01).collect();

    let inputs = vec![0.5, -0.3];

    (
        weights_layer1,
        bias_layer1,
        weights_layer2,
        bias_layer2,
        inputs,
    )
}
