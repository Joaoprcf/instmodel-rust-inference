import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import time
import tensorflow as tf

INPUT_SIZE = 250
HIDDEN_SIZE = 300
OUTPUT_SIZE = 200
NUM_SAMPLES = 200_000
BATCH_SIZE = 8192

def format_bytes(b):
    if b >= 1024 * 1024:
        return f"{b / (1024 * 1024):.2f} MB"
    elif b >= 1024:
        return f"{b / 1024:.2f} KB"
    return f"{b} B"

def main():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(INPUT_SIZE,), dtype=tf.float32),
        tf.keras.layers.Dense(HIDDEN_SIZE, activation="relu"),
        tf.keras.layers.Dense(OUTPUT_SIZE, activation="sigmoid"),
    ])
    model.compile(optimizer='adam', loss='mse')

    inputs = np.random.randn(NUM_SAMPLES, INPUT_SIZE).astype(np.float32)

    # Warmup
    _ = model.predict(inputs[:BATCH_SIZE], verbose=0, batch_size=BATCH_SIZE)

    # Benchmark
    start = time.perf_counter()
    outputs = model.predict(inputs, verbose=0, batch_size=BATCH_SIZE)
    duration = time.perf_counter() - start

    print(f"TensorFlow CPU: {duration:.3f}s ({NUM_SAMPLES / duration:.3f} inferences/sec)")

    # Model memory (weights + bias)
    model_params = model.count_params()
    model_bytes = model_params * 4  # float32

    # Memory footprint
    input_bytes = inputs.nbytes
    output_bytes = outputs.nbytes
    batch_buffer_bytes = BATCH_SIZE * (INPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE) * 4
    total_bytes = input_bytes + output_bytes + batch_buffer_bytes

    print("\n=== Memory Footprint ===")
    print(f"Model (weights+bias): {format_bytes(model_bytes)} ({model_params:,} params)")
    print(f"Input buffer:         {format_bytes(input_bytes)}")
    print(f"Output buffer:        {format_bytes(output_bytes)}")
    print(f"Batch buffer (~):     {format_bytes(batch_buffer_bytes)}")
    print(f"Total (~):            {format_bytes(total_bytes)}")

if __name__ == "__main__":
    main()