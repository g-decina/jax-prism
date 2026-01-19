
import time
import os
import shutil
import numpy as np
import polars as pl
import jax

from src.jax_forecasting.core.data_loader import JAXDataLoader

# Force 4-device simulation for the benchmark
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# Import your class (adjust import path as needed)
# from src.core.data_loader import JAXDataLoader 
# assuming you pasted the class in the same file for testing or import it:

def generate_dummy_data(path: str, rows: int, cols: int):
    print(f"Generating {rows:,} rows of dummy data...")
    # Create random data
    data = np.random.randn(rows, cols).astype(np.float32)
    schema = [f"feat_{i}" for i in range(cols)]
    
    # Save as Parquet
    df = pl.DataFrame(data, schema=schema)
    df.write_parquet(path)
    print(f"Saved to {path} ({os.path.getsize(path) / 1024**2:.2f} MB)")
    return schema

def benchmark():
    # Config
    TMP_FILE = "benchmark_data.parquet"
    N_ROWS = 1_000_000
    N_COLS = 50
    BATCH_SIZE = 1024  # Global batch size
    TIME_STEPS = 60
    
    try:
        features = generate_dummy_data(TMP_FILE, N_ROWS, N_COLS)
        
        print("\n--- Starting Benchmark ---")
        print(f"Device Count: {jax.local_device_count()}")
        
        loader = JAXDataLoader(
            source=TMP_FILE, 
            batch_size=BATCH_SIZE, 
            time_steps=TIME_STEPS, 
            features=features,
            buffer_size=100_000
        )
        
        start_time = time.time()
        batch_count = 0
        
        # We assume the first batch incurs JIT/Alloc overhead, so we track it
        first_batch_time = 0
        
        for batch_x, batch_mask in loader:
            # Simulate device consumption (block untill data is actually on device)
            batch_x.block_until_ready()
            
            if batch_count == 0:
                first_batch_time = time.time() - start_time
                print(f"First batch latency: {first_batch_time:.4f}s (Includes initialization)")
            
            batch_count += 1
            
            # Simple shape check to ensure padding works
            assert batch_x.shape == (4, 256, 60, 50)
            
        total_time = time.time() - start_time
        avg_time = (total_time - first_batch_time) / (batch_count - 1)
        throughput = (batch_count * BATCH_SIZE) / total_time
        
        print(f"\nTotal Batches: {batch_count}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} samples/sec")
        print(f"Latency per batch: {avg_time*1000:.2f} ms")
        
        if throughput < 5000:
            print("\n[CRITICAL WARN] Throughput is suspiciously low. Check disk I/O or padding logic.")
        else:
            print("\n[PASS] Data Loader is performant.")

    finally:
        if os.path.exists(TMP_FILE):
            os.remove(TMP_FILE)

if __name__ == "__main__":
    benchmark()