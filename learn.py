import cupy as cp
import time

# Check GPU availability
if not cp.cuda.is_available():
    print("GPU not detected! Ensure CUDA and CuPy are properly installed.")
else:
    device = cp.cuda.Device(0)
    print(f"Using GPU: {device.name}")

# Matrix size (change for bigger tests)
N = 10000  # 10,000 x 10,000 matrix

# Create large random matrices on GPU
A = cp.random.rand(N, N, dtype=cp.float32)
B = cp.random.rand(N, N, dtype=cp.float32)

# Measure computation time
cp.cuda.Device(0).synchronize()  # Ensure GPU is ready
start = time.time()
C = cp.dot(A, B)  # Matrix multiplication
cp.cuda.Device(0).synchronize()  # Wait for GPU to finish
end = time.time()

print(f"Matrix multiplication completed in {end - start:.4f} seconds")
