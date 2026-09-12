import time
import torch

device = "cuda"

SIZE = 4096
DURATION = 30

a = torch.randn(SIZE, SIZE, device=device)
b = torch.randn(SIZE, SIZE, device=device)

# Warm up CUDA and GEMM path.
for _ in range(5):
    _ = a @ b

torch.cuda.synchronize()

print("GPU:", torch.cuda.get_device_name(0))
print(f"Generating controlled GPU load for ~{DURATION} seconds...")

start = time.perf_counter()
iterations = 0

while True:
    c = a @ b

    # Important:
    # Ensure this iteration actually completes before
    # evaluating elapsed host wall-clock time.
    torch.cuda.synchronize()

    iterations += 1

    elapsed = time.perf_counter() - start

    if elapsed >= DURATION:
        break

print()
print(f"Iterations:  {iterations}")
print(f"Elapsed:     {elapsed:.2f} seconds")
print(f"Matmuls/sec: {iterations / elapsed:.2f}")
