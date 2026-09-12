import time
import torch

DEVICE = "cuda"
SIZE = 4096

a = torch.randn((SIZE, SIZE), device=DEVICE)
b = torch.randn((SIZE, SIZE), device=DEVICE)

# Warmup
for _ in range(5):
    _ = a @ b
torch.cuda.synchronize()

# ---------------------------------------------------------
# Method 1: naive CPU wall-clock timing
# ---------------------------------------------------------
cpu_start = time.perf_counter()

c = a @ b

cpu_elapsed_without_sync = time.perf_counter() - cpu_start

# ---------------------------------------------------------
# Method 2: CPU wall-clock timing WITH synchronization
# ---------------------------------------------------------
torch.cuda.synchronize()

cpu_start = time.perf_counter()

c = a @ b
torch.cuda.synchronize()

cpu_elapsed_with_sync = time.perf_counter() - cpu_start

# ---------------------------------------------------------
# Method 3: CUDA events
# ---------------------------------------------------------
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()

c = a @ b

end_event.record()

torch.cuda.synchronize()

gpu_elapsed_ms = start_event.elapsed_time(end_event)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print()
print(
    f"CPU timer WITHOUT synchronize: "
    f"{cpu_elapsed_without_sync * 1000:.4f} ms"
)
print(
    f"CPU timer WITH synchronize:    "
    f"{cpu_elapsed_with_sync * 1000:.4f} ms"
)
print(
    f"CUDA event timing:             "
    f"{gpu_elapsed_ms:.4f} ms"
)
