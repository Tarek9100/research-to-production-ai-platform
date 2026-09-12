import torch

device = "cuda"
feature_size = 4096

weight = torch.randn(
    feature_size,
    feature_size,
    dtype=torch.float32,
    device=device,
)

batch_sizes = [
    1024,
    4096,
    8192,
    16384,
    32768,
]

print("GPU:", torch.cuda.get_device_name(0))
print()
print(
    f"{'batch':>8} "
    f"{'current_MiB':>14} "
    f"{'peak_MiB':>12}"
)

for batch_size in batch_sizes:

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(
        batch_size,
        feature_size,
        dtype=torch.float32,
        device=device,
    )

    y = x @ weight

    torch.cuda.synchronize()

    current = (
        torch.cuda.memory_allocated()
        / 1024**2
    )

    peak = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    print(
        f"{batch_size:8d} "
        f"{current:14.1f} "
        f"{peak:12.1f}"
    )

    del x
    del y

torch.cuda.empty_cache()
