import torch

device = "cuda"

feature_size = 4096

for batch_size in [1024, 4096, 8192, 16384, 32768]:
    try:
        torch.cuda.empty_cache()

        x = torch.randn(
            batch_size,
            feature_size,
            dtype=torch.float32,
            device=device,
        )

        y = x @ torch.randn(
            feature_size,
            feature_size,
            dtype=torch.float32,
            device=device,
        )

        torch.cuda.synchronize()

        print(
            f"batch={batch_size:<6} OK "
            f"allocated="
            f"{torch.cuda.memory_allocated()/1024**2:.1f} MiB"
        )

        del x
        del y

    except torch.OutOfMemoryError:
        print(f"batch={batch_size:<6} OOM")
        break
