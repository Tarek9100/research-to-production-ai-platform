import torch

device = "cuda"
elements = 100_000_000

for dtype in [torch.float32, torch.float16]:
    torch.cuda.empty_cache()

    x = torch.empty(
        elements,
        dtype=dtype,
        device=device,
    )

    torch.cuda.synchronize()

    print(
        f"{dtype}: "
        f"{torch.cuda.memory_allocated()/1024**2:.1f} MiB"
    )

    del x
    torch.cuda.empty_cache()
