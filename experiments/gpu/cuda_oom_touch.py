import torch

device = torch.device("cuda")
block_mb = 256
blocks = []

props = torch.cuda.get_device_properties(0)

print("GPU:", props.name)
print(f"Reported total VRAM: {props.total_memory / 1024**3:.2f} GiB")
print()

try:
    i = 0

    while True:
        num_elements = (block_mb * 1024 * 1024) // 4

        block = torch.empty(
            num_elements,
            dtype=torch.float32,
            device=device,
        )

        # Force GPU to actually touch/write the allocation.
        block.fill_(1.0)
        torch.cuda.synchronize()

        blocks.append(block)
        i += 1

        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2

        free_bytes, total_bytes = torch.cuda.mem_get_info()

        print(
            f"block={i:02d} "
            f"allocated={allocated:7.1f} MiB "
            f"reserved={reserved:7.1f} MiB "
            f"CUDA_free={free_bytes / 1024**2:7.1f} MiB "
            f"CUDA_total={total_bytes / 1024**2:7.1f} MiB"
        )

except torch.OutOfMemoryError as exc:
    print("\n=== EXPECTED CUDA OOM ===")
    print(exc)

    free_bytes, total_bytes = torch.cuda.mem_get_info()

    print("\n=== FINAL STATE ===")
    print(
        f"allocated={torch.cuda.memory_allocated() / 1024**2:.1f} MiB"
    )
    print(
        f"reserved={torch.cuda.memory_reserved() / 1024**2:.1f} MiB"
    )
    print(
        f"CUDA free={free_bytes / 1024**2:.1f} MiB"
    )
    print(
        f"CUDA total={total_bytes / 1024**2:.1f} MiB"
    )
