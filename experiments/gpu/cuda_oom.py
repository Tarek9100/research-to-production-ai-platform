import torch

device = torch.device("cuda")

print("GPU:", torch.cuda.get_device_name(0))
print(
    "Total VRAM:",
    torch.cuda.get_device_properties(0).total_memory / 1024**3,
    "GiB",
)

blocks = []
block_mb = 256

try:
    i = 0

    while True:
        # 256 MiB FP32 allocation:
        # number of float32 values = bytes / 4
        num_elements = (block_mb * 1024 * 1024) // 4

        blocks.append(
            torch.empty(
                num_elements,
                dtype=torch.float32,
                device=device,
            )
        )

        i += 1

        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2

        print(
            f"block={i:02d} "
            f"allocated={allocated:.1f} MiB "
            f"reserved={reserved:.1f} MiB"
        )

except torch.OutOfMemoryError as exc:
    print("\n=== EXPECTED CUDA OOM ===")
    print(exc)

    print("\n=== MEMORY AT FAILURE ===")
    print(
        "allocated:",
        torch.cuda.memory_allocated() / 1024**2,
        "MiB",
    )
    print(
        "reserved:",
        torch.cuda.memory_reserved() / 1024**2,
        "MiB",
    )

    print("\n=== PYTORCH MEMORY SUMMARY ===")
    print(torch.cuda.memory_summary())
