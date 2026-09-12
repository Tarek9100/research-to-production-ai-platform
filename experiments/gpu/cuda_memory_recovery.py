import gc
import torch

device = "cuda"

def show(label):
    free_b, total_b = torch.cuda.mem_get_info()

    print(f"\n--- {label} ---")
    print(
        f"allocated: "
        f"{torch.cuda.memory_allocated()/1024**2:.1f} MiB"
    )
    print(
        f"reserved:  "
        f"{torch.cuda.memory_reserved()/1024**2:.1f} MiB"
    )
    print(
        f"CUDA free: "
        f"{free_b/1024**2:.1f} MiB"
    )

show("initial")

x = torch.randn(
    600_000_000,
    dtype=torch.float32,
    device=device,
)

torch.cuda.synchronize()

show("after allocation")

del x

show("after del x")

gc.collect()

show("after gc.collect()")

torch.cuda.empty_cache()

show("after empty_cache()")
