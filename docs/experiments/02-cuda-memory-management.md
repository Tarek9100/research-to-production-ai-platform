# CUDA Memory Management and OOM Investigation

## Environment

- GPU: NVIDIA GTX 1650
- Physical VRAM: approximately 4 GiB
- Execution environment: WSL2
- PyTorch: 2.11.0+cu128
- Compute Capability: 7.5

## Objective

Understand:

- CUDA OOM behavior
- PyTorch allocated vs reserved memory
- caching allocator behavior
- tensor lifetime
- `empty_cache()`
- FP32 vs FP16 storage
- batch-size memory scaling
- WSL-specific GPU memory behavior

## OOM Experiment

A test repeatedly allocated 256 MiB CUDA tensors.

CUDA reported approximately 4 GiB of total device memory.

The CUDA free-memory counter reached zero after approximately 3.3 GiB of PyTorch allocation, but PyTorch allocator counters subsequently continued growing beyond physical VRAM before the operation eventually failed.

This behavior must not be interpreted as the GTX 1650 physically containing more than 4 GiB of VRAM.

The experiment runs through WSL2/WDDM GPU virtualization, where GPU memory accounting and residency behavior differ from a native Linux datacenter GPU environment.

The OOM message also contained clearly invalid non-PyTorch memory reporting, reinforcing that individual framework metrics must be interpreted in context.

## Memory-Recovery Experiment

Approximately 2.29 GiB was allocated.

Observed:

| State | Allocated | Reserved | CUDA Free |
|---|---:|---:|---:|
| Initial | 0 MiB | 0 MiB | 3294.8 MiB |
| After allocation | 2288.8 MiB | 2290 MiB | 1002.8 MiB |
| After `del x` | 0 MiB | 2290 MiB | 1002.8 MiB |
| After `gc.collect()` | 0 MiB | 2290 MiB | 1002.8 MiB |
| After `empty_cache()` | 0 MiB | 0 MiB | 3292.8 MiB |

### Interpretation

`del x` removed the live tensor allocation.

The PyTorch caching allocator retained the freed block for future reuse, so reserved memory remained high.

`gc.collect()` made no difference because the tensor reference had already been removed.

`torch.cuda.empty_cache()` returned unused cached blocks to CUDA.

`empty_cache()` is therefore not a general solution for live-tensor OOMs.

## Precision Experiment

100 million values were allocated using FP32 and FP16.

Observed:

| dtype | Allocated |
|---|---:|
| FP32 | 382.0 MiB |
| FP16 | 190.7 MiB |

FP16 used approximately half of the storage of FP32, as expected from 4-byte versus 2-byte elements.

## Batch-Size Experiment

A matrix workload was tested with increasing batch sizes.

| Batch Size | Allocated |
|---:|---:|
| 1,024 | 40.1 MiB |
| 4,096 | 136.1 MiB |
| 8,192 | 264.1 MiB |
| 16,384 | 520.1 MiB |
| 32,768 | 1032.1 MiB |

Memory consumption increased approximately linearly with batch size.

## Engineering Lessons

1. GPU OOM is a symptom; the cause must be identified.
2. Physical VRAM, CUDA-driver accounting, and framework allocator accounting are different layers.
3. `memory_allocated()` represents live PyTorch tensor memory.
4. `memory_reserved()` includes PyTorch's reusable CUDA cache.
5. `empty_cache()` does not free live tensors.
6. Lower precision can substantially reduce tensor storage.
7. Batch size directly affects activation/input memory.
8. Peak memory is more important than average memory for workload capacity planning.
9. WSL2 measurements must not be treated as equivalent to bare-metal Linux GPU behavior.

## Production Difference

THIS IS A LAPTOP-SCALE WSL2 EXPERIMENT.

Production AI nodes would typically run native Linux with datacenter GPUs, substantially larger HBM, NUMA-aware CPU/GPU topology, stronger telemetry, ECC, and potentially multiple GPUs connected by NVLink/NVSwitch.

The allocator concepts remain relevant, but WSL-specific memory-accounting behavior should not be extrapolated directly to production GPU nodes.

## Interview Story

I deliberately generated CUDA memory pressure on a 4-GiB GTX 1650 and compared PyTorch allocator metrics with CUDA driver memory information.

WSL2 produced an unexpected case where allocator counters exceeded physical VRAM while CUDA reported no physical memory free. Instead of treating one metric as authoritative, I investigated the different memory-accounting layers and documented the virtualization limitation.

I then demonstrated tensor lifetime and PyTorch caching behavior: deleting a tensor reduced live allocated memory to zero, while reserved memory remained cached until `empty_cache()` returned unused blocks.

I also measured FP16 at approximately half the memory footprint of FP32 and demonstrated roughly linear memory scaling with batch size.
