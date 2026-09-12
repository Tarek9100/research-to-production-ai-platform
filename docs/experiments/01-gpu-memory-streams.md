# GPU Fundamentals Lab — Memory Transfers, CUDA Timing, and Streams

## Environment

- Host OS: Windows with WSL2
- Guest environment: Ubuntu 24.04 LTS
- GPU: NVIDIA GeForce GTX 1650
- VRAM: 4 GiB
- Compute Capability: 7.5 (`sm_75`)
- NVIDIA Driver: 596.21
- PyTorch: 2.11.0+cu128
- PyTorch CUDA Runtime: 12.8

## Objective

Understand and measure:

- GPU memory allocation
- pageable vs pinned host memory
- host-to-device transfer throughput
- asynchronous CUDA execution
- synchronization
- CUDA event timing
- CUDA streams

## Experiment 1 — GPU Memory Allocation

Two FP32 matrices of shape 4096 × 4096 were allocated on the GPU.

Observed:

- After A and B:
  - allocated: 128 MiB
  - reserved: 128 MiB
- After C = A @ B:
  - allocated: 200.125 MiB
  - reserved: 212 MiB

This demonstrated the distinction between live PyTorch tensor allocations and memory reserved by the PyTorch CUDA caching allocator.

## Experiment 2 — Sustained Matrix Multiplication

100 matrix multiplications of 4096 × 4096 FP32 matrices completed in:

- 8.40 seconds

This generated sustained GPU compute activity on the GTX 1650.

## Experiment 3 — Pageable vs Pinned Host Memory

A ~256 MiB FP32 tensor was transferred from host RAM to GPU VRAM ten times.

Observed:

| Memory Type | Time | Effective Throughput |
|---|---:|---:|
| Pageable | 0.4061 s | 6.61 GB/s |
| Pinned | 0.2069 s | 12.97 GB/s |

Pinned memory achieved approximately 1.96× the effective transfer throughput in this lab.

This result must not be interpreted as a universal 1.96× improvement. It reflects this specific laptop, WSL2 implementation, workload size, and software stack.

## Experiment 4 — CUDA Asynchronous Execution

A 4096 × 4096 matrix multiplication was timed three ways.

Observed:

| Timing Method | Result |
|---|---:|
| CPU timer without CUDA synchronization | 0.9778 ms |
| CPU timer with synchronization | 83.9055 ms |
| CUDA events | 83.1215 ms |

The unsynchronized CPU timer measured mostly kernel submission latency rather than actual GPU execution time.

CUDA events and synchronized wall-clock timing produced similar values and therefore represented actual execution much more accurately.

## Experiment 5 — CUDA Streams

Twenty pairs of 3072 × 3072 matrix multiplications were run using one stream and two streams.

Observed:

| Execution | Time |
|---|---:|
| Single CUDA stream | 1231.42 ms |
| Two CUDA streams | 1243.95 ms |

Two streams did not improve performance.

Interpretation:

The workload likely already consumes most of the available execution resources or memory bandwidth of the GTX 1650. Multiple CUDA streams expose concurrency opportunities but cannot create additional physical compute capacity.

## Key Engineering Lessons

1. CUDA execution is asynchronous relative to the CPU.
2. Naive Python wall-clock measurements can measure enqueue latency instead of execution latency.
3. CUDA events are appropriate for GPU-side timing.
4. Pinned host memory can substantially improve H2D transfer efficiency.
5. `non_blocking=True` is most useful when combined with pinned source memory.
6. Multiple streams permit concurrency but do not guarantee speedup.
7. GPU performance depends on the full data pipeline, not only arithmetic throughput.

## Lab Limitation

THIS IS A LAPTOP-SCALE EXPERIMENT.

The GTX 1650 has:

- one GPU
- 4 GiB VRAM
- no NVLink
- no NVSwitch
- no MIG
- no datacenter ECC capabilities
- WSL2-mediated GPU access

A production AI node may instead contain eight H100/B200-class GPUs connected through NVLink/NVSwitch, NUMA-aware CPU topology, high-speed NICs/HCAs, RDMA, and substantially larger HBM capacity and bandwidth.

The CUDA programming concepts remain valid, but the topology, bottlenecks, failure modes, and scale are different.

## Interview Story

I built a GPU experimentation environment on a GTX 1650 through WSL2 to study CUDA behavior directly.

I measured pinned vs pageable H2D transfer throughput and observed approximately 12.97 GB/s vs 6.61 GB/s in my environment. I also demonstrated CUDA's asynchronous execution behavior: naive CPU timing measured about 0.98 ms while synchronized and CUDA-event measurements showed the actual operation required about 83 ms.

I then tested multiple CUDA streams and found that two streams were slightly slower than one for large GEMM operations. That reinforced that concurrency only helps when the hardware has idle execution resources or independent engines available; multiple streams do not automatically increase throughput.
