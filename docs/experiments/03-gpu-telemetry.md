# GPU Telemetry Under Sustained Compute Load

## Environment

- GPU: NVIDIA GeForce GTX 1650
- VRAM: 4 GiB
- Runtime: WSL2
- PyTorch: 2.11.0+cu128
- Workload: repeated FP32 4096 × 4096 matrix multiplication

## Objective

Observe the relationship between:

- GPU workload
- SM utilization
- memory utilization
- power
- clocks
- temperature
- framebuffer memory
- idle-to-load and load-to-idle transitions

## Controlled Workload

The workload was modified to synchronize after each matrix multiplication so that a 30-second host-side duration corresponded closely to 30 seconds of actual GPU activity.

Observed:

- Iterations: 355
- Elapsed: 30.06 seconds
- Throughput: 11.81 matrix multiplications/second

## Telemetry During Load

Representative values:

- SM utilization: 99–100%
- Memory utilization: 25–26%
- Power: 48–49 W
- Temperature: approximately 68–70 C
- Memory clock: 6000 MHz
- GPU clock: approximately 1500–1530 MHz
- Framebuffer memory: approximately 415 MB

## Idle State

After the workload stopped:

- SM utilization: 0%
- Power: approximately 3–4 W
- GPU clock: approximately 300 MHz
- Memory clock: approximately 405 MHz
- Framebuffer memory: approximately 13 MB

Temperature declined gradually after compute stopped.

## Interpretation

The workload saturated GPU compute resources while using only a fraction of the reported memory-subsystem activity.

This is consistent with a compute-intensive dense matrix multiplication workload.

GPU frequency and power increased substantially under load, demonstrating dynamic voltage/frequency behavior.

Temperature changed more slowly than utilization and power because thermal response has physical inertia.

## Important Failure Discovered

The original stress-test implementation used a host-side timer around asynchronous CUDA kernel submission.

That caused the CPU to enqueue more than 30 seconds of GPU work before calling `torch.cuda.synchronize()`.

As a result, the GPU remained at 100% utilization well after the host-side loop had stopped submitting work.

The load generator was corrected by synchronizing each iteration before evaluating elapsed wall-clock time.

This demonstrated that CUDA kernel launches are asynchronous relative to the CPU and that host-side timers can produce incorrect workload-duration assumptions.

## Operational Lesson

GPU utilization alone does not prove optimal performance.

A GPU may report 100% utilization while operating at:

- reduced clocks
- a power limit
- a thermal limit
- different precision
- different kernel efficiency

Performance troubleshooting therefore requires correlated telemetry such as:

- utilization
- clocks
- power
- temperature
- VRAM
- throttling reasons
- error counters
- application throughput

## Production Difference

Production GPU clusters typically expose these metrics through NVIDIA DCGM / DCGM Exporter into Prometheus and Grafana rather than relying on manual `nvidia-smi` sessions.

Metrics should also be correlated with scheduler context such as:

- node
- GPU
- Slurm job ID
- user
- training run
- model
