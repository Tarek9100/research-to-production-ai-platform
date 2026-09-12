# Training Performance Benchmark

## Baseline Workload

- Model: 2-layer LSTM
- Hidden size: 64
- Parameters: 50,497
- Sequence length: 24
- Training observations: 14,000
- GPU: NVIDIA GeForce GTX 1650
- PyTorch: 2.11.0+cu128

## CPU vs GPU

Batch size 256, 5 epochs:

| Device | Throughput | Total Time |
|---|---:|---:|
| CPU | 9,650 samples/s | 7.241 s |
| GTX 1650 | 41,556 samples/s | 1.682 s |

The GPU provided approximately 4.3x higher training throughput.

## GPU Batch-Size Scaling

| Batch Size | Throughput | Peak GPU Memory | Epoch-5 Loss |
|---:|---:|---:|---:|
| 64 | 22,106.7 samples/s | 47.3 MiB | 0.027353 |
| 256 | 42,704.6 samples/s | 89.7 MiB | 0.029665 |
| 1024 | 50,679.4 samples/s | 256.9 MiB | 0.178353 |
| 2048 | 52,340.3 samples/s | 480.5 MiB | 0.392420 |

## Interpretation

Larger batches increased GPU throughput and memory usage.

Throughput gains began to flatten after batch size 1024. Increasing from 1024 to 2048 improved throughput by only a few percent while substantially increasing memory consumption.

Model convergence also changed. Larger batches resulted in fewer optimizer updates per epoch, so comparing a fixed number of epochs does not imply equal optimization work.

For approximately 14,000 training samples:

- batch 64: roughly 218 optimizer steps per epoch
- batch 256: roughly 55 steps per epoch
- batch 1024: roughly 14 steps per epoch
- batch 2048: roughly 7 steps per epoch

Therefore infrastructure throughput and model convergence must be evaluated together.

## Engineering Lesson

The configuration with the highest samples-per-second is not automatically the best training configuration.

A production MLOps platform should track both:

- systems metrics: runtime, throughput, GPU memory, utilization
- ML metrics: loss, validation metrics, convergence behavior

Batch size is simultaneously an ML hyperparameter and an infrastructure capacity parameter.

## FP32 vs Automatic Mixed Precision

Batch size 1024, 10 epochs:

| Precision | Throughput | Total Time | Peak GPU Memory | Epoch-10 Loss |
|---|---:|---:|---:|---:|
| FP32 | 49,514.5 samples/s | 2.823 s | 256.9 MiB | 0.033208 |
| AMP/FP16 | 31,950.2 samples/s | 4.374 s | 146.9 MiB | 0.033220 |

AMP reduced peak GPU memory by approximately 43%.

However, AMP reduced training throughput on this workload. The model is small and the additional autocast and gradient-scaling overhead outweighed the benefit of reduced-precision computation on this hardware/workload combination.

The nearly identical final losses indicate that AMP preserved training behavior in this experiment.

### Engineering Lesson

Mixed precision should be benchmarked rather than assumed to improve performance.

Its value may come from:

- reduced GPU memory consumption,
- increased throughput,
- enabling larger batches,
- enabling models that otherwise would not fit.

For this laptop workload, memory reduction was the clear benefit while FP32 delivered higher throughput.
