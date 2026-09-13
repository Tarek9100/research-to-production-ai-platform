# Containerized Training Design

## Goal

Run the existing PyTorch/MLflow training workload inside a reproducible GPU-enabled container while preserving Git, DVC/MinIO, MLflow, and dataset lineage.

## Architecture

The training container will use the real NVIDIA GeForce GTX 1650 exposed through Docker Desktop and WSL2.

The image will use:

- PyTorch 2.11.0
- CUDA 12.8 runtime
- cuDNN 9
- Python 3.12
- project Python dependencies
- DVC with S3 support

External services remain outside the training container:

- MLflow tracking server on the WSL host
- MinIO on the WSL/Docker host

The training container will access those services through host networking reachable from Docker.

## Data flow

Git source/config
    |
    v
Training container
    |
    +---- DVC pull ----> MinIO
    |
    +---- GPU training ----> GTX 1650
    |
    +---- MLflow logs ----> MLflow tracking server

Datasets are never copied permanently into the image.

## Reproducibility

Every run must retain:

- Git commit
- Git dirty state
- YAML training config
- DVC dataset hash
- dataset SHA-256
- container image identity
- PyTorch version
- CUDA runtime
- GPU identity
- MLflow run ID

## Lab vs Production

THIS LAB:
- Docker Desktop
- WSL2 GPU passthrough
- GTX 1650
- single-node MinIO
- local MLflow

PRODUCTION DIFFERENCE:
- managed/container runtime on GPU nodes
- image registry with immutable digests
- production object storage
- centralized MLflow
- secrets management
- scheduler such as Slurm or Kubernetes
- production NVIDIA drivers/toolkit

## Validation order

1. Prove GPU access from an official PyTorch image.
2. Build project training image.
3. Prove project imports.
4. Prove MinIO connectivity.
5. Prove DVC pull.
6. Prove MLflow connectivity.
7. Run full containerized training.
8. Compare metrics with host baseline.
