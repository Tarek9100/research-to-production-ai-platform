# Containerized Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Containerize the existing GPU PyTorch training workload without changing its model behavior or data/versioning architecture.

**Architecture:** Use the official PyTorch 2.11.0 CUDA 12.8 runtime image as the base. Keep MinIO and MLflow external and let the training container retrieve its DVC-managed dataset and report experiments over the network.

**Tech Stack:** Docker Desktop, WSL2, NVIDIA GPU-PV, PyTorch 2.11.0, CUDA 12.8, DVC S3, MinIO, MLflow 3.x.

**Spec:** `docs/superpowers/specs/2026-09-13-containerized-training-design.md`

## Global Constraints

- Preserve PyTorch 2.11.0 / CUDA 12.8 behavior.
- Use the real GTX 1650 for the Docker GPU test.
- Do not bake datasets or secrets into container images.
- DVC + MinIO remains the dataset versioning workflow.
- SHA-256 remains an additional dataset integrity fingerprint.
- Keep MinIO and MLflow external during this phase.
- Validate each integration independently before full training.
- Slurm GPU simulation is not part of this phase.

---

### Task 1: Validate Docker GPU Runtime

- [ ] Pull/run the official PyTorch CUDA runtime image.
- [ ] Verify PyTorch version.
- [ ] Verify CUDA runtime.
- [ ] Verify `torch.cuda.is_available()`.
- [ ] Verify GTX 1650 identity.

### Task 2: Build Training Image

**Files:**
- Create: `docker/training/Dockerfile`
- Create: `docker/training/entrypoint.sh`
- Create: `.dockerignore`

- [ ] Add project source and dependencies.
- [ ] Install DVC S3 dependencies.
- [ ] Verify imports.
- [ ] Tag image with Git SHA.

### Task 3: Validate External Service Networking

- [ ] Resolve host from container.
- [ ] Reach MinIO health endpoint.
- [ ] Reach MLflow health/API endpoint.
- [ ] Keep credentials outside the image.

### Task 4: Validate DVC Dataset Retrieval

- [ ] Start container without dataset workspace copy.
- [ ] Configure MinIO credentials at runtime.
- [ ] Run `dvc pull`.
- [ ] Verify DVC hash.
- [ ] Verify SHA-256.

### Task 5: Run Containerized Training

- [ ] Start with `--gpus all`.
- [ ] Execute baseline YAML config.
- [ ] Verify MLflow run creation.
- [ ] Verify Git/DVC/SHA metadata.
- [ ] Compare model metrics to host baseline.

### Task 6: Record Container Lineage

- [ ] Record image tag.
- [ ] Record image digest/ID.
- [ ] Log runtime image identity into MLflow.
- [ ] Document lab versus production differences.
- [ ] Commit containerization files.
