#!/usr/bin/env bash
set -euo pipefail

cd /workspace

echo "===== TRAINING CONTAINER ====="

# The runtime image intentionally does not contain .git.
# Git lineage is injected as immutable build metadata instead.
# DVC therefore operates in supported no-SCM mode inside the container.
dvc config --local core.no_scm true

if [[ -n "${MINIO_ENDPOINT:-}" ]]; then
    echo "Configuring DVC MinIO endpoint: ${MINIO_ENDPOINT}"

    dvc remote modify --local \
        minio endpointurl \
        "${MINIO_ENDPOINT}"
fi

if [[ -n "${MINIO_ACCESS_KEY:-}" ]]; then
    dvc remote modify --local \
        minio access_key_id \
        "${MINIO_ACCESS_KEY}"
fi

if [[ -n "${MINIO_SECRET_KEY:-}" ]]; then
    dvc remote modify --local \
        minio secret_access_key \
        "${MINIO_SECRET_KEY}"
fi

echo
echo "===== DVC PULL ====="
dvc pull

echo
echo "===== DVC STATUS ====="
dvc status

echo
echo "===== DATASET SHA-256 ====="
sha256sum data/raw/timeseries.csv

echo
echo "===== WORKLOAD ====="
exec "$@"
