FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# git is required by the current experiment-lineage helper.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git \
       ca-certificates \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata first so Docker can cache dependency installation.
COPY pyproject.toml /workspace/pyproject.toml

# Package discovery currently expects src/.
COPY src /workspace/src

RUN python -m pip install \
      --break-system-packages \
      -e . \
      "dvc[s3]==3.67.1"

# Copy the remaining project metadata/configuration.
COPY configs /workspace/configs
COPY data/raw/timeseries.csv.dvc /workspace/data/raw/timeseries.csv.dvc
COPY .dvc /workspace/.dvc
COPY .dvcignore /workspace/.dvcignore
COPY .git /workspace/.git

CMD ["python", "-c", "import torch; print(torch.__version__)"]

COPY docker/training/entrypoint.sh /usr/local/bin/training-entrypoint

RUN chmod +x /usr/local/bin/training-entrypoint

ENTRYPOINT ["/usr/local/bin/training-entrypoint"]
