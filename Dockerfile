# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /app

ARG ENABLE_CUDA=0
ARG TORCH_CUDA_WHL_INDEX=https://download.pytorch.org/whl/cu124

RUN python -m venv "${VENV_PATH}" && \
    pip install --upgrade pip setuptools wheel

# Kept in builder stage only, to reduce runtime image size if any dependency needs compilation.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN if [ "${ENABLE_CUDA}" = "1" ]; then \
      PIP_INDEX_URL="${TORCH_CUDA_WHL_INDEX}" PIP_EXTRA_INDEX_URL="https://pypi.org/simple" pip install -e .; \
    else \
      pip install -e .; \
    fi


FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv \
    PYTHONPATH=/app/src \
    PIPELINE_CONFIG=/app/src/real_time_visual_defect_detection/config/default.yaml
ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /app

# Common runtime libraries required by opencv/scikit-learn wheels.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY . .

# Keep backward compatibility with existing config defaults that write/read under ./data
# while still exposing an explicit container volume at /data.
RUN rm -rf /app/data && \
    mkdir -p /data /config && \
    ln -s /data /app/data

VOLUME ["/data", "/config"]

# Additional CLI args can be passed after the image name, e.g.:
# docker run ... image --dataset-path /data/raw/sample.zip --extract-dir /data/raw/extracted
ENTRYPOINT ["sh", "-c", "exec python scripts/main.py --config \"$PIPELINE_CONFIG\" \"$@\"", "--"]
