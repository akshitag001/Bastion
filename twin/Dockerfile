# syntax=docker/dockerfile:1.7
# Bastion Raspberry Pi 5 digital twin: the real aarch64 software stack.
#   On an x86 host this runs under QEMU user emulation (functionally correct, timings NOT Pi 5).
#   On Apple Silicon / Graviton / a real Pi it runs natively.
# Build:  python twin/twinctl.py build      (= docker buildx build --platform linux/arm64 ...)
#
# Pinned versions (verified on PyPI 2026-11-18, see docs/EXECUTORCH_NOTES.md):
#   ExecuTorch 1.6.0 (ships manylinux_2_28 aarch64 wheels), torch 2.15.0.
ARG DEBIAN_TAG=bookworm-slim

# ---------------------------------------------------------------- base: Pi OS-like userland
FROM --platform=linux/arm64 debian:${DEBIAN_TAG} AS base
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore \
    PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=8 VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev ca-certificates procps \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

# ---------------------------------------------------------------- deps: slow layer, cached
FROM base AS deps
ARG EXECUTORCH_VERSION=1.6.0
ARG TORCH_VERSION=2.15.0
# PyPI's aarch64 torch wheel is CPU-only, which is what the Pi uses.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch==${TORCH_VERSION}" "executorch==${EXECUTORCH_VERSION}"
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/requirements.txt

# ---------------------------------------------------------------- app
FROM deps AS app
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps .
COPY tests ./tests
COPY training ./training
COPY benchmarks ./benchmarks
COPY data/samples ./data/samples
COPY twin ./twin
# Models are mounted read-only at runtime (-v ./models:/app/models:ro), never baked into the image.
RUN mkdir -p /app/models /app/benchmarks/results
ENV BASTION_ROOT=/app BASTION_TWIN_PROFILE=pi5-8gb
ENTRYPOINT ["sh", "/app/twin/entrypoint.sh"]
CMD ["bastion", "status"]
