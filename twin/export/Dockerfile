# x86-64 export environment: Llama 3.2 1B Instruct -> ExecuTorch .pte (XNNPACK, 8da4w).
# Export is architecture-independent: the .pte runs on the arm64 twin / Raspberry Pi 5.
# Use the SAME ExecuTorch version as the twin runtime (twin/Dockerfile).
FROM --platform=linux/amd64 python:3.11-slim-bookworm
ARG EXECUTORCH_VERSION=1.6.0
ARG TORCH_VERSION=2.15.0
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=8 \
    PYTHONUNBUFFERED=1 HF_HOME=/work/models/llama/.hf-cache
# CPU-only torch (PyPI's default x86 Linux wheel pulls several GB of CUDA libraries we don't need).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch==${TORCH_VERSION}+cpu" --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
 && pip install "executorch==${EXECUTORCH_VERSION}" "huggingface_hub[cli]"
WORKDIR /work
COPY twin/export/export.sh /usr/local/bin/bastion-export
ENTRYPOINT ["sh", "/usr/local/bin/bastion-export"]
