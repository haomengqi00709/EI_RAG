# ── EI MAR RAG — RunPod Pod ───────────────────────────────────────────────────
# PyTorch 2.4 base satisfies sentence-transformers>=3.0 (requires torch>=2.4).
# Source code lives at /app/src — NOT under /workspace, so the network
# filesystem mount at /workspace doesn't mask it.
# Data files (chunked, eval, embeddings) are transferred to /workspace after
# pod start via scp (they're too large to bake into the image).

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Python packages
COPY requirements_cuda.txt .
RUN pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir -r requirements_cuda.txt runpod

# Source code at /app/src — survives the /workspace network mount
COPY src/ /app/src/

# Startup script — injects RunPod SSH keys + starts sshd + sleeps
COPY pod_start.sh /pod_start.sh
RUN chmod +x /pod_start.sh && mkdir -p /run/sshd

ENV DATA_ROOT=/workspace
ENV HF_HOME=/workspace/hf-cache
ENV TRANSFORMERS_CACHE=/workspace/hf-cache
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["/pod_start.sh"]
