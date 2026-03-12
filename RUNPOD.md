# RunPod Deployment Guide

## Is this the right long-term approach?

**Short answer:** The pod approach is fine for development and evaluation. For production (serving the API to real users), you'd want a persistent pod or serverless with a network volume.

| Use case | Approach |
|---|---|
| Running evals, re-embedding, experimenting | Pod (current) ✓ |
| Serving the RAG API 24/7 | Persistent pod or serverless |
| Team sharing the same model/data | Network volume (see below) |

**The main pain point** with the current approach is re-transferring data every time the pod is recreated. The fix is a **RunPod Network Volume** — a persistent disk that survives pod restarts. First-time setup transfers the data once; all subsequent pods just mount it and the data is already there. This is the right long-term solution.

---

## Current Architecture

```
Mac (development)                RunPod Pod
─────────────────                ─────────────────────────────────
src/         ──── scp ────────→  /workspace/src/     (code, transferred once)
data/chunked ──── scp ────────→  /workspace/data/    (data, transferred once)
data/eval    ──── scp ────────→
                                 /app/src/            (code baked into image)
                                 /workspace/hf-cache/ (model weights, downloaded once)
```

**Why `/app/src` AND `/workspace/src`?**
- `/app/src` — baked into the Docker image, always available on pod start
- `/workspace/src` — transferred copy, used if you need to test a quick code change without rebuilding Docker
- `PYTHONPATH=/app/src` so scripts use the baked-in version by default

---

## One-time Setup

### 1. Add SSH key to RunPod account
Go to **RunPod Settings → SSH Public Keys** and paste your public key:
```bash
cat ~/.ssh/id_ed25519.pub   # run this on your Mac
```
This is injected automatically into every pod via `$PUBLIC_KEY` env var.

### 2. Build and push Docker image
```bash
cd "/Users/jasonhao/Desktop/Trust AI Advisory/RAG_experiment2"
docker buildx build --platform linux/amd64 -t nejoasfa/ei-rag:latest --push .
```
Rebuild only when you change `src/` code or `requirements_cuda.txt`.

---

## Starting a Pod

### 1. Create pod on RunPod
- Image: `nejoasfa/ei-rag:latest`
- GPU: RTX 3090 / A4000 (24GB+ VRAM for reranker)
- Disk: 50GB+
- Container disk: 20GB+

### 2. SSH in (check Connect button for IP/port)
```bash
ssh root@<IP> -p <PORT>
```
SSH works immediately — the startup script injects your key from RunPod's env.

### 3. Transfer data files (first time or after enrichment updates)
Run on your **Mac**:
```bash
cd "/Users/jasonhao/Desktop/Trust AI Advisory/RAG_experiment2"
scp -P <PORT> -r data/chunked data/eval src root@<IP>:/workspace/
```
Then on the **pod**:
```bash
mkdir -p /workspace/data
mv /workspace/chunked /workspace/data/chunked
mv /workspace/eval /workspace/data/eval
# src/ lands at /workspace/src/ — optional, /app/src is used by default
```

### 4. Run embedding (after data transfer or enrichment updates)
```bash
python /app/src/embed.py
```
Downloads Qwen3-Embedding-4B (~8GB) on first run, cached in `/workspace/hf-cache/`.

---

## Running Evaluations

```bash
# Retrieval quality (recall@1/3/5/10, MRR)
python /app/src/evaluate_retrieval.py

# End-to-end quality (semantic match, abstention rate)
python /app/src/evaluate_e2e.py
```

---

## Retrieving Results Back to Mac

```bash
# From your Mac
scp -P <PORT> -r root@<IP>:/workspace/data/eval data/
scp -P <PORT> -r root@<IP>:/workspace/data/embeddings data/
```

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| Web terminal won't open | Container CMD crashed on startup | Check pod logs; current image uses `sleep infinity` so this shouldn't happen |
| SSH asks for password | Key not injected | Add key to RunPod Settings → SSH Public Keys; pod must be restarted after adding |
| `AutoModel requires PyTorch` | `sentence-transformers` requires torch>=2.4, old base image had 2.2 | Fixed — new base image is `runpod/pytorch:2.4.0` |
| `torchvision::nms does not exist` | torchvision version mismatch after torch upgrade | `pip install --upgrade torchvision --index-url https://download.pytorch.org/whl/cu121` |
| CUDA OOM during reranking | Qwen3-Reranker-4B too large for available VRAM | `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before running |
| `/workspace/data` not found | scp drops directories one level up (chunked/ not data/chunked/) | `mkdir -p /workspace/data && mv /workspace/chunked /workspace/data/chunked` |
| runpodctl "room not ready" | Transfer session expired | Re-run `runpodctl send` and immediately receive; use scp instead |

---

## Rebuilding the Docker Image

Rebuild when:
- You change Python dependencies (`requirements_cuda.txt`)
- You make permanent code changes to `src/`

```bash
docker buildx build --platform linux/amd64 -t nejoasfa/ei-rag:latest --push .
```

Don't rebuild for:
- Data file changes (transfer via scp instead)
- Quick code experiments (edit files directly on pod or scp src/)

---

## Future: Network Volume (recommended for frequent use)

A RunPod Network Volume persists across pod restarts — data is transferred once and always available.

Setup:
1. Create a Network Volume in RunPod (50–100GB)
2. Mount it at `/workspace` when creating a pod
3. Transfer data files once
4. All future pods mount the same volume with data already there

This eliminates the scp step on every new pod.
