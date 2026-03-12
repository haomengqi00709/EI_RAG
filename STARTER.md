# Starter Guide

Everything you need to get the server running locally or on RunPod.

---

## Running Locally

```bash
./start_local.sh
```

Opens at `http://localhost:5000`.

---

## Running on RunPod (Fresh Pod)

### Step 1 — Create a new pod on RunPod
- Go to [runpod.io](https://runpod.io) → deploy a pod
- Attach your **network volume** (so files and packages persist)
- Under **Expose HTTP Ports**, add `5000`
- Once running, grab the SSH details from the **Connect** tab:
  ```
  ssh root@{IP} -p {PORT} -i ~/.ssh/id_ed25519
  ```

### Step 2 — Upload files (first time only, or after code changes)

Run these from your **local Mac terminal**. Replace `PORT` and `IP` with your pod's values.

**Source code:**
```bash
scp -i ~/.ssh/id_ed25519 -P PORT "/Users/jasonhao/Desktop/Trust AI Advisory/RAG_experiment2/src/"*.py "/Users/jasonhao/Desktop/Trust AI Advisory/RAG_experiment2/src/index.html" root@IP:/workspace/RAG_experiment2/src/
```

**Embeddings** (large, first time only):
```bash
scp -i ~/.ssh/id_ed25519 -P PORT "/Users/jasonhao/Desktop/Trust AI Advisory/RAG_experiment2/data/embeddings/vectors.npy" "/Users/jasonhao/Desktop/Trust AI Advisory/RAG_experiment2/data/embeddings/manifest.jsonl" root@IP:/workspace/RAG_experiment2/data/embeddings/
```

**Chunked files** (first time only):
```bash
scp -i ~/.ssh/id_ed25519 -P PORT "/Users/jasonhao/Desktop/Trust AI Advisory/RAG_experiment2/data/chunked/"*.jsonl root@IP:/workspace/RAG_experiment2/data/chunked/
```

**PDFs** (first time only):
```bash
scp -i ~/.ssh/id_ed25519 -P PORT "/Users/jasonhao/Desktop/Trust AI Advisory/RAG_experiment2/Documents/"*.pdf root@IP:/workspace/RAG_experiment2/Documents/
```

**Startup script** (first time only):
```bash
scp -i ~/.ssh/id_ed25519 -P PORT "/Users/jasonhao/Desktop/Trust AI Advisory/RAG_experiment2/start_pod.sh" root@IP:/workspace/RAG_experiment2/
```

> **Note:** Embeddings, chunked files, PDFs, and the startup script only need to be uploaded once — they live on the volume and persist across pod restarts.
> Only re-upload `src/` files when you make code changes.

### Step 3 — SSH into the pod

**Local Mac terminal:**
```bash
ssh root@IP -p PORT -i ~/.ssh/id_ed25519
```

### Step 4 — Start the server

**Pod terminal:**
```bash
bash /workspace/RAG_experiment2/start_pod.sh
```

- First run: creates venv, installs packages (~3–5 min), downloads embedding model (~2.3 GB)
- Subsequent runs on same volume: starts in ~30 seconds
- Will prompt for `GOOGLE_API_KEY` — enter it when asked (never stored to disk)

### Step 5 — Get your public URL

RunPod dashboard → your pod → **Connect** tab → click **HTTP Service** next to Port 5000:
```
https://{pod-id}-5000.proxy.runpod.net
```

This is your shareable link. Works as long as the pod is running.

---

## Restarting After a Pod Restart (Same Volume)

Files and packages are already on the volume. Just:

1. SSH in: `ssh root@NEW_IP -p NEW_PORT -i ~/.ssh/id_ed25519`
2. Run: `bash /workspace/RAG_experiment2/start_pod.sh`

> IP and port change every time you restart a pod — get the new ones from the RunPod Connect tab.

---

## Updating Code

When you make changes locally and want to push to RunPod:

**Local Mac terminal:**
```bash
scp -i ~/.ssh/id_ed25519 -P PORT "/Users/jasonhao/Desktop/Trust AI Advisory/RAG_experiment2/src/"*.py "/Users/jasonhao/Desktop/Trust AI Advisory/RAG_experiment2/src/index.html" root@IP:/workspace/RAG_experiment2/src/
```

Then restart the server on the pod:
```bash
pkill -f server.py && bash /workspace/RAG_experiment2/start_pod.sh
```

---

## Directory Structure on RunPod

```
/workspace/
├── venv/                          ← Python packages (persists on volume)
└── RAG_experiment2/
    ├── src/                       ← All Python source + index.html
    ├── data/
    │   ├── embeddings/            ← vectors.npy + manifest.jsonl
    │   └── chunked/               ← *.jsonl chunk files
    ├── Documents/                 ← PDF source files (for page viewer)
    └── start_pod.sh               ← Startup script
```

---

## Common Errors & Fixes

| Error | Fix |
|---|---|
| `GOOGLE_API_KEY not found` | Run `export GOOGLE_API_KEY="your-key"` before starting |
| `torchvision::nms does not exist` | `pip uninstall torchvision -y` |
| `accelerate` not found | `pip install accelerate -q` |
| `flask` blinker conflict | `pip install flask --ignore-installed blinker -q` |
| Port 5000 shows "Initializing" | Server hasn't started yet — run `start_pod.sh` first |
| PDF viewer not working | PDFs not uploaded — run the PDFs scp command in Step 2 |
| New pod, packages missing | Volume venv missing — `start_pod.sh` will recreate it automatically |

---

## API Keys

- `GOOGLE_API_KEY` — required for Gemini (answer generation, enrichment, evaluation)
- Never store API keys in files on the pod — always enter them when prompted

---

## Key Files

| File | Purpose |
|---|---|
| `src/server.py` | Flask backend — retrieval + generation pipeline |
| `src/index.html` | Browser UI |
| `src/retrieve.py` | Hybrid retrieval (dense + BM25 + RRF), Stage 3 deep search |
| `src/generate.py` | Filter → expand → reduce generation pipeline |
| `src/embed.py` | Embedding model loader |
| `start_local.sh` | One-command local startup |
| `start_pod.sh` | One-command RunPod startup |
| `TRACKING.md` | Eval results and pipeline improvement history |
| `PROJECT_OVERVIEW.md` | Full technical overview (presentation-ready) |
