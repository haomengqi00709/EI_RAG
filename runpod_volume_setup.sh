#!/bin/bash
# ── RunPod Network Volume setup ───────────────────────────────────────────────
# Run this ONCE on a temporary pod that has the network volume mounted.
# After this, delete the temp pod — the volume persists and is reused by
# all serverless workers.
#
# Steps:
#   1. Create a network volume (50GB) in RunPod dashboard
#   2. Create a temporary pod (any GPU) with that volume mounted at /runpod-volume
#   3. SSH into the pod and run this script
#   4. Delete the temporary pod

set -e

VOLUME=/runpod-volume

echo "=== Creating directory structure ==="
mkdir -p $VOLUME/data/embeddings
mkdir -p $VOLUME/data/chunked
mkdir -p $VOLUME/data/eval
mkdir -p $VOLUME/hf-cache

echo ""
echo "=== Installing packages for model download ==="
pip install -q transformers accelerate torch

echo ""
echo "=== Downloading Qwen3-Embedding-4B ==="
python3 - <<'EOF'
from transformers import AutoModel, AutoTokenizer
print("  Downloading tokenizer...")
AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B",
    cache_dir="/runpod-volume/hf-cache")
print("  Downloading model weights (fp16, ~8GB)...")
import torch
AutoModel.from_pretrained("Qwen/Qwen3-Embedding-4B",
    torch_dtype=torch.float16,
    cache_dir="/runpod-volume/hf-cache")
print("  Done.")
EOF

echo ""
echo "=== Downloading Qwen3-Reranker-4B ==="
python3 - <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
print("  Downloading tokenizer...")
AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-4B", padding_side="left",
    cache_dir="/runpod-volume/hf-cache")
print("  Downloading model weights (fp16, ~8GB)...")
import torch
AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-4B",
    torch_dtype=torch.float16,
    cache_dir="/runpod-volume/hf-cache")
print("  Done.")
EOF

echo ""
echo "=== Volume setup complete ==="
echo ""
echo "Next: upload your data files from your local machine:"
echo ""
echo "  rsync -avz --progress data/embeddings/ root@<POD_IP>:/runpod-volume/data/embeddings/"
echo "  rsync -avz --progress data/chunked/    root@<POD_IP>:/runpod-volume/data/chunked/"
echo "  rsync -avz --progress data/eval/       root@<POD_IP>:/runpod-volume/data/eval/"
echo ""
echo "Then create the .env file on the volume:"
echo "  echo 'GOOGLE_API_KEY=your_key' > /runpod-volume/.env"
echo "  echo 'GEMINI_MODEL=gemini-2.0-flash' >> /runpod-volume/.env"
echo ""
echo "Then delete this temp pod — the volume is ready for serverless."
