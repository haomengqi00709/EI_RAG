#!/bin/bash
# ── RunPod setup script for EI MAR RAG pipeline ───────────────────────────────
# Run once after uploading code and data to the pod.
# Assumes: RunPod PyTorch image (torch + CUDA already installed)

set -e
cd /workspace/RAG_experiment2

echo "=== Installing Python packages ==="
pip install -r requirements_cuda.txt

echo ""
echo "=== Pre-downloading Qwen models from HuggingFace ==="
python3 - <<'EOF'
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

print("Downloading Qwen3-Embedding-4B ...")
AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B")
AutoModel.from_pretrained("Qwen/Qwen3-Embedding-4B", torch_dtype=torch.float16)
print("  done.")

print("Downloading Qwen3-Reranker-4B ...")
AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-4B", padding_side="left")
AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-4B", torch_dtype=torch.float16)
print("  done.")
EOF

echo ""
echo "=== Verifying GPU ==="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo ""
echo "=== Setup complete. ==="
echo ""
echo "Next steps:"
echo "  1. Create .env with your GOOGLE_API_KEY"
echo "  2. Run eval:  python src/evaluate_e2e.py"
echo "  3. Run server: python src/server.py"
