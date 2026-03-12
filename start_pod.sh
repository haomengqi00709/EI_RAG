#!/bin/bash
# ── Start server on RunPod ────────────────────────────────────────
cd /workspace/RAG_experiment2

# ── API key ──────────────────────────────────────────────────────
if [ -z "$GOOGLE_API_KEY" ]; then
  read -rsp "Enter GOOGLE_API_KEY: " GOOGLE_API_KEY
  echo
  export GOOGLE_API_KEY
fi

# ── Virtual environment ──────────────────────────────────────────
VENV=/workspace/venv
if [ ! -d "$VENV" ]; then
  echo "Creating venv on volume..."
  python -m venv "$VENV"
fi
source "$VENV/bin/activate"

# ── Install packages if missing ───────────────────────────────────
python -c "import flask" 2>/dev/null || {
  echo "Installing packages..."
  pip install flask google-generativeai python-dotenv rank-bm25 pymupdf transformers torch accelerate sentence-transformers --ignore-installed blinker -q
  pip uninstall torchvision -y 2>/dev/null || true
}

# ── Start server ─────────────────────────────────────────────────
echo "Starting server..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/server.py
