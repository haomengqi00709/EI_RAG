#!/bin/bash
# ── Start server locally ──────────────────────────────────────────
cd "$(dirname "$0")"
source venv/bin/activate
python src/server.py
