#!/bin/bash
# ── RunPod Pod Startup Script ─────────────────────────────────────────────────
# Injects SSH public keys from RunPod's environment, starts SSH, keeps alive.

set -e

# RunPod injects your account's SSH public keys as $PUBLIC_KEY
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    echo "[startup] SSH public key injected from RunPod environment."
else
    echo "[startup] WARNING: PUBLIC_KEY env var not set — SSH key login won't work."
    echo "          Add your public key in RunPod Settings → SSH Public Keys."
fi

# Start SSH daemon
service ssh start
echo "[startup] SSH daemon started."

# Keep container alive for interactive pod use
echo "[startup] Pod ready. Connect via: ssh root@<IP> -p <PORT>"
sleep infinity
