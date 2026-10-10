#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/serve-local.sh /path/to/llama-server /path/to/model.gguf" >&2
  exit 2
fi
if [ ! -x "$1" ] || [ ! -f "$2" ]; then
  echo "An executable llama-server and an existing GGUF model file are required." >&2
  exit 2
fi

# CPU-only, one generation slot, loopback binding. No downloads or cloud calls.
exec "$1" --model "$2" --alias deepscholar-qwen-local \
  --host 127.0.0.1 --port 8080 --ctx-size 4096 \
  --threads 4 --parallel 1 --n-gpu-layers 0 --no-webui \
  --cors-origins http://127.0.0.1:8000
