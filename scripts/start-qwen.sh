#!/bin/zsh
# Launch Qwen3.6-35B-A3B (Q4_K_S) via llama-server, tuned for Apple M1 Pro / 32 GB.
# Serves an OpenAI-compatible API at http://<HOST>:8080  (web UI at the same URL).
#
# Defaults bind 0.0.0.0 and use a 64K context with a single sequence (-np 1) so a
# sandboxed agent in a container can reach the model via host.docker.internal and
# get the full window per request. 64K fits ~2.4 GB KV alongside the 21.5 GB model
# on 32 GB. Override with env vars if you want loopback-only / a different context:
#   LLAMA_HOST=127.0.0.1 LLAMA_CTX=32768 ./scripts/start-qwen.sh
# Note: 0.0.0.0 exposes :8080 on your LAN — firewall it on untrusted networks.
# (Do NOT swap in the MTP/speculative GGUF for agent use — it breaks the prompt
#  cache and forces a full re-prefill every turn.)
# --cache-ram 2048: cap the server-side prompt cache (default is 8192 MiB!). A
#  single linear agent conversation only needs ~1-2 cached prompts (~250 MiB
#  each); 2 GB is 8x headroom and keeps warm prefills, while reclaiming ~6 GB of
#  host RAM. Don't set 0 — cache-idle-slots needs it, or every turn cold-prefills.

LLM_DIR="$HOME/llm"
BIN="$LLM_DIR/llama-b9466"                                  # extracted llama.cpp release (b9466: includes PR #22929, hybrid/recurrent prompt-cache fix)
MODEL="$LLM_DIR/models/Qwen3.6-35B-A3B-Q4_K_S.gguf"         # ~21.5 GB GGUF

HOST="${LLAMA_HOST:-0.0.0.0}"
CTX="${LLAMA_CTX:-65536}"

# dylibs live next to the binaries in the release tarball
export DYLD_LIBRARY_PATH="$BIN:$DYLD_LIBRARY_PATH"

"$BIN/llama-server" \
  -m "$MODEL" \
  --host "$HOST" --port 8080 \
  -ngl 99 \
  -c "$CTX" \
  -np 1 \
  -fa on \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --cache-ram 2048 \
  -t 6 \
  --jinja \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0
