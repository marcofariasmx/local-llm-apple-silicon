#!/bin/zsh
# Launch Qwen3.6-27B (dense, Q4_K_S) via llama-server, tuned for Apple M1 Pro / 32 GB.
# Serves an OpenAI-compatible API at http://<HOST>:8080  (web UI at the same URL).
#
# Was the 35B-A3B MoE; switched to the dense 27B after a controlled bake-off
# (~/llm/sandbox-27b/): both reuse llama.cpp's hybrid checkpoint cache equally well
# on a STABLE prefix (~100% restore — the agent now keeps its prompt prefix byte-
# stable), but the 27B is ~15.9 GB vs the 35B's ~21.5 GB → ~5 GB less wired memory,
# which ends the host swap. It's ~3-5x slower per token (dense: 27B active vs the
# A3B's 3B) — an accepted trade for the RAM headroom + better reasoning quality.
#
# Defaults bind 0.0.0.0 and use a 64K context with a single sequence (-np 1) so a
# sandboxed agent in a container can reach the model via host.docker.internal and
# get the full window per request. Override with env vars for loopback-only / other ctx:
#   LLAMA_HOST=127.0.0.1 LLAMA_CTX=32768 ./scripts/start-qwen.sh
# Note: 0.0.0.0 exposes :8080 on your LAN — firewall it on untrusted networks.
# (Do NOT swap in the MTP/speculative GGUF for agent use — it breaks the prompt
#  cache and forces a full re-prefill every turn.)
# --cache-ram 2048: cap the server-side prompt cache (default is 8192 MiB!). It holds
#  the per-turn checkpoints that make warm prefills possible; 2 GB is ample for a
#  single linear conversation while reclaiming ~6 GB of host RAM. Don't set 0 —
#  cache-idle-slots needs it, or every turn cold-prefills.

LLM_DIR="$HOME/llm"
BIN="$LLM_DIR/llama-b9466"                                  # extracted llama.cpp release (b9466: includes PR #22929, hybrid/recurrent prompt-cache fix)
MODEL="$LLM_DIR/models/Qwen3.6-27B-Q4_K_S.gguf"            # ~15.9 GB GGUF (dense 27B)

HOST="${LLAMA_HOST:-0.0.0.0}"
# 32K: the agent keeps its prompt prefix byte-stable, so llama.cpp's checkpoint cache
# reuses across turns (only NEW tokens are prefilled most turns) — the window is usable
# instead of being re-prefilled every turn. Chosen to MAXIMISE RAM headroom (the whole
# point of moving to the smaller 27B): 32K is the most economical window that still
# comfortably holds the ~16K compaction target + 4K response reserve + system/tools,
# with margin. (64K CRASHES the dense 27B here — its worst-case compute buffer exceeds
# the ~25 GB Metal budget; 48K starts at 18.6 GB. The dense 27B's per-pass compute
# buffer, unlike the sparse 35B-A3B's, scales with context, so a big window is costly.)
# A smaller window also caps the worst-case cold prefill on this slower model. Override
# with LLAMA_CTX (must stay above SCA_COMPACT_TARGET_TOKENS + response_reserve).
CTX="${LLAMA_CTX:-32768}"

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
