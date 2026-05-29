#!/bin/zsh
# Launch Qwen3.6-35B-A3B (Q4_K_S) via llama-server, tuned for Apple M1 Pro / 32 GB.
# Serves an OpenAI-compatible API at http://127.0.0.1:8080  (web UI at the same URL).
#
# Adjust LLM_DIR / MODEL to wherever you extracted llama.cpp and downloaded the GGUF.

LLM_DIR="$HOME/llm"
BIN="$LLM_DIR/llama-b9384"                                  # extracted llama.cpp release
MODEL="$LLM_DIR/models/Qwen3.6-35B-A3B-Q4_K_S.gguf"         # ~21.5 GB GGUF

# dylibs live next to the binaries in the release tarball
export DYLD_LIBRARY_PATH="$BIN:$DYLD_LIBRARY_PATH"

"$BIN/llama-server" \
  -m "$MODEL" \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 \
  -c 32768 \
  -fa on \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  -t 6 \
  --jinja \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0
