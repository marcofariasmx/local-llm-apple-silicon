#!/bin/zsh
# Smoke test: is the local llama-server up and answering?
BASE="${LLAMA_BASE:-http://127.0.0.1:8080}"

echo -n "health: "; curl -s "$BASE/health" || { echo "DOWN — start it with scripts/start-qwen.sh"; exit 1; }
echo
echo "chat test (thinking disabled for a fast, direct answer):"
curl -s "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d '{
  "messages": [{"role":"user","content":"Reply with exactly: hello from local"}],
  "max_tokens": 64,
  "chat_template_kwargs": {"enable_thinking": false}
}' | python3 -c "import sys,json; print(' ->', json.load(sys.stdin)['choices'][0]['message']['content'])"
