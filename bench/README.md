# bench/

Measurement and comparison harnesses. All drive `llama-server` over HTTP and read
each response's `timings` object (no log scraping). Start a server first (see the
[root README](../README.md)); JSON outputs land in [`../results/`](../results).

| Script | Purpose |
|---|---|
| `bench.py` | Throughput vs context depth, cold/warm time-to-first-token, thinking-vs-no-think, tool-calling reliability. `python3 bench.py --out ../results/bench-$(date +%Y%m%d).md` (`--quick` for a smoke test). |
| `tool-test.py` | Tool-calling demo: gives the model three real tools and runs the agentic loop, printing each call. `python3 tool-test.py "what time is it and what is 12*34?"` |
| `run_eval.py` | The **dense-vs-MoE bake-off**. Replays a fixed, deterministic 12-turn *growing* conversation with a byte-stable prefix and records, per turn: prefilled tokens (cache reuse), prefill/decode tok/s, and quality outputs. Run once per model. |
| `compare.py` | Side-by-side report from two `run_eval.py` result files. |
| `verify_prefix_fix.py` | Demonstrates that changing only a *trailing* message re-prefills just that suffix (≈40 tokens) while the cached `[system + history]` prefix is reused — the mechanism behind cross-turn cache reuse on the hybrid models. |

## Reproducing the dense-vs-MoE comparison

```bash
# 1. serve model A (e.g. the dense 27B) on :8080, then:
python3 run_eval.py --base-url http://127.0.0.1:8080 --label qwen27b-dense \
  --server-log ~/llm/server.log
# 2. stop it, serve model B (e.g. the MoE 35B-A3B) on :8080, then:
python3 run_eval.py --base-url http://127.0.0.1:8080 --label qwen35b-a3b \
  --server-log ~/llm/server.log
# 3. side-by-side:
python3 compare.py qwen27b-dense qwen35b-a3b
```

The two models cannot be co-resident on a 32 GB host, so run them sequentially.
Results from a representative run: [`../results/comparison-dense-27b-vs-moe-35b-a3b-20260602.md`](../results/comparison-dense-27b-vs-moe-35b-a3b-20260602.md).
