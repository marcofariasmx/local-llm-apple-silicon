# local-llm-apple-silicon

**Running large reasoning models fully locally on a MacBook Pro M1 Pro (32 GB) — at usable speed, with reliable tool-calling — and a measured comparison of a dense model against a sparse Mixture-of-Experts model on the same hardware.**

This repository documents an end-to-end setup for serving [Qwen3.6](https://qwen.ai) models locally via [llama.cpp](https://github.com/ggml-org/llama.cpp) on Apple Silicon, with reproducible benchmarks. Two models are covered:

- **Qwen3.6-27B** — dense, ~15.9 GB at `Q4_K_S`. Smaller footprint, higher benchmark quality.
- **Qwen3.6-35B-A3B** — Mixture-of-Experts (35B total, ~3B active), ~20 GB at `Q4_K_S`. Faster per token, larger usable context.

The value of this repository is the *methodology and the measured numbers* — including a head-to-head [dense-vs-MoE comparison](results/comparison-dense-27b-vs-moe-35b-a3b-20260602.md) — not the (uncommittable) weights.

> **Platform:** macOS 26.5 · Apple M1 Pro · 32 GB unified memory · 14-core GPU
> **Stack:** llama.cpp (build `b9466`, Metal) · Qwen3.6-27B / Qwen3.6-35B-A3B `Q4_K_S` · OpenCode

---

## Summary

Both models run locally on a 32 GB M1 Pro with working tool-calling. The choice between them is a trade, not a strict upgrade:

| | Dense **27B** | MoE **35B-A3B** |
|---|---:|---:|
| Memory (wired @ 32K) | **17.9 GB** | 22.4 GB |
| Host swap, agent workload | **~1 GB** | ~7.6 GB |
| Decode throughput | 5.3 tok/s | **25.4 tok/s** |
| Max context (this host) | ≤ 48K (32K rec.) | **64K** |
| Quality (BenchLM aggregate) | **73** | 66 |

- **Choose the dense 27B** when memory headroom or output quality is the priority — it removes swap pressure on a tight 32 GB box and scores higher on reasoning/agentic/coding benchmarks.
- **Choose the MoE 35B-A3B** when token throughput or long context is the priority — it is ~3–5× faster per token and runs to 64K.

The common bottleneck on this hardware is **prompt processing (prefill)**, not generation. With a stable prompt prefix, llama.cpp's hybrid prompt-cache reuses prior context across turns, so the cold-start cost is largely one-time per session. Full numbers in [Benchmarks](#benchmarks).

---

## Hardware

| | |
|---|---|
| Machine | MacBook Pro 14" (MacBookPro18,3) |
| Chip | Apple M1 Pro (8-core CPU: 6P+2E, 14-core GPU) |
| Memory | 32 GB unified |
| OS | macOS 26.5 |

The key enabler is **unified memory**: the GPU addresses the same 32 GB as the CPU, so a 16–22 GB model fits where a discrete GPU with 8–12 GB VRAM could not.

---

## Model selection

Both models are Qwen3.6-generation reasoning models (hidden chain-of-thought emitted to a separate `reasoning_content` field), share the same chat template and sampler defaults, and quantize to `Q4_K_S` to fit 32 GB with KV-cache headroom.

**Dense vs MoE — the architectural trade:**

- **MoE (35B-A3B): pay RAM for a big model, pay compute for a small one.** Only ~3B of the 35B parameters activate per token, so it generates at the speed of a small model while retaining a large model's knowledge — but *all* experts must be resident, so it needs ~20 GB regardless. Its per-forward-pass compute is small, which lets it run to a 64K context.
- **Dense (27B): every parameter activates per token.** Smaller weights (~15.9 GB) but a larger per-token compute cost, so decode is slower and the per-pass compute buffer grows with context. On this host that caps the practical context at ≤ 48K (see [the Metal ceiling](#key-findings)).

Quantization note: `Q4_K_S` is chosen to fit 32 GB with headroom for KV cache and the OS. `Q4_K_M` is marginally higher quality but tighter; lower quants (`IQ4_XS`) trade quality for context room.

For a full, reproducible head-to-head on identical inputs, see **[results/comparison-dense-27b-vs-moe-35b-a3b-20260602.md](results/comparison-dense-27b-vs-moe-35b-a3b-20260602.md)**.

---

## Why llama.cpp, not Ollama

Ollama is the easy on-ramp, but at the time of writing it had **open bugs in Qwen3 tool-calling** that matter for agentic use:

- Tool definitions passed via the `tools` parameter were rendered as Go-struct strings instead of valid JSON.
- Prior tool calls were stripped from conversation history across turns.
- `thinking + tools` could return empty output.

llama.cpp's `llama-server` with `--jinja` uses the model's own chat template and emits clean, structured `tool_calls`. The [tool-calling reliability benchmark](#benchmarks) confirms this: clean tool episodes, no malformed output. For plain chat, Ollama is fine; for tools, llama.cpp is the safer choice on this stack.

---

## Install

### 1. llama.cpp (prebuilt, Metal-enabled)

Build `b9466` is recommended: it includes the hybrid/recurrent prompt-cache fix ([PR #22929](https://github.com/ggml-org/llama.cpp/pull/22929)) required for cross-turn cache reuse on the Qwen3.6 architecture.

```bash
mkdir -p ~/llm && cd ~/llm
curl -L -o llama.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b9466/llama-b9466-bin-macos-arm64.tar.gz
tar -xzf llama.tar.gz          # -> ~/llm/llama-b9466/  (binaries + dylibs)
xattr -dr com.apple.quarantine ~/llm/llama-b9466   # clear Gatekeeper quarantine
```

> The `.dylib`s live next to the binaries; the launch script sets `DYLD_LIBRARY_PATH` accordingly.

### 2. The model

**Dense Qwen3.6-27B (~15.9 GB, default):**

```bash
mkdir -p ~/llm/models && cd ~/llm/models
curl -L -C - -o Qwen3.6-27B-Q4_K_S.gguf \
  https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-Q4_K_S.gguf
```

**MoE Qwen3.6-35B-A3B (~20 GB, alternative):**

```bash
curl -L -C - -o Qwen3.6-35B-A3B-Q4_K_S.gguf \
  https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen_Qwen3.6-35B-A3B-Q4_K_S.gguf
```

> Use `-C -` so the download resumes if it stalls. Do **not** use the `-MTP` GGUF variants for agentic use — the speculative head breaks the prompt cache and forces a full re-prefill every turn.

### 3. (Optional) Raise the GPU memory ceiling

macOS caps GPU-usable unified memory at ~75% by default. The configurations here run within the default limit; raising it gives extra headroom for long contexts:

```bash
sudo sysctl iogpu.wired_limit_mb=28672   # 28 GB; resets on reboot
```

### 4. Launch the server

[`scripts/start-qwen.sh`](scripts/start-qwen.sh) starts `llama-server` on `http://0.0.0.0:8080` (override the bind/context with `LLAMA_HOST` / `LLAMA_CTX`). It defaults to the dense 27B at a 32K context. The flags, and why each matters on this hardware:

| flag | rationale |
|------|-----------|
| `-ngl 99` | offload all layers to the Metal GPU |
| `-c 32768` | 32K context. The dense 27B aborts at 64K on this host (compute-buffer limit); 32K is the recommended operating point |
| `-np 1` | single sequence — full window per request, lowest KV cost |
| `-fa on` | flash attention; required for the quantized KV cache below |
| `--cache-type-k q8_0 --cache-type-v q4_0` | quantize the KV cache to fit longer contexts in memory |
| `--cache-ram 2048` | cap the server-side prompt cache at 2 GB (default is 8 GB/instance, a major source of swap) |
| `-t 6` | use the 6 performance cores |
| `--jinja` | use the model's real chat template → correct tool-calling |
| `--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0` | Qwen's recommended (non-thinking) sampling |

```bash
chmod +x scripts/start-qwen.sh
./scripts/start-qwen.sh                       # dense 27B @ 32K on :8080
LLAMA_CTX=32768 ./scripts/start-qwen.sh       # explicit context override
```

---

## Usage

### Browser UI
Open **http://127.0.0.1:8080** — a full chat UI that handles the reasoning display.

### OpenAI-compatible API
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages":[{"role":"user","content":"Explain recursion to a 10-year-old"}],
  "max_tokens":1500
}' | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

### Tool-calling demo
[`bench/tool-test.py`](bench/tool-test.py) gives the model three real tools (`get_time`, `list_dir`, `calculate`) and runs the full agentic loop, printing each tool call:

```bash
python3 bench/tool-test.py "What time is it, what's in ~/llm, and what is 1234 * 5678?"
```

### OpenCode (coding agent)
Point [OpenCode](https://opencode.ai) at the local server with [`config/opencode.json`](config/opencode.json) (copy it to `~/.config/opencode/opencode.json`):

```bash
cd your-project
opencode                                                  # pick "llama.cpp (local)"
# or headless:
opencode run --model "llamacpp/Qwen3.6-27B-Q4_K_S.gguf" "summarize this project"
```

---

## Benchmarks

All numbers are measured on the hardware above by driving the server over HTTP and reading each response's `timings` object (no log scraping).

### Dense 27B vs MoE 35B-A3B — head-to-head (build `b9466`)

Identical 12-turn growing conversation replayed against each model with a stable prefix. Full methodology and tables: [results/comparison-dense-27b-vs-moe-35b-a3b-20260602.md](results/comparison-dense-27b-vs-moe-35b-a3b-20260602.md).

| Metric | Dense 27B | MoE 35B-A3B |
|---|---:|---:|
| GGUF size (Q4_K_S) | **15.86 GB** | 20.0 GB |
| Wired @ 32K context | **17.9 GB** | 22.4 GB |
| Host swap (agent workload) | **~1 GB** | ~7.6 GB |
| Prefill @ ~9K ctx (tok/s) | 12.9 | **38.6** |
| Decode (tok/s) | 5.3 | **25.4** |
| Warm-turn wall time @ ~9K ctx | 43.5 s | **13.5 s** |
| Cross-turn cache reuse (stable prefix) | 87% | 87% |
| Max context (this host) | ≤ 48K | **64K** |

### MoE 35B-A3B — throughput vs context depth (build `b9384`, `bench/bench.py`)

Generation slows as context grows; prefill is slowest at deep context as attention cost rises:

| context depth (tok) | prefill tok/s | generation tok/s |
|--------------------:|--------------:|-----------------:|
| ~0 | 79 | **25.6** |
| ~2,000 | 183 | 22.4 |
| ~8,000 | 74 | 16.6 |
| ~16,000 | 37 | 12.4 |

### Cold vs warm time-to-first-token (35B-A3B, 7.5K-token prompt)

The headline M1 Pro caveat, and why prompt caching matters:

| mode | TTFT (s) | speedup |
|------|---------:|--------:|
| cold (no cache) | **92.8** | 1.0× |
| warm (cache hit) | **0.2** | 541× |

### Tool-calling reliability (10 sequential multi-tool episodes, `--jinja`)

| clean runs | avg rounds | malformed |
|-----------:|-----------:|----------:|
| **10/10** | 2 | 0 |

**Reproduce the depth/TTFT/tool sweep:** start the server, then `python3 bench/bench.py --out results/bench-$(date +%Y%m%d).md` (add `--quick` for a smoke test).

---

## Key findings

1. **Prefill, not generation, is the M1 Pro bottleneck.** Generation is steady; processing a large prompt is what's slow. A large agent system prompt therefore produces a long time-to-first-token on a cold session.

2. **Prompt caching is essential — and it works for these hybrid models on a stable prefix.** With build `b9466` and a *byte-stable* prompt prefix, llama.cpp restores its hybrid context checkpoint each turn and prefills only the new tokens (measured ~87% reuse, zero full re-prefills, for *both* models). The frequently-reported "re-prefill every turn" behavior on Gated DeltaNet models is caused by an *unstable* client-side prefix, not by the model or server.

3. **The dense 27B has a context ceiling the MoE does not.** The dense model aborts at a 64K context (compute buffer exceeds the ~25 GB Metal working-set limit during init: 32K → 17.9 GB, 48K → 18.6 GB, 64K → crash), because it activates all parameters per token and its per-pass compute scales with context. The sparse MoE — whose bulk is static weights and whose per-pass compute is small — runs to 64K. **Operate the dense 27B at ≤ 48K (32K recommended).**

4. **Disabling reasoning is a request parameter, not a soft token.** These are reasoning models and emit a hidden chain-of-thought first. Always set `max_tokens` ≥ 1000, or `content` can come back empty (truncated mid-thought). The `/no_think` soft token is unreliable; pass `chat_template_kwargs: {"enable_thinking": false}` to reliably disable thinking (verified: 0 reasoning tokens).

5. **Tool-calling: use llama.cpp + `--jinja`.** See [Why llama.cpp, not Ollama](#why-llamacpp-not-ollama).

6. **Speculative decoding (MTP) is a GPU lever, not a CPU one — and not for agents.** Qwen3.5/3.6 ship a built-in Multi-Token-Prediction head; mainline llama.cpp uses it via `--spec-type draft-mtp` with an MTP-bundled GGUF. It speeds *generation* (~1.4–2.2× reported on GPU), not prefill, so it does not address the cold-TTFT caveat — and it **breaks the prompt cache**, forcing a full re-prefill every turn, which is a net loss for multi-turn agentic use. On a CPU-only edge box (4-core Pi 5) it was a net loss outright. Treat MTP as a GPU/Metal lever for single-shot long generations only.

---

## Running on less memory (the MoE RAM math)

A 35B-A3B MoE is a fast *big* model, but "only 3B active" is often misread as "only needs ~3B of RAM." It does not. Two different counts govern two different things:

- **Active params (`A3B`) → speed** — how much compute and memory bandwidth runs per token.
- **Total params (`35B`) → RAM** — *all* experts must be resident, because any token can route to any expert and the routing cannot be predicted.

So Qwen3.6-35B-A3B needs the full **~20 GB** for weights regardless of how few experts activate per token — which is why it wants a 32 GB unified-memory machine. **MoE = pay RAM for a big model, pay compute for a small one:** a strong deal when RAM is spare, unhelpful when RAM is the bottleneck. A dense model inverts the trade: less RAM, more compute per token.

**On a smaller box (8–16 GB, or a Raspberry Pi):** pick a model whose *total* fits in RAM — a dense ≤4–8B, or a **small-total MoE with few active params** such as `LFM2-8B-A1B` (8.3B total / 1.5B active, ~5 GB at Q4) or `Granite-4.0-H-Tiny` (~7B / 1B). Those run fast *and* fit. (A 30B-A3B can be mmap'd out-of-core from fast NVMe at aggressive quant, but on an SD-card Pi that expert paging crawls.)

---

## Assessment

For **private, offline chat, reasoning, and short-context tool use** on an M1 Pro / 32 GB, both models are genuinely useful local assistants. For **long-context agentic loops**, prefill cost makes either slower than a hosted frontier model — usable, with a noticeable wait on cold sessions.

Within the 32 GB envelope, the dense 27B is the better default when memory pressure or quality matters (it removes swap and scores higher on benchmarks), and the MoE 35B-A3B is the better choice when per-token speed or 64K context matters. Neither is a drop-in replacement for a fast hosted frontier model; both are capable, controllable, and fully local.

---

## Credits & licenses

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — MIT (ggml-org). Binaries are downloaded from their releases, not redistributed here.
- **Qwen3.6-27B / Qwen3.6-35B-A3B** — see each model's license on its [Hugging Face page](https://huggingface.co/Qwen).
- **This repository's own scripts/docs** — [MIT](LICENSE).
