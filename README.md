# local-llm-apple-silicon

**Running a 35B reasoning MoE fully locally on a MacBook Pro M1 Pro (32 GB) — at usable speed, with working tool-calling.**

This repo documents an end-to-end setup for running [Qwen3.6-35B-A3B](https://qwen.ai) locally via [llama.cpp](https://github.com/ggml-org/llama.cpp) on Apple Silicon, plus honest benchmarks and the gotchas I hit along the way. It's a guide as much as a config dump — the value is the *methodology and the measured numbers*, not the (uncommittable) 20 GB of weights.

> **Platform:** macOS 26.5 · Apple M1 Pro · 32 GB unified memory · 14-core GPU
> **Stack:** llama.cpp (build `b9384`, Metal) · Qwen3.6-35B-A3B `Q4_K_S` · OpenCode

---

## TL;DR verdict

**Worth it if** you want a private, offline, genuinely capable model with real tool-calling for chat, reasoning, and short-context agentic work. The MoE design (35B total, ~3B active per token) is what makes a model this smart run at interactive speed on a laptop.

**Painful if** you need fast responses on *long* contexts: on the M1 Pro the bottleneck is **prompt processing (prefill)**, not generation — a large agent system prompt can mean a noticeable wait before the first token on a cold session. Generation itself is fine.

See [Benchmarks](#benchmarks) for the actual numbers.

---

## Hardware

| | |
|---|---|
| Machine | MacBook Pro 14" (MacBookPro18,3) |
| Chip | Apple M1 Pro (8-core CPU: 6P+2E, 14-core GPU) |
| Memory | 32 GB unified |
| OS | macOS 26.5 |

The key enabler is **unified memory**: the GPU can address the same 32 GB the CPU uses, so a 21.5 GB model fits where a discrete GPU with 8–12 GB VRAM never could.

---

## Why this model (the journey)

I started with `Qwen3-30B-A3B-Instruct-2507` (18.6 GB) and upgraded to the newer **Qwen3.6-35B-A3B** (released April 2026), which scores higher across reasoning/coding/tool benchmarks and fits the same machine.

- **MoE, 35B total / ~3B active.** Only a small expert subset runs per token, so it's *much* faster than a dense 35B while keeping a large model's knowledge.
- **It's a reasoning model.** It emits a hidden chain-of-thought to a separate `reasoning_content` field before the final `content`. Great for hard problems; see [Gotchas](#gotchas--insights) for the catch.
- **Quant `Q4_K_S` (21.5 GB).** Chosen to fit 32 GB with headroom for KV cache and the OS. `Q4_K_M` (22.3 GB) is slightly higher quality but tighter; `IQ4_XS` (19.7 GB) gives more room for long context.

---

## Why llama.cpp, not Ollama

Ollama is the easy on-ramp, but at the time of writing it had **open bugs in Qwen3 tool-calling** that matter for agentic use:

- Tool definitions passed via the `tools` parameter were rendered as Go-struct strings instead of valid JSON.
- Prior tool calls were stripped from conversation history across turns.
- `thinking + tools` could return empty output.

llama.cpp's `llama-server` with `--jinja` uses the model's own chat template and emits clean, structured `tool_calls`. The [tool-calling reliability benchmark](#benchmarks) backs this up: **clean tool episodes, no malformed output.** If your use is plain chat, Ollama is fine; for tools, llama.cpp is the safer choice here.

---

## Install

### 1. llama.cpp (prebuilt, Metal-enabled)

Download the macOS arm64 release (this setup uses build `b9384`) from the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), then extract:

```bash
mkdir -p ~/llm && cd ~/llm
curl -L -o llama.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b9384/llama-b9384-bin-macos-arm64.tar.gz
tar -xzf llama.tar.gz          # -> ~/llm/llama-b9384/  (binaries + dylibs)
xattr -dr com.apple.quarantine ~/llm/llama-b9384   # clear Gatekeeper quarantine
```

> The `.dylib`s live next to the binaries; the launch script sets `DYLD_LIBRARY_PATH` accordingly.

### 2. The model (~21.5 GB)

```bash
mkdir -p ~/llm/models && cd ~/llm/models
curl -L -C - -o Qwen3.6-35B-A3B-Q4_K_S.gguf \
  https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen_Qwen3.6-35B-A3B-Q4_K_S.gguf
```

> Use `-C -` so the download resumes if it stalls. Plan for the full 21.5 GB.

### 3. (Optional) Raise the GPU memory ceiling

macOS caps GPU-usable unified memory at ~75% by default. At 21.5 GB weights + KV cache that's tight, so give it headroom:

```bash
sudo sysctl iogpu.wired_limit_mb=28672   # 28 GB; resets on reboot
```

### 4. Launch the server

[`scripts/start-qwen.sh`](scripts/start-qwen.sh) starts `llama-server` on `http://127.0.0.1:8080`. The flags, and why each matters on this hardware:

| flag | why |
|------|-----|
| `-ngl 99` | offload all layers to the Metal GPU |
| `-c 32768` | 32K context (model trains to 262K, but KV cache costs memory — a deliberate tradeoff) |
| `-fa on` | flash attention; required for the quantized KV cache below |
| `--cache-type-k q8_0 --cache-type-v q4_0` | quantize the KV cache to fit longer contexts in memory |
| `-t 6` | use the 6 performance cores |
| `--jinja` | use the model's real chat template → correct tool-calling |
| `--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0` | Qwen's recommended sampling |

```bash
chmod +x scripts/start-qwen.sh
./scripts/start-qwen.sh          # ~25s to load; serves on :8080
```

---

## Usage

### Browser UI
Open **http://127.0.0.1:8080** — a full chat UI that handles the reasoning display for you.

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

### OpenCode (real coding agent)
Point [OpenCode](https://opencode.ai) at the local server with [`config/opencode.json`](config/opencode.json) (copy it to `~/.config/opencode/opencode.json`). Then:

```bash
cd your-project
opencode                                                  # pick "llama.cpp (local)"
# or headless:
opencode run --model "llamacpp/Qwen3.6-35B-A3B-Q4_K_S.gguf" "summarize this project"
```

---

## Benchmarks

Measured on the hardware above with [`bench/bench.py`](bench/bench.py), which drives the server over HTTP and reads each response's `timings` object (no log scraping). Full results: [`results/`](results/).

**Generation & prefill vs context depth** — generation slows as context grows; prefill is slowest at deep context (attention cost rises):

| context depth (tok) | prefill tok/s | generation tok/s |
|--------------------:|--------------:|-----------------:|
| ~0 | 79 | **25.6** |
| ~2,000 | 183 | 22.4 |
| ~8,000 | 74 | 16.6 |
| ~16,000 | 37 | 12.4 |

**Cold vs warm time-to-first-token** (7.5K-token prompt, like an agent system prompt) — this is the headline M1 Pro caveat, and why prompt caching matters so much:

| mode | TTFT (s) | speedup |
|------|---------:|--------:|
| cold (no cache) | **92.8** | 1.0× |
| warm (cache hit) | **0.2** | 541× |

**Tool-calling reliability** — 10 sequential multi-tool agentic episodes via `--jinja`:

| clean runs | avg rounds | malformed |
|-----------:|-----------:|----------:|
| **10/10** | 2 | 0 |

**Thinking vs `/no_think`** (same question) — note the `/no_think` soft token did *not* suppress reasoning (see gotcha #1):

| mode | wall (s) | reasoning tok | final tok | gen tok/s |
|------|---------:|--------------:|----------:|----------:|
| thinking | 24.0 | 474 | 136 | 26.0 |
| `/no_think` | 25.8 | 651 | 5 | 26.1 |

**Reproduce:** start the server, then `python3 bench/bench.py --out results/bench-$(date +%Y%m%d).md` (add `--quick` for a fast smoke test).

---

## Gotchas & insights

1. **It's a reasoning model — give it room, and disabling thinking is finicky.** It "thinks" before answering. Always set `max_tokens` ≥ 1000, or the response gets cut off mid-thought and `content` comes back empty. The `/no_think` soft token is **unreliable** on this model — in my benchmark it still emitted 651 reasoning tokens. To actually disable thinking, pass `chat_template_kwargs: {"enable_thinking": false}` in the request (verified: 0 reasoning tokens, direct answer).
2. **Prefill, not generation, is the M1 Pro bottleneck.** Generation is steady; processing a large prompt is what's slow. This is why a big agent system prompt produces a long *time-to-first-token* on a cold session (see benchmark §3).
3. **Prompt caching saves you.** `llama-server` caches the prompt prefix, so follow-up turns in the same session reuse it and respond much faster — the cold wait is mostly one-time per session.
4. **Tool-calling: use llama.cpp + `--jinja`.** See [Why llama.cpp, not Ollama](#why-llamacpp-not-ollama).
5. **Tune the GPU wired limit** (`iogpu.wired_limit_mb`) for headroom on long contexts.

---

## Would I recommend it?

For **private/offline chat, reasoning, and short-context tool use** on an M1 Pro / 32 GB: yes, genuinely useful. For **long-context agentic loops**, the prefill cost makes it slower than cloud models — usable, but you'll feel the wait on cold sessions. The honest framing: this is a capable local assistant, not a drop-in replacement for a fast hosted frontier model.

---

## Credits & licenses

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — MIT (ggml-org). Binaries are downloaded from their releases, not redistributed here.
- **Qwen3.6-35B-A3B** — see the model's license on its [Hugging Face page](https://huggingface.co/Qwen).
- **This repo's own scripts/docs** — [MIT](LICENSE).
