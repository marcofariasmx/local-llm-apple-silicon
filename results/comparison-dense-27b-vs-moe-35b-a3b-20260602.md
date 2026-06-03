# Comparison — Dense Qwen3.6-27B vs MoE Qwen3.6-35B-A3B (Q4_K_S)

- **Date:** 2026-06-02
- **Build:** llama.cpp `b9466` (Metal)
- **Hardware:** Apple M1 Pro, 32 GB unified memory, 14-core GPU, macOS 26.5
- **Models:** `Qwen3.6-27B-Q4_K_S` (dense) vs `Qwen3.6-35B-A3B-Q4_K_S` (MoE, ~3B active)
- **Server flags:** `-ngl 99 -np 1 -fa on --cache-type-k q8_0 --cache-type-v q4_0 --cache-ram 2048 -t 6 --jinja`
- **Method:** a fixed, deterministic 12-turn *growing* conversation is replayed
  incrementally against each model with a byte-stable system prefix; per-turn
  `prompt_n` / `prompt_ms` / `predicted_per_second` are read from each response's
  `timings` object. Identical token sequences are sent to both models, so the
  prompt-cache and throughput comparison is like-for-like.

## 1. Memory footprint

| Metric | Dense 27B | MoE 35B-A3B |
|---|---:|---:|
| GGUF on disk (Q4_K_S) | **15.86 GB** | 20.0 GB |
| Wired @ 16K ctx | 17.4 GB | — |
| Wired @ 32K ctx | **17.9 GB** | 22.4 GB |
| Wired @ 40K ctx | 18.1 GB | — |
| Wired @ 48K ctx | 18.6 GB | — |
| Wired @ 64K ctx | **crash at init** | runs (~24 GB) |
| Host swap, full agent workload | **~1 GB** | ~7.6 GB |

The dense 27B's ~5 GB-smaller weights move a representative 32 GB workload back
below the swap threshold (≈7.6 GB → ≈1 GB swap).

## 2. The dense-model Metal ceiling (64K crashes the 27B)

Counter-intuitively, the *smaller* dense 27B fails to start at a 64K context where
the *larger* MoE 35B-A3B runs. Cause: a sparse MoE activates only ~3B parameters per
token, so its per-forward-pass compute buffer is small and its bulk is static
weights; a **dense** model activates all 27B parameters per token, so its compute
buffer (full-width activations, the 17,408-wide FFN intermediates, context-scaled
attention) grows with context and, at 64K, exceeds the M1 Pro's ~25 GB Metal
working-set limit during initialization. Measured startup wired: 32K → 17.9 GB,
48K → 18.6 GB, 64K → abort. **Operational limit for the dense 27B on this host: ≤ 48K;
32K recommended.**

## 3. Throughput — prefill (tok/s) vs conversation depth

Both models reused llama.cpp's hybrid context-checkpoint cache on every turn of a
stable prefix (11/11 restored, 0 full re-prefills), so each turn prefills only the
new tokens. Per-token prefill speed still falls with depth as attention cost rises:

| approx. context (tok) | Dense 27B | MoE 35B-A3B |
|---:|---:|---:|
| ~960 | 47.4 | 218.4 |
| ~2,600 | 31.6 | 111.9 |
| ~4,300 | 22.9 | 75.6 |
| ~6,000 | 18.5 | 57.3 |
| ~7,600 | 15.5 | 46.6 |
| ~9,300 | 12.9 | 38.6 |

## 4. Throughput — decode and per-turn latency

| Metric | Dense 27B | MoE 35B-A3B |
|---|---:|---:|
| Decode (generation) tok/s | **5.3** | 25.4 |
| Wall time, warm turn @ ~9K ctx | 43.5 s | 13.5 s |

The MoE is ~3–5× faster per token (decode ~4.8×) because it activates ~3B params per
token versus the dense model's full 27B. This is the cost paid for the 27B's smaller
memory footprint.

## 5. Prompt-cache reuse (correcting a common assumption)

On a **byte-stable prefix**, b9466's hybrid checkpoint restore works for *both*
architectures on this single Metal GPU — each turn logged `restored context
checkpoint` and prefilled only the new ~930 tokens (≈87% reuse), with **zero**
`forcing full prompt re-processing`. The "re-prefill every turn" behavior often
attributed to these hybrid (Gated DeltaNet) models is driven by an *unstable* prompt
prefix on the client side, not by the model or the server.

## 6. Quality (published benchmarks, for reference)

| Benchmark | Dense 27B | MoE 35B-A3B |
|---|---:|---:|
| BenchLM (provisional, aggregate) | 73 | 66 |
| Agentic (average) | 59.3 | 51.5 |
| SWE-bench Verified | 77.2 | 51.5 |

Source: vendor/community leaderboards (BenchLM, Artificial Analysis). The dense 27B
leads on quality, particularly agentic and coding tasks.

## Conclusion

On a 32 GB M1 Pro the choice is a clear trade, not a strict upgrade:

- **Dense Qwen3.6-27B** — ~5 GB less memory (eliminates swap on tight workloads),
  higher quality, but ~3–5× slower per token and capped at ≤48K context.
- **MoE Qwen3.6-35B-A3B** — markedly faster per token and runs to 64K, at ~4–5 GB
  more memory and lower benchmark quality.

Pick the dense 27B when memory headroom or output quality dominates; pick the MoE
when token throughput and context length dominate.
