#!/usr/bin/env python3
"""
Isolated bake-off harness: compare a candidate llama-server (Qwen3.6-27B) against
the production 35B-A3B on the things that actually matter for this agent:

  TEST A  PREFILL / CACHE REUSE  (the core question: "is the prefill issue fixed?")
          Replay a FIXED, deterministic, growing multi-turn conversation. Each turn
          re-sends the whole conversation-so-far. If cross-turn cache reuse works,
          the server should only PREFILL the new tokens each turn (prompt_n stays
          small); if it's broken, prompt_n ~ full context every turn (full reprefill).
          We also tail the server log for "restored context checkpoint" vs
          "forcing full prompt re-processing".

  TEST B  DECODE THROUGHPUT  (tokens/sec generation).

  TEST C  QUALITY  (reasoning / agentic tool-calling / coding) — outputs saved for
          side-by-side human + LLM-judge comparison.

Uses ONLY the Python stdlib (no pip installs) so it stays isolated.
Identical token sequences are sent to every model (fixed scripted conversation +
fixed assistant turns), so the prefill comparison is fair.
"""
import argparse, json, time, urllib.request, urllib.error, os, sys

def post(base_url, payload, timeout=900):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body, time.time() - t0

def logmarks(path):
    """Return (#restored, #forcing_full) currently in the server log."""
    if not path or not os.path.exists(path):
        return (None, None)
    restored = forcing = 0
    with open(path, "rb") as f:
        for line in f:
            s = line.decode(errors="ignore")
            if "restored context checkpoint" in s: restored += 1
            elif "forcing full prompt re-processing" in s: forcing += 1
    return (restored, forcing)

def build_conversation(n_turns=12, filler_per_turn=420):
    """Deterministic growing conversation. Fixed user+assistant text per turn so
    every model prefills identical tokens. ~filler words/turn drive context growth."""
    sys_msg = {"role": "system", "content":
        "You are a meticulous senior software engineer assisting with a long-running "
        "project. Keep prior context in mind and answer precisely."}
    msgs = [sys_msg]
    topics = [
        "designing a write-ahead journal for a memory queue",
        "bounding per-turn latency under a fixed token budget",
        "choosing a KV-cache quantization for a 32GB box",
        "structuring a LangGraph perpetual-learning loop",
        "deduplicating episodic nodes in a knowledge graph",
        "handling backpressure when an embedding service is slow",
        "stabilizing a prompt prefix to maximize cache reuse",
        "compaction strategy that preserves durable knowledge",
        "graceful degradation when the GPU is contended",
        "idempotent recovery after a crash mid-ingestion",
        "rate-limiting tool calls without starving the planner",
        "observability: what to log for a self-reflective agent",
    ]
    pad = ("Consider edge cases, failure modes, ordering guarantees, and the "
           "memory/latency trade-off carefully. ")
    for i in range(n_turns):
        t = topics[i % len(topics)]
        user = (f"Turn {i+1}: Let's discuss {t}. " + pad * 3 +
                f"Summarize the key decision for step {i+1} in one sentence.")
        # fixed canned assistant reply (deterministic, sized by filler)
        asst = (f"For step {i+1} on {t}, the key decision is to favor correctness and "
                f"recoverability over raw speed. " + pad * (filler_per_turn // len(pad.split())))
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": asst})
    return msgs

def test_prefill(base_url, conv, log_path, label, out):
    print(f"\n=== TEST A: PREFILL/CACHE REUSE ({label}) ===")
    rows = []
    # turn k uses messages up to the k-th user message, asking for a tiny completion
    n_turns = (len(conv) - 1) // 2
    for k in range(1, n_turns + 1):
        upto = conv[: 1 + (k - 1) * 2 + 1]  # system + (k-1) full exchanges + k-th user
        before = logmarks(log_path)
        payload = {"model": label, "messages": upto, "max_tokens": 8,
                   "temperature": 0, "cache_prompt": True, "stream": False}
        try:
            body, wall = post(base_url, payload)
        except Exception as e:
            print(f"  turn {k}: ERROR {e}"); rows.append({"turn": k, "error": str(e)}); continue
        tm = body.get("timings", {}) or {}
        usage = body.get("usage", {}) or {}
        prompt_n = tm.get("prompt_n", usage.get("prompt_tokens"))
        prompt_ms = tm.get("prompt_ms")
        after = logmarks(log_path)
        d_restored = (after[0] - before[0]) if before[0] is not None else None
        d_forcing = (after[1] - before[1]) if before[1] is not None else None
        cache_event = ("restored" if d_restored else ("forcing-full" if d_forcing else "?"))
        rows.append({"turn": k, "ctx_tokens": usage.get("prompt_tokens"),
                     "prefilled_n": prompt_n, "prefill_ms": prompt_ms,
                     "prefill_tps": tm.get("prompt_per_second"),
                     "wall_s": round(wall, 2), "cache_event": cache_event})
        print(f"  turn {k:2d}: ctx={usage.get('prompt_tokens'):>6} | prefilled={prompt_n:>6} "
              f"| {round((prompt_ms or 0)/1000,1):>5}s | {cache_event}")
    out["prefill"] = rows

def test_decode(base_url, label, out, reps=3):
    print(f"\n=== TEST B: DECODE THROUGHPUT ({label}) ===")
    prompt = [{"role": "user", "content":
        "Write a detailed, 300-word technical explanation of how a write-ahead log "
        "guarantees durability and crash recovery in a queueing system."}]
    rows = []
    for r in range(reps):
        payload = {"model": label, "messages": prompt, "max_tokens": 256,
                   "temperature": 0.6, "top_p": 0.95, "cache_prompt": False, "stream": False}
        try:
            body, wall = post(base_url, payload)
        except Exception as e:
            print(f"  rep {r+1}: ERROR {e}"); continue
        tm = body.get("timings", {}) or {}
        tps = tm.get("predicted_per_second")
        rows.append({"rep": r+1, "predicted_n": tm.get("predicted_n"),
                     "decode_tps": tps, "wall_s": round(wall, 2)})
        print(f"  rep {r+1}: {tm.get('predicted_n')} tok @ {round(tps or 0,1)} tok/s ({round(wall,1)}s)")
    out["decode"] = rows

def test_quality(base_url, label, out):
    print(f"\n=== TEST C: QUALITY ({label}) ===")
    prompts = [
        {"id": "reason", "messages": [{"role": "user", "content":
            "A train leaves city A at 9:00 going 60 km/h. Another leaves city B "
            "(300 km away) at 9:30 going 90 km/h toward A. At what clock time do they "
            "meet? Show your reasoning step by step, then give the final time."}]},
        {"id": "code", "messages": [{"role": "user", "content":
            "Write a Python function `dedupe_episodes(nodes)` that removes duplicate "
            "episodic memory nodes (same .content within 5s of each other by .ts), "
            "keeping the earliest. Include a docstring and handle empty input."}]},
        {"id": "tool", "messages": [
            {"role": "system", "content":
                "You have a tool `graph_query(cypher: str)`. When you need data, respond "
                "ONLY with a JSON object {\"tool\":\"graph_query\",\"args\":{\"cypher\":\"...\"}}."},
            {"role": "user", "content":
                "How many Episodic nodes are in the graph? Use the tool."}]},
    ]
    rows = []
    for p in prompts:
        payload = {"model": label, "messages": p["messages"], "max_tokens": 1024,
                   "temperature": 0.6, "top_p": 0.95, "cache_prompt": False, "stream": False}
        try:
            body, wall = post(base_url, payload)
            msg = body["choices"][0]["message"]
            text = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            tm = body.get("timings", {}) or {}
        except Exception as e:
            text = f"ERROR {e}"; reasoning = ""; wall = 0; tm = {}
        rows.append({"id": p["id"], "wall_s": round(wall, 2), "output": text,
                     "reasoning": reasoning, "decode_tps": tm.get("predicted_per_second"),
                     "predicted_n": tm.get("predicted_n")})
        print(f"  [{p['id']}] {round(wall,1)}s, content={len(text)} chars, "
              f"reasoning={len(reasoning)} chars, {round(tm.get('predicted_per_second') or 0,1)} tok/s")
    out["quality"] = rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--server-log", default="")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "results"))
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--skip", default="", help="comma list: prefill,decode,quality")
    args = ap.parse_args()

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    conv = build_conversation(n_turns=args.turns)
    out = {"label": args.label, "base_url": args.base_url, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    if "prefill" not in skip: test_prefill(args.base_url, conv, args.server_log, args.label, out)
    if "decode"  not in skip: test_decode(args.base_url, args.label, out)
    if "quality" not in skip: test_quality(args.base_url, args.label, out)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.label}.json")
    with open(path, "w") as f: json.dump(out, f, indent=2)
    print(f"\nwrote {path}")

if __name__ == "__main__":
    main()
