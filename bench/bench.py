#!/usr/bin/env python3
"""Stress-test / benchmark harness for a local llama-server (Apple Silicon).

Drives the OpenAI-compatible + native llama.cpp endpoints over HTTP and reads the
per-response `timings` object (build-independent; we never scrape the log).

Modules:
  1. generation tok/s vs context depth
  2. prefill (prompt-processing) tok/s vs context depth
  3. cold vs warm time-to-first-token (prompt-cache hit)
  4. thinking vs /no_think on the same question
  5. tool-calling reliability over K rounds
  6. memory footprint

Usage:
  python3 bench.py --out results/bench-$(date +%Y%m%d).md
  LLAMA_BASE=http://127.0.0.1:8080 LLAMA_MODEL=... python3 bench.py --quick

Dependencies: python3 standard library only.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

BASE = os.environ.get("LLAMA_BASE", "http://127.0.0.1:8080")
MODEL = os.environ.get("LLAMA_MODEL", "Qwen3.6-35B-A3B-Q4_K_S.gguf")
TIMEOUT = 1200

# A fixed filler paragraph used to pad prompts to a target token depth.
FILLER = (
    "The unified memory architecture of Apple Silicon lets the GPU and CPU share "
    "one pool of RAM, which is what makes running a large quantized language model "
    "on a laptop feasible at all. "
)


# --------------------------------------------------------------------------- HTTP
def _post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def _get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.load(r)


def count_tokens(text):
    """Exact token count via the server's /tokenize endpoint."""
    return len(_post("/tokenize", {"content": text})["tokens"])


def build_prompt(target_tokens):
    """Return a user string whose token count is within ~5% of target_tokens."""
    if target_tokens <= 0:
        return "Hi."
    # Estimate, then correct using the real tokenizer.
    approx_tok_per_filler = max(1, count_tokens(FILLER))
    n = max(1, target_tokens // approx_tok_per_filler)
    text = FILLER * n
    # converge
    for _ in range(6):
        c = count_tokens(text)
        if abs(c - target_tokens) <= max(20, target_tokens * 0.05):
            break
        ratio = target_tokens / max(1, c)
        n = max(1, int(n * ratio))
        text = FILLER * n
    return text


def chat(messages, max_tokens, no_think=False, tools=None, cache_prompt=True,
         extra=None):
    if no_think and messages and messages[-1]["role"] == "user":
        messages = messages[:-1] + [
            {**messages[-1], "content": messages[-1]["content"] + " /no_think"}
        ]
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "cache_prompt": cache_prompt,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if extra:
        body.update(extra)
    return _post("/v1/chat/completions", body)


def stream_ttft(messages, max_tokens, cache_prompt):
    """Stream a chat request; return (ttft_seconds, total_seconds)."""
    body = {
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.7, "cache_prompt": cache_prompt, "stream": True,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    ttft = None
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                ttft = time.monotonic() - start
    total = time.monotonic() - start
    return ttft if ttft is not None else total, total


def median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


# --------------------------------------------------------------------- modules
def mod_gen_and_prefill(depths, reps):
    """Modules 1 + 2: gen tok/s and prefill tok/s vs context depth."""
    rows = []
    for depth in depths:
        prompt = build_prompt(depth)
        actual = count_tokens(prompt)
        gen_speeds, prefill_speeds = [], []
        for _ in range(reps):
            # generation: ask for real output
            g = chat([{"role": "user", "content": prompt + "\n\nWrite ~200 words of prose."}],
                     max_tokens=256, no_think=True, cache_prompt=False)
            t = g.get("timings", {})
            gen_speeds.append(t.get("predicted_per_second"))
            prefill_speeds.append(t.get("prompt_per_second"))
        rows.append({
            "depth": actual,
            "prefill": median(prefill_speeds),
            "gen": median(gen_speeds),
        })
        log(f"  depth ~{depth:>5} (actual {actual}): prefill {rows[-1]['prefill']} t/s, "
            f"gen {rows[-1]['gen']} t/s")
    return rows


def mod_cold_warm_ttft(big_depth):
    """Module 3: cold vs warm TTFT on a large prompt (mirrors an agent system prompt)."""
    prompt = build_prompt(big_depth)
    actual = count_tokens(prompt)
    msgs = [{"role": "user", "content": prompt + "\n\nReply with one short sentence. /no_think"}]
    cold_ttft, _ = stream_ttft(msgs, max_tokens=64, cache_prompt=False)
    warm_ttft, _ = stream_ttft(msgs, max_tokens=64, cache_prompt=True)
    speedup = round(cold_ttft / warm_ttft, 1) if warm_ttft else None
    log(f"  prompt {actual} tok: cold TTFT {round(cold_ttft,1)}s, warm {round(warm_ttft,1)}s, {speedup}x")
    return {"tokens": actual, "cold": round(cold_ttft, 1),
            "warm": round(warm_ttft, 1), "speedup": speedup}


def mod_thinking(question, max_tokens):
    """Module 4: thinking vs /no_think on the same question."""
    out = {}
    for mode, no_think in (("thinking", False), ("/no_think", True)):
        start = time.monotonic()
        r = chat([{"role": "user", "content": question}], max_tokens=max_tokens,
                 no_think=no_think, cache_prompt=False)
        wall = time.monotonic() - start
        m = r["choices"][0]["message"]
        t = r.get("timings", {})
        reason = m.get("reasoning_content") or ""
        final = m.get("content") or ""
        out[mode] = {
            "wall": round(wall, 1),
            "reason_tok": count_tokens(reason) if reason else 0,
            "final_tok": count_tokens(final) if final else 0,
            "gen": round(t.get("predicted_per_second"), 1) if t.get("predicted_per_second") else None,
            "finish": r["choices"][0].get("finish_reason"),
        }
        log(f"  {mode}: {out[mode]['wall']}s, reason {out[mode]['reason_tok']} tok / "
            f"final {out[mode]['final_tok']} tok, {out[mode]['gen']} t/s")
    return out


TOOLS_SPEC = [
    {"type": "function", "function": {"name": "get_time",
        "description": "Get the current local date and time",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "list_dir",
        "description": "List files in a directory",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path"}}}}},
    {"type": "function", "function": {"name": "calculate",
        "description": "Evaluate a basic arithmetic expression",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}}, "required": ["expression"]}}},
]
TOOL_IMPL = {
    "get_time": lambda: "Monday 2026-05-28 18:00:00",
    "list_dir": lambda path=".": "a.txt\nb.txt",
    "calculate": lambda expression="": str(eval(expression, {"__builtins__": {}}, {}))
                 if set(expression) <= set("0123456789+-*/(). ") else "error",
}


def run_tool_episode(max_rounds=6):
    """One agentic tool episode. Returns (clean, rounds)."""
    messages = [{"role": "user", "content":
                 "What is 1234 * 5678, and what time is it? Use the tools."}]
    used_tool = False
    for rnd in range(1, max_rounds + 1):
        r = chat(messages, max_tokens=2000, tools=TOOLS_SPEC, cache_prompt=False)
        msg = r["choices"][0]["message"]
        messages.append(msg)
        calls = msg.get("tool_calls")
        if calls:
            used_tool = True
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    return False, rnd  # malformed args = not clean
                try:
                    result = TOOL_IMPL.get(name, lambda **k: "unknown")(**args)
                except Exception as e:
                    result = f"error: {e}"
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""),
                                 "content": str(result)})
            continue
        # final assistant message; clean if a tool was used and content exists
        return (used_tool and bool(msg.get("content"))), rnd
    return False, max_rounds


def mod_tools(k):
    """Module 5: tool-calling reliability over k episodes."""
    clean = 0
    rounds = []
    malformed = 0
    for i in range(k):
        ok, rnd = run_tool_episode()
        rounds.append(rnd)
        if ok:
            clean += 1
        else:
            malformed += 1
        log(f"  episode {i+1}/{k}: {'clean' if ok else 'FAILED'} in {rnd} rounds")
    return {"k": k, "clean": clean, "avg_rounds": round(statistics.mean(rounds), 1),
            "failed": malformed}


def mod_memory():
    """Module 6: memory footprint from props + ps + sysctl."""
    info = {}
    try:
        out = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True)
        pid = out.stdout.strip().split("\n")[0]
        info["pid"] = pid
        if pid:
            rss = subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True)
            kb = int(rss.stdout.strip() or 0)
            info["rss_gb"] = round(kb / 1024 / 1024, 2)
    except Exception as e:
        info["ps_error"] = str(e)
    try:
        wl = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"],
                            capture_output=True, text=True)
        info["wired_limit_mb"] = wl.stdout.strip()
    except Exception:
        pass
    log(f"  pid {info.get('pid')}, RSS {info.get('rss_gb')} GB, "
        f"wired_limit_mb {info.get('wired_limit_mb')}")
    return info


# ----------------------------------------------------------------------- output
def log(msg):
    print(msg, flush=True)


def render(props, gen_rows, ttft, think, tools, mem, started):
    dg = props.get("default_generation_settings", {})
    n_ctx = props.get("n_ctx") or dg.get("n_ctx")
    lines = []
    A = lines.append
    A(f"# Benchmark — {MODEL}\n")
    A(f"- **Date:** {started}")
    A(f"- **Build:** {props.get('build_info')}")
    A(f"- **Model path:** `{props.get('model_path')}`")
    A(f"- **Context (n_ctx):** {n_ctx}")
    A(f"- **Slots / parallel:** {props.get('total_slots')}")
    A(f"- **Hardware:** Apple M1 Pro, 32 GB unified memory, 14-core GPU, macOS 26.5")
    A(f"- **Measurement:** median of {REPS} reps; speeds read from each response's `timings`.\n")

    if gen_rows:
        A("## 1 & 2. Generation & prefill vs context depth\n")
        A("| context depth (tok) | prefill tok/s | generation tok/s |")
        A("|--------------------:|--------------:|-----------------:|")
        for r in gen_rows:
            A(f"| {r['depth']} | {r['prefill']} | {r['gen']} |")
        A("")

    if ttft and ttft.get("tokens"):
        A("## 3. Cold vs warm time-to-first-token\n")
        A(f"Large prompt (~{ttft['tokens']} tokens, mirrors an agent system prompt):\n")
        A("| mode | TTFT (s) | speedup |")
        A("|------|---------:|--------:|")
        A(f"| cold (no cache) | {ttft['cold']} | 1.0x |")
        A(f"| warm (cache hit) | {ttft['warm']} | {ttft['speedup']}x |")
        A("")

    if think and think.get("thinking"):
        A("## 4. Thinking vs /no_think (same question)\n")
        A("| mode | wall (s) | reasoning tok | final tok | gen tok/s | finish |")
        A("|------|---------:|--------------:|----------:|----------:|--------|")
        for mode in ("thinking", "/no_think"):
            d = think[mode]
            A(f"| {mode} | {d['wall']} | {d['reason_tok']} | {d['final_tok']} | {d['gen']} | {d['finish']} |")
        A("")

    if tools and tools.get("k"):
        A("## 5. Tool-calling reliability\n")
        A(f"{tools['k']} sequential agentic episodes (multi-tool prompt), via llama.cpp `--jinja`:\n")
        A("| clean runs | avg rounds | failed |")
        A("|-----------:|-----------:|-------:|")
        A(f"| {tools['clean']}/{tools['k']} | {tools['avg_rounds']} | {tools['failed']} |")
        A("")

    if mem:
        A("## 6. Memory\n")
        A("> Note: process RSS under-reports memory for memory-mapped (mmap) model weights — "
          "the 21.5 GB of weights live mostly in the unified-memory file cache / Metal wired "
          "allocation, not process RSS.\n")
        A("| llama-server RSS | iogpu.wired_limit_mb |")
        A("|-----------------:|---------------------:|")
        A(f"| {mem.get('rss_gb')} GB | {mem.get('wired_limit_mb')} (0 = macOS default ~75%) |")
        A("")
    return "\n".join(lines)


REPS = 3


def main():
    global REPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="fewer depths/reps for a smoke test")
    ap.add_argument("--only", default="", help="comma list: gen,ttft,think,tools,mem")
    args = ap.parse_args()

    only = set(s.strip() for s in args.only.split(",") if s.strip())
    run = lambda name: (not only) or (name in only)

    REPS = 1 if args.quick else 3
    depths = [0, 2048] if args.quick else [0, 2048, 8192, 16384]
    big = 4096 if args.quick else 7700
    tool_k = 3 if args.quick else 10

    started = subprocess.run(["date", "+%Y-%m-%d %H:%M"], capture_output=True, text=True).stdout.strip()
    log(f"== bench against {BASE} ({MODEL}) ==")
    props = _get("/props")

    gen_rows = ttft = think = tools = mem = None
    if run("gen"):
        log("[1+2] generation & prefill vs depth"); gen_rows = mod_gen_and_prefill(depths, REPS)
    if run("ttft"):
        log("[3] cold vs warm TTFT"); ttft = mod_cold_warm_ttft(big)
    if run("think"):
        log("[4] thinking vs /no_think")
        think = mod_thinking("A bat and ball cost $1.10. The bat costs $1.00 more than the ball. "
                             "How much is the ball?", 2000)
    if run("tools"):
        log("[5] tool-calling reliability"); tools = mod_tools(tool_k)
    if run("mem"):
        log("[6] memory"); mem = mod_memory()

    report = render(props, gen_rows or [], ttft or {"tokens": 0, "cold": "-", "warm": "-", "speedup": "-"},
                    think or {"thinking": {}, "/no_think": {}},
                    tools or {"k": 0, "clean": "-", "avg_rounds": "-", "failed": "-"},
                    mem or {}, started)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(report)
        log(f"\nwrote {args.out}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
