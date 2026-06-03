#!/usr/bin/env python3
"""Read two result JSONs from run_eval.py and print a side-by-side comparison.

Usage: python3 compare.py <labelA> <labelB>   (reads ../results/<label>.json;
override the directory with the BENCH_RESULTS_DIR env var).
"""
import json, sys, os

_RESULTS = os.environ.get("BENCH_RESULTS_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "results")

def load(label):
    p = os.path.join(_RESULTS, f"{label}.json")
    with open(p) as f: return json.load(f)

def summarize_prefill(d):
    rows = d.get("prefill", [])
    rows = [r for r in rows if "prefilled_n" in r and r["prefilled_n"] is not None]
    if not rows: return None
    last_half = rows[len(rows)//2:]               # steady-state turns
    avg_prefill = sum(r["prefilled_n"] for r in last_half)/len(last_half)
    avg_ctx = sum((r.get("ctx_tokens") or 0) for r in last_half)/len(last_half)
    reuse = 1 - avg_prefill/avg_ctx if avg_ctx else 0
    restored = sum(1 for r in rows if r.get("cache_event") == "restored")
    forcing = sum(1 for r in rows if r.get("cache_event") == "forcing-full")
    avg_wall = sum(r.get("wall_s") or 0 for r in rows)/len(rows)
    return dict(avg_prefill=avg_prefill, avg_ctx=avg_ctx, reuse=reuse,
                restored=restored, forcing=forcing, n=len(rows), avg_wall=avg_wall)

def summarize_decode(d):
    rows = [r for r in d.get("decode", []) if r.get("decode_tps")]
    if not rows: return None
    return sum(r["decode_tps"] for r in rows)/len(rows)

def main():
    labels = sys.argv[1:] or ["qwen35b-a3b", "qwen27b-dense"]
    data = {}
    for l in labels:
        try: data[l] = load(l)
        except FileNotFoundError: print(f"(missing results for {l})");
    print("\n" + "="*72)
    print("PREFILL / CACHE REUSE  (steady-state = 2nd half of turns)")
    print("="*72)
    print(f"{'model':<18}{'avg ctx':>9}{'avg prefill':>13}{'reuse%':>9}{'restored':>10}{'forcing':>9}{'avg wall':>10}")
    for l in labels:
        if l not in data: continue
        s = summarize_prefill(data[l])
        if not s: print(f"{l:<18} (no prefill data)"); continue
        print(f"{l:<18}{s['avg_ctx']:>9.0f}{s['avg_prefill']:>13.0f}{s['reuse']*100:>8.0f}%"
              f"{s['restored']:>10}{s['forcing']:>9}{s['avg_wall']:>9.1f}s")
    print("\n" + "="*72)
    print("DECODE THROUGHPUT")
    print("="*72)
    for l in labels:
        if l not in data: continue
        tps = summarize_decode(data[l])
        print(f"{l:<18}{(str(round(tps,1))+' tok/s') if tps else 'n/a':>14}")
    print("\n" + "="*72)
    print("QUALITY OUTPUTS  (see results/*.json for full text)")
    print("="*72)
    for l in labels:
        if l not in data: continue
        print(f"\n--- {l} ---")
        for q in data[l].get("quality", []):
            preview = (q["output"][:240].replace("\n", " ")) if q.get("output") else ""
            print(f"  [{q['id']}] {q.get('wall_s')}s: {preview}...")

if __name__ == "__main__":
    main()
