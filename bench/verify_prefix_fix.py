"""Validate the prefix-stability fix against the live 27B (:8080).
Mimics the agent's build(): STABLE system + append-only history + a CHANGING
trailing user suffix (the ticking low-context warning). Proves that a suffix change
does NOT re-prefill the history."""
import json, urllib.request, time

BASE="http://127.0.0.1:8080/v1/chat/completions"
def call(messages, tag):
    body=json.dumps({"messages":messages,"max_tokens":4,"temperature":0,
                     "cache_prompt":True,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    req=urllib.request.Request(BASE,data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=300) as r: d=json.load(r)
    tm=d.get("timings",{}); u=d.get("usage",{})
    print(f"  {tag:32s} ctx={u.get('prompt_tokens'):>6}  prefilled={tm.get('prompt_n'):>6}")
    return tm.get("prompt_n"), u.get("prompt_tokens")

SYS={"role":"system","content":"# Core\n"+("You are a meticulous engineer. "*60)+"\n# Mission\nLearn."}
def exch(i): return [{"role":"user","content":f"Turn {i}: analyze subsystem {i}. "+("detail "*180)},
                     {"role":"assistant","content":f"Subsystem {i} analysis: "+("finding "*180)}]
def suffix(pct): return {"role":"user","content":f"(Ambient context — continue.)\n# ⚠ Context running low\nYour context is ~{pct}% full; remember now."}

hist=[]
for i in range(1,6): hist+=exch(i)
print("Build history to ~5 exchanges, then:")
print("\n[1] cold (first call):")
call([SYS,*hist,suffix(70)],"cold")
print("\n[2] SAME history, SUFFIX CHANGED (70->71%): should prefill ONLY the suffix:")
call([SYS,*hist,suffix(71)],"suffix changed only")
print("\n[3] history GREW by one exchange, suffix changed (72%): prefill ~ new exch + suffix:")
hist+=exch(6)
call([SYS,*hist,suffix(72)],"history grew + suffix changed")
print("\n[4] grew again (73%):")
hist+=exch(7)
call([SYS,*hist,suffix(73)],"history grew + suffix changed")
