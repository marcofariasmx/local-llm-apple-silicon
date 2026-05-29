#!/usr/bin/env python3
"""Demo of Qwen3.6-35B-A3B tool-calling against a local llama-server.
Defines 3 real tools, lets the model call them in a loop, prints everything.

Usage:
  python3 tool-test.py "what files are in my home dir and what time is it?"

Config via env:
  LLAMA_BASE   default http://127.0.0.1:8080
  LLAMA_MODEL  default Qwen3.6-35B-A3B-Q4_K_S.gguf
"""
import json, sys, urllib.request, datetime, os

BASE = os.environ.get("LLAMA_BASE", "http://127.0.0.1:8080")
API = BASE.rstrip("/") + "/v1/chat/completions"
MODEL = os.environ.get("LLAMA_MODEL", "Qwen3.6-35B-A3B-Q4_K_S.gguf")

# --- the actual tools the model is allowed to call -------------------------
def get_time(_=None):
    return datetime.datetime.now().strftime("%A %Y-%m-%d %H:%M:%S")

def list_dir(path="."):
    path = os.path.expanduser(path)
    return "\n".join(sorted(os.listdir(path))[:50]) or "(empty)"

def calculate(expression):
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "error: only basic arithmetic allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"error: {e}"

TOOLS_IMPL = {"get_time": get_time, "list_dir": list_dir, "calculate": calculate}

TOOLS_SPEC = [
    {"type": "function", "function": {"name": "get_time",
        "description": "Get the current local date and time", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "list_dir",
        "description": "List files in a directory",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Directory path, defaults to current dir"}}}}},
    {"type": "function", "function": {"name": "calculate",
        "description": "Evaluate a basic arithmetic expression",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]

def call_api(messages):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "tools": TOOLS_SPEC, "tool_choice": "auto",
                       "max_tokens": 2000, "temperature": 0.7}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]

def main():
    user = sys.argv[1] if len(sys.argv) > 1 else \
        "What time is it, what files are in ~/llm, and what is 1234 * 5678?"
    print(f"\n\033[1mUSER:\033[0m {user}\n")
    messages = [{"role": "user", "content": user}]
    for step in range(6):  # allow up to 6 tool rounds
        msg = call_api(messages)
        messages.append(msg)
        calls = msg.get("tool_calls")
        if calls:
            for c in calls:
                name = c["function"]["name"]
                args = json.loads(c["function"]["arguments"] or "{}")
                print(f"\033[33m  -> tool call:\033[0m {name}({args})")
                result = TOOLS_IMPL.get(name, lambda **k: "unknown tool")(**args)
                print(f"\033[36m     result:\033[0m {result.splitlines()[0] if result else ''}"
                      + (" ..." if result.count(chr(10)) else ""))
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": str(result)})
            continue  # let the model see the tool results
        print(f"\n\033[1;32mASSISTANT:\033[0m {msg.get('content','')}\n")
        return
    print("(stopped after max tool rounds)")

if __name__ == "__main__":
    main()
