#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Predictable code-class peak-decode probe for the deep-MTP regime."""
from __future__ import annotations

import concurrent.futures as futures
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("O14_BASE_URL", "http://127.0.0.1:8211").rstrip("/")
CONCURRENCY = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LABEL = sys.argv[2] if len(sys.argv) > 2 else "peak"
PROMPT = (
    "Write a Python module with 40 small functions named func_001 through "
    "func_040. Each function must follow the exact same pattern: "
    "def func_NNN(x): return x + NNN. No comments or docstrings."
)


def counters() -> tuple[float, float, float]:
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as response:
        text = response.read().decode()

    def total(name: str) -> float:
        prefix = "vllm:" + name
        values = [
            float(line.rsplit(" ", 1)[1])
            for line in text.splitlines()
            if line.startswith(prefix) and not line.startswith("#")
        ]
        if not values:
            raise RuntimeError(f"missing required metric: {prefix}")
        return sum(values)

    return (
        total("spec_decode_num_drafts_total"),
        total("spec_decode_num_accepted_tokens_total"),
        total("generation_tokens_total"),
    )


def request_one(index: int) -> tuple[int, float]:
    body = {
        "model": os.environ.get("O14_MODEL", "glm-5.2"),
        "stream": False,
        "max_tokens": 900,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": PROMPT + f" # variant {index}"}],
        "chat_template_kwargs": {"thinking": False},
    }
    request = urllib.request.Request(
        BASE + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=900) as response:
        data = json.load(response)
    return int(data["usage"]["completion_tokens"]), time.time() - started


def main() -> int:
    drafts0, accepted0, generated0 = counters()
    started = time.time()
    with futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = list(pool.map(request_one, range(CONCURRENCY)))
    wall = time.time() - started
    drafts1, accepted1, generated1 = counters()
    tokens = sum(item[0] for item in results)
    steps = drafts1 - drafts0
    print(json.dumps({
        "label": LABEL,
        "concurrency": CONCURRENCY,
        "total_tokens": tokens,
        "wall_seconds": round(wall, 2),
        "aggregate_tok_s": round(tokens / wall, 2),
        "accepted_per_step": round((accepted1 - accepted0) / max(steps, 1), 3),
        "steps": int(steps),
        "server_generation_delta": generated1 - generated0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
