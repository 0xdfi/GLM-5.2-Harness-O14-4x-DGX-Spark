#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cache-busted C1/C4 prose-decode probe using vLLM server counters.

The probe refuses to start unless idle and marks counter contamination. It does
not infer throughput from HTTP wall time alone. Set O14_BASE_URL explicitly for
a remote deployment; the default is local-only.
"""
from __future__ import annotations

import concurrent.futures as futures
import json
import os
import random
import string
import sys
import time
import urllib.request

BASE = os.environ.get("O14_BASE_URL", "http://127.0.0.1:8211").rstrip("/")
CONCURRENCY = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
LABEL = sys.argv[3] if len(sys.argv) > 3 else "unlabeled"
TOPICS = [
    "the history and craft of traditional wooden boat building",
    "how mountain weather systems form and evolve over a single day",
    "the daily life of a lighthouse keeper in the 1890s",
    "the ecology of an old-growth forest floor through the seasons",
]
COUNTERS = (
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:num_requests_running",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def metrics() -> dict[str, float]:
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as response:
        text = response.read().decode()
    result: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key in COUNTERS:
            if line.startswith(key):
                result[key] = result.get(key, 0.0) + float(line.rsplit(" ", 1)[-1])
    missing = sorted(set(COUNTERS) - result.keys())
    if missing:
        raise RuntimeError(f"missing required metrics: {missing}")
    return result


def request_one(index: int) -> dict[str, float | int]:
    nonce = "".join(random.choices(string.ascii_lowercase + string.digits, k=24))
    body = {
        "model": os.environ.get("O14_MODEL", "glm-5.2"),
        "stream": False,
        "max_tokens": MAX_TOKENS,
        "temperature": 1.0,
        "top_p": 0.95,
        "messages": [{
            "role": "user",
            "content": (
                f"[session {nonce}] Write a long, detailed essay about "
                f"{TOPICS[index % len(TOPICS)]}. Prose only, no lists."
            ),
        }],
        "chat_template_kwargs": {"thinking": False},
    }
    request = urllib.request.Request(
        BASE + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=1800) as response:
        data = json.load(response)
    elapsed = time.time() - started
    usage = data.get("usage", {})
    completion = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    return {
        "index": index,
        "seconds": round(elapsed, 2),
        "completion_tokens": completion,
        "prompt_tokens": prompt_tokens,
        "client_tok_s": round(completion / elapsed, 2),
    }


def main() -> int:
    before = metrics()
    if before["vllm:num_requests_running"] > 0:
        print(json.dumps({"label": LABEL, "refused": "not idle"}))
        return 4
    started = time.time()
    with futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = list(pool.map(request_one, range(CONCURRENCY)))
    wall = time.time() - started
    time.sleep(2)
    after = metrics()
    delta = {key: after[key] - before[key] for key in COUNTERS}
    generated = delta["vllm:generation_tokens_total"]
    successes = delta["vllm:request_success_total"]
    drafts = delta["vllm:spec_decode_num_drafts_total"]
    drafted = delta["vllm:spec_decode_num_draft_tokens_total"]
    accepted = delta["vllm:spec_decode_num_accepted_tokens_total"]
    own_tokens = sum(int(item["completion_tokens"] or 0) for item in results)
    record = {
        "label": LABEL,
        "concurrency": CONCURRENCY,
        "max_tokens": MAX_TOKENS,
        "server_generation_tokens": generated,
        "own_completion_tokens": own_tokens,
        "contaminated": successes != CONCURRENCY or generated > own_tokens + 8,
        "requests_completed_delta": successes,
        "aggregate_tok_s_wall": round(generated / wall, 2),
        "aggregate_tok_s_stream_sum": round(
            sum(int(item["completion_tokens"] or 0) / float(item["seconds"]) for item in results),
            2,
        ),
        "drafted_per_step": round(drafted / drafts, 3) if drafts else None,
        "accepted_per_step": round(accepted / drafts, 3) if drafts else None,
        "steps_per_s": round(drafts / wall, 2),
        "per_request": results,
    }
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
