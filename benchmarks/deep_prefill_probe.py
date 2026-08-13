#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cache-busted deep cold-prefill gate used by the final O14 battery."""
from __future__ import annotations

import json
import os
import random
import string
import sys
import time
import urllib.request

BASE = os.environ.get("O14_BASE_URL", "http://127.0.0.1:8211").rstrip("/")
LABEL = sys.argv[1] if len(sys.argv) > 1 else "prefill"
NONCE = "".join(random.choices(string.ascii_lowercase, k=16))
WORDS = ("the mountain weather shifted through " + NONCE + " valley meadows and ") * 11000
BODY = {
    "model": os.environ.get("O14_MODEL", "glm-5.2"),
    "stream": False,
    "max_tokens": 32,
    "messages": [{"role": "user", "content": WORDS + " Summarize in one sentence."}],
    "chat_template_kwargs": {"thinking": False},
}
REQUEST = urllib.request.Request(
    BASE + "/v1/chat/completions",
    json.dumps(BODY).encode(),
    {"Content-Type": "application/json"},
)
started = time.time()
try:
    with urllib.request.urlopen(REQUEST, timeout=1800) as response:
        data = json.load(response)
    elapsed = time.time() - started
    tokens = int(data["usage"]["prompt_tokens"])
    record = {
        "label": LABEL,
        "deep_cold_prefill_gate": "PASS",
        "prompt_tokens": tokens,
        "seconds": round(elapsed, 1),
        "prefill_tok_s": round(tokens / elapsed, 1),
    }
    status = 0
except Exception as error:  # evidence tool must emit a machine-readable failure
    record = {
        "label": LABEL,
        "deep_cold_prefill_gate": "FAIL",
        "error": str(error)[:250],
    }
    status = 1
print(json.dumps(record, sort_keys=True))
raise SystemExit(status)
