#!/usr/bin/env python3
"""Small reproducible latency/throughput sweep for the local vLLM server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def post_json(url: str, payload: dict, timeout: int = 900) -> tuple[dict, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    return body, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18300")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", default="results/benchmarks.jsonl")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    endpoint = args.base_url.rstrip("/") + "/v1/completions"
    rows: list[dict] = []

    coherence, elapsed = post_json(
        endpoint,
        {
            "model": "qwen3.8-flash-next",
            "prompt": "The capital of France is",
            "max_tokens": 12,
            "temperature": 0,
        },
    )
    rows.append(
        {
            "case": "coherence",
            "elapsed_s": elapsed,
            "text": coherence["choices"][0]["text"],
            "usage": coherence.get("usage", {}),
        }
    )

    for requested_tokens in (512, 4_000, 8_000, 16_000):
        for repeat in range(args.repeats):
            body, elapsed = post_json(
                endpoint,
                {
                    "model": "qwen3.8-flash-next",
                    "prompt": "word " * requested_tokens,
                    "max_tokens": 1,
                    "temperature": 0,
                },
            )
            actual = body["usage"]["prompt_tokens"]
            rows.append(
                {
                    "case": "prefill",
                    "requested_tokens": requested_tokens,
                    "repeat": repeat,
                    "prompt_tokens": actual,
                    "elapsed_s": elapsed,
                    "tokens_per_s": actual / elapsed,
                }
            )

    for repeat in range(args.repeats):
        body, elapsed = post_json(
            endpoint,
            {
                "model": "qwen3.8-flash-next",
                "prompt": "Write a detailed technical explanation of speculative decoding.",
                "max_tokens": 256,
                "temperature": 0,
                "ignore_eos": True,
            },
        )
        actual = body["usage"]["completion_tokens"]
        rows.append(
            {
                "case": "decode",
                "repeat": repeat,
                "completion_tokens": actual,
                "elapsed_s": elapsed,
                "tokens_per_s": actual / elapsed,
            }
        )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "base_url": args.base_url,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
