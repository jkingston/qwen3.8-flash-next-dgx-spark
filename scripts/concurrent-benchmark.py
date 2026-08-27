#!/usr/bin/env python3
"""Benchmark a vLLM endpoint with several simultaneous agent-like requests."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def request(base_url: str, agent: int, prompt_tokens: int, output_tokens: int) -> dict:
    marker = f"Agent {agent}: retain this marker. "
    # `word` is one token for this tokenizer. Keep agent identity in the short
    # marker; numbered repeated strings can split into multiple tokens.
    prompt = marker + ("word " * prompt_tokens)
    payload = {
        "model": "qwen3.8-flash-next",
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - started
    usage = body["usage"]
    return {
        "agent": agent,
        "elapsed_s": elapsed,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "finish_reason": body["choices"][0]["finish_reason"],
        "text_prefix": body["choices"][0]["text"][:120],
    }


def run_case(base_url: str, concurrency: int, prompt_tokens: int, output_tokens: int) -> dict:
    started = time.perf_counter()
    rows = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(request, base_url, agent, prompt_tokens, output_tokens): agent
            for agent in range(concurrency)
        }
        for future in concurrent.futures.as_completed(futures):
            agent = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append({"agent": agent, "error": repr(exc)})
    wall = time.perf_counter() - started
    total_prompt = sum(row["prompt_tokens"] for row in rows)
    total_output = sum(row["completion_tokens"] for row in rows)
    latencies = [row["elapsed_s"] for row in rows]
    return {
        "concurrency": concurrency,
        "requested_prompt_tokens_each": prompt_tokens,
        "requested_output_tokens_each": output_tokens,
        "wall_s": wall,
        "successes": len(rows),
        "errors": errors,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_output,
        "aggregate_prompt_tokens_s": total_prompt / wall if wall else None,
        "aggregate_completion_tokens_s": total_output / wall if wall else None,
        "latency_min_s": min(latencies) if latencies else None,
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "latency_max_s": max(latencies) if latencies else None,
        "rows": sorted(rows, key=lambda row: row["agent"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18300")
    parser.add_argument("--label", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", default="results/concurrent-benchmarks.jsonl")
    parser.add_argument(
        "--cases",
        default="8000:256,64000:64,125000:32",
        help="Comma-separated prompt:output token pairs",
    )
    args = parser.parse_args()

    # Small warmup prevents first-shape JIT from contaminating all four requests.
    request(args.base_url, -1, 512, 8)
    cases = []
    for value in args.cases.split(","):
        prompt_tokens, output_tokens = (int(part) for part in value.split(":"))
        print(f">> {args.concurrency} x prompt={prompt_tokens}, output={output_tokens}", flush=True)
        result = run_case(
            args.base_url,
            args.concurrency,
            prompt_tokens,
            output_tokens,
        )
        cases.append(result)
        print(json.dumps(result, indent=2), flush=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "base_url": args.base_url,
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
