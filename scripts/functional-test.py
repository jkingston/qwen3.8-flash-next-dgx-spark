#!/usr/bin/env python3
"""Exercise OpenAI-compatible tools, vision, and repeated text requests."""

from __future__ import annotations

import argparse
import base64
import json
import struct
import time
import urllib.request
import zlib
from pathlib import Path


def post(base: str, payload: dict, timeout: int = 300) -> tuple[dict, float]:
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response), time.perf_counter() - started


def red_png_data_url(width: int = 64, height: int = 64) -> str:
    raw = b"".join(b"\x00" + bytes((255, 0, 0)) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18300")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", default="results/functional.jsonl")
    parser.add_argument("--stability-requests", type=int, default=20)
    args = parser.parse_args()

    common = {"model": "qwen3.8-flash-next", "temperature": 0}
    tool_body, tool_seconds = post(
        args.base_url,
        {
            **common,
            "messages": [{"role": "user", "content": "What is the weather in London? Use the weather tool."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
            "tool_choice": "auto",
            "max_tokens": 128,
        },
    )
    tool_message = tool_body["choices"][0]["message"]

    vision_body, vision_seconds = post(
        args.base_url,
        {
            **common,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": red_png_data_url()}},
                    {"type": "text", "text": "What is the dominant color? Answer with one color word."},
                ],
            }],
            "max_tokens": 256,
        },
    )

    latencies = []
    errors = []
    for index in range(args.stability_requests):
        try:
            body, elapsed = post(
                args.base_url,
                {
                    **common,
                    "messages": [{"role": "user", "content": f"Reply with exactly: pong-{index}"}],
                    "max_tokens": 32,
                },
            )
            latencies.append(elapsed)
            if not body.get("choices"):
                errors.append({"index": index, "error": "missing choices"})
        except Exception as exc:  # preserve the whole run for diagnosis
            errors.append({"index": index, "error": repr(exc)})

    result = {
        "label": args.label,
        "tool": {
            "elapsed_s": tool_seconds,
            "tool_calls": tool_message.get("tool_calls"),
            "content": tool_message.get("content"),
        },
        "vision": {
            "elapsed_s": vision_seconds,
            "message": vision_body["choices"][0]["message"],
        },
        "stability": {
            "requests": args.stability_requests,
            "successes": args.stability_requests - len(errors),
            "errors": errors,
            "min_s": min(latencies) if latencies else None,
            "mean_s": sum(latencies) / len(latencies) if latencies else None,
            "max_s": max(latencies) if latencies else None,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
