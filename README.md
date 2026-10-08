# Qwen3.8-Flash-Next on DGX Spark

Production configuration promoted on 2026-10-07: **UltraFast W4A16 + MTP3**.
Public model ID remains `qwen3.8-flash-next`. Direct serving is loopback-only
on port 18300; the existing unified API provides authentication and routing.

## Selected configuration

| Setting | Production |
|---|---|
| Checkpoint | W4A16 AutoRound experts, FP8 side layers, INT8 lm_head, dense int4 MTP |
| Image | `qwen38-flash-dgx:iter6d-20260910`, pinned by local SHA256 in manifest |
| Engine | `v0.1.dev20073+g8e685d198` with UltraFast recipe patches |
| Loader | Standard `safetensors`, not tensorizer |
| Context / lanes | 262144 maximum per request / 4 schedulable requests |
| KV allocation | `22g` = 22,000,000,000 bytes; BF16 via `auto` |
| SSM | Image default FP32 |
| Speculation | MTP3, 65536-token draft vocabulary, block rejection / probabilistic draft |
| Prefix caching | Enabled; retention interval 6400 |
| Prefill | Chunked, 4096 batched tokens |
| PLE | Mapped FP8 table, prewarm and prefetch off |
| Graphs | PIECEWISE with the manifest's splitting operations |
| Parsers | `qwen3_xml` tools, `qwen3` reasoning |
| Agent thinking | Native `medium` in tested requests; not a server-enforced default |

The explicit KV allocation overrides `--gpu-memory-utilization 0.01`: this
is **not a 1% total-memory configuration**. Captured argv contains 8192 and then
4096 for batched tokens; the latter is effective. We preserve deployed argv.

Four simultaneous ~128K windows passed retrieval/cache tests. **Four full
262K windows are not qualified or guaranteed.** Keep the existing 12 GiB memory
guard enabled; temperature monitoring is observational, not an added cutoff.

## Reuse the deployed profile

[profiles/ultrafast-production-create.json](profiles/ultrafast-production-create.json)
contains the deployed Docker settings with portable host-path placeholders.
The launcher resolves these environment variables, defaulting to the current
user's home directory:

| Variable | Default |
|---|---|
| `MODEL_DIR` | `~/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-mtpdense-g32` |
| `PLE_DIR` | `~/models/ple-table-fp8` |
| `DRAFT_VOCAB_FILE` | `~/.cache/qwen38-v16b/draft-vocab-ids-K65536.txt` |

Override them for your storage layout; spaces and relative paths are supported.
Tildes are expanded by the launcher. Colons and newlines are rejected because
Docker bind-volume syntax cannot represent them safely here. No shell evaluation
is performed, and the launcher does not read `.env` files automatically.

```bash
export MODEL_DIR="/srv/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-mtpdense-g32"
export PLE_DIR="/srv/models/ple-table-fp8"
export DRAFT_VOCAB_FILE="/srv/models/draft-vocab-ids-K65536.txt"
python3 runtime/production.py --dry-run
```

The pinned local image is
`sha256:ba63307007a14b9c05185cdcdda07c7dc075551c0d96c029950fb8d27254e6a8`.

That SHA is a **local image ID, not a pullable registry digest**.
UltraFast source observed at promotion:
[dime-online/qwen3.8-Flash-DGX-UltraFast](https://github.com/dime-online/qwen3.8-Flash-DGX-UltraFast/tree/0c391a3e74b6a775cfe248691ca7fd855b1876a5).
The recipe revision alone does not establish full image/checkpoint build provenance.
The root Dockerfile and preparation scripts are the **historical NVFP4 build**,
not instructions to rebuild this UltraFast image or checkpoint.

Inspect without touching Docker:

```bash
python3 runtime/production.py --dry-run
```

Install the downloader dependency in your Python environment first:

```bash
python3 -m pip install huggingface_hub
```

On an idle machine with the pinned Docker image installed, create the container:

```bash
python3 runtime/production.py --create
```

Missing assets are now downloaded automatically. The production model is a
**derived checkpoint**, so the launcher downloads the base and runs the pinned
upstream dense-MTP g32 conversion and verification before publishing the model
directory. It also downloads the PLE table and installs the exact SHA256-checked
draft vocabulary. Sources:

- Base: `Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid`
  at `8b82f0b7abe3d1150a7827d298c75e86267636ae`.
- PLE: `Saren/Qwen3.8-Flash-Next-ple-table-fp8`
  at `50511b0a41aa1d34b8beb7e5d4bb06a0b650dc14`.
- Builder/vocabulary: the UltraFast recipe commit linked above, not mutable main.

Allow approximately 130 GB for public downloads plus 5 GB for the converted
shard and additional space for Docker. The base is kept beside `MODEL_DIR` in
`.qwen38-base-<revision>`; unchanged shards are hardlinked, not duplicated.
Requires Git, Bash, GNU coreutils and the existing pinned image. The conversion
helper is CPU-only, network-disabled and limited to 8 GiB. Source downloads
are resumable through Hugging Face's local metadata. Existing complete assets
need no network. Gated access, if required, uses your normal HF authentication;
model license terms still apply.

Safetensors headers/sizes, indexed shards and the conversion report are checked;
these are completeness checks, not a full rehash of every tensor at each launch.
The upstream builder additionally checks its pinned source index/extra-shard hashes
and verifies the conversion. Incomplete existing derived model directories and
unexpected draft vocabularies are preserved and rejected, not overwritten.
Failed build directories and fetched recipe checkouts remain for diagnosis.
Missing PLE shards can be resumed. `--no-download` disables network preparation;
`--dry-run` performs no downloads or Docker calls.

Or create and serve in the foreground with `bash runtime/run-vllm.sh`.
Both refuse to overwrite an existing container. **Do not run these on the live
Spark just to update documentation**: its production container already exists.
Its systemd service uses `docker start --attach qwen38-flash`, with
`docker stop --timeout 45 qwen38-flash` on shutdown and a 20-minute startup timeout.
Keep the existing service and memory guard; no service changes are needed.

## Evidence and limitations

Real usage through 2026-10-08 11:38 UTC: 82 token-accounted OpenCode requests
versus 260 on October 1. Median full-request latency 14.7 -> 8.7 seconds,
p90 55.4 -> 43.6 seconds, median effective output rate 25.3 -> 44.4 tok/s,
with median input 65K -> 117K. Effective rate includes waiting/prefill, not
pure decode. All 83 post-promotion gateway records were HTTP200; one lacked
token usage and is not proven complete. Thinking settings, concurrency and
tasks differ: observational evidence, not a causal A/B or quality result.

Qualification included four ~128K cached lanes without preemption and a
49-minute mixed Pi/OpenCode soak (21/24 tasks, no API errors or preemptions).
Generated-test hangs and correctness failures remain possible. Promotion
does not establish perfect agent reliability or full-context reasoning quality.

## Rollback and historical recipe

Spark retains stopped container `qwen38-base-rollback-20261007`.
The deployed idle-checked rollback script is
`scripts/rollback-retention6400.sh` under the model directory in the separate
`dgxspark` deployment checkout (not included in this repository).
Preserve the base container and checkpoints while evaluating production.

Previous documentation is in [docs/LEGACY-NVFP4.md](docs/LEGACY-NVFP4.md);
its launcher is `runtime/run-nvfp4-legacy.sh`. Its old ports, lane counts and
build instructions do **not** describe the selected production profile.
