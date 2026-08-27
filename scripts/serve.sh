#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on a single DGX Spark / GB10 with the PLE table mmapped
# from disk. OpenAI-compatible API on $PORT.
#
#   scripts/serve.sh                 # defaults: ctx 32768, MTP off
#   MTP=2 CTX=65536 scripts/serve.sh # speculative decoding + longer context
#   docker logs -f qwen38-flash      # wait for "Application startup complete"
#
# Tunables (env):
#   PORT=18083        host port for the dgxspark unified API
#   CTX=131072        max context length per lane
#   SEQS=8            maximum concurrent sequences
#   GPU_MEM=0.85      candidate eight-lane memory target
#   MTP=0             speculative tokens (2-3 = the model's MTP head; ~1.6x decode)
#   PREWARM=0         1 = stream the 48 GiB table once at boot to warm the page cache
#   IMAGE=qwen38-flash-dgx
#   MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4
set -euo pipefail

NAME="${NAME:-qwen38-flash-next}"
IMAGE="${IMAGE:-qwen38-flash-dgx:82ed48d}"
MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
REVISION="${REVISION:-7b719225242aacd3dbd3f9407468c2ee9a9d2594}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PORT="${PORT:-18083}"
CTX="${CTX:-131072}"
SEQS="${SEQS:-8}"
GPU_MEM="${GPU_MEM:-0.85}"
MTP="${MTP:-0}"
PREWARM="${PREWARM:-0}"

# Resolve the local snapshot directory and map it to the in-container mount.
REPO_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"
SNAP_HOST="$REPO_DIR/snapshots/$REVISION"
if [ ! -d "$SNAP_HOST" ]; then
  echo "!! checkpoint not found under $REPO_DIR"
  echo "   run scripts/download-weights.sh first."
  exit 1
fi
SNAP_IN="/hf/hub/models--${MODEL//\//--}/snapshots/$REVISION"

# The PLE gather is a CPU op + a pageable host->device copy: it MUST run outside
# CUDA graphs. We declare it a splitting op and use PIECEWISE capture (never FULL*).
# --enforce-eager also works but is slower.
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
CC="${CC:--cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=$SPLIT}"

SPEC=""
[ "$MTP" != 0 ] && SPEC="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
# shellcheck disable=SC2086
docker run -d --name "$NAME" --gpus all --ipc=host --shm-size 16g -p "${PORT}:8000" \
  -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS="${WORKERS:-32}" -e VLLM_PLE_MMAP_PREWARM="$PREWARM" \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  "$IMAGE" \
  "$SNAP_IN" --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port 8000 --load-format safetensors \
    --max-model-len "$CTX" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GPU_MEM" \
    --no-enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 \
    $CC \
    --no-enable-flashinfer-autotune \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    $SPEC

echo ">> $NAME starting on :$PORT (model 'qwen3.8-flash-next')"
echo ">> first boot loads ~76 GiB of weights (~8 min). Follow:  docker logs -f $NAME"
echo ">> ready when the log says 'Application startup complete'. Then: scripts/smoke-test.sh"
