#!/usr/bin/env bash
# Production foreground launcher for the dgxspark user systemd service.
set -euo pipefail

NAME="${NAME:-qwen38-flash-next}"
IMAGE="${IMAGE:-qwen38-flash-dgx:82ed48d}"
MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
REVISION="${REVISION:-7b719225242aacd3dbd3f9407468c2ee9a9d2594}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
VLLM_CACHE="${VLLM_CACHE:-$HOME/.cache/qwen3.8-flash-next}"
PORT="${PORT:-18083}"
CTX="${CTX:-131072}"
SEQS="${SEQS:-8}"
GPU_MEM="${GPU_MEM:-0.85}"
MTP="${MTP:-0}"
PREWARM="${PREWARM:-0}"

repo_dir="$HF_CACHE/hub/models--${MODEL//\//--}"
snapshot_host="$repo_dir/snapshots/$REVISION"
if [[ ! -d "$snapshot_host" ]]; then
  echo "checkpoint not found: $snapshot_host" >&2
  exit 1
fi
snapshot_container="/hf/hub/models--${MODEL//\//--}/snapshots/$REVISION"
mkdir -p "$VLLM_CACHE"

split_ops='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
spec=()
if [[ "$MTP" != 0 ]]; then
  spec=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run --rm --name "$NAME" --gpus all --ipc=host --shm-size 16g \
  -p "127.0.0.1:${PORT}:8000" \
  -v "$HF_CACHE:/hf" -v "$VLLM_CACHE:/root/.cache" \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS="${WORKERS:-32}" \
  -e VLLM_PLE_MMAP_PREWARM="$PREWARM" -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  "$IMAGE" "$snapshot_container" \
  --served-model-name qwen3.8-flash-next \
  --host 0.0.0.0 --port 8000 --load-format safetensors \
  --max-model-len "$CTX" --max-num-seqs "$SEQS" \
  --gpu-memory-utilization "$GPU_MEM" \
  --no-enable-prefix-caching --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  -cc.cudagraph_mode=PIECEWISE "-cc.splitting_ops=$split_ops" \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 "${spec[@]}"

