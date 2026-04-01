#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROFILE="${PROFILE:-deepseek_v32_default}"
DEFAULT_DOCKER_IMAGE="${DEFAULT_DOCKER_IMAGE:-rocm/sgl-dev:sglang-0.5.8-rocm700-mi35x-mori-0210}"

case "${PROFILE}" in
    deepseek_v32_default)
        default_model_name="DeepSeek-V3.2"
        default_docker_image="${DEFAULT_DOCKER_IMAGE}"
        default_xp=1
        default_yd=1
        default_prefill_tp=8
        default_prefill_ep=false
        default_prefill_dp=false
        default_decode_tp=8
        default_decode_ep=false
        default_decode_dp=false
        default_decode_mtp=0
        default_bench_input=3300
        default_bench_output=400
        default_bench_concurrency=256
        ;;
    deepseek_v32_recommended)
        default_model_name="DeepSeek-V3.2"
        default_docker_image="${DEFAULT_DOCKER_IMAGE}"
        default_xp=1
        default_yd=1
        default_prefill_tp=8
        default_prefill_ep=false
        default_prefill_dp=true
        default_decode_tp=8
        default_decode_ep=false
        default_decode_dp=true
        default_decode_mtp=0
        default_bench_input=3300
        default_bench_output=400
        default_bench_concurrency=512
        ;;
    deepseek_r1_default)
        default_model_name="DeepSeek-R1"
        default_docker_image="${DEFAULT_DOCKER_IMAGE}"
        default_xp=1
        default_yd=1
        default_prefill_tp=8
        default_prefill_ep=false
        default_prefill_dp=false
        default_decode_tp=8
        default_decode_ep=false
        default_decode_dp=false
        default_decode_mtp=0
        default_bench_input=3300
        default_bench_output=400
        default_bench_concurrency=1536
        ;;
    *)
        echo "Unsupported PROFILE: ${PROFILE}" >&2
        echo "Supported profiles: deepseek_v32_default, deepseek_v32_recommended, deepseek_r1_default" >&2
        exit 1
        ;;
esac

export MODEL_NAME="${MODEL_NAME:-${default_model_name}}"
export MODEL_DIR="${MODEL_DIR:-/mnt/nfs/RAID/shared/huggingface/hub/}"
export DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-${default_docker_image}}"

export xP="${xP:-${default_xp}}"
export yD="${yD:-${default_yd}}"
export NUM_NODES="${NUM_NODES:-}"

export PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-${default_prefill_tp}}"
export PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP:-${default_prefill_ep}}"
export PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-${default_prefill_dp}}"
export DECODE_TP_SIZE="${DECODE_TP_SIZE:-${default_decode_tp}}"
export DECODE_ENABLE_EP="${DECODE_ENABLE_EP:-${default_decode_ep}}"
export DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-${default_decode_dp}}"
export DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-${default_decode_mtp}}"

export BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-${default_bench_input}}"
export BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-${default_bench_output}}"
export BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-1}"
export BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
export BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-${default_bench_concurrency}}"
export BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
export BENCH_AUTO_CLAMP="${BENCH_AUTO_CLAMP:-true}"

export LOAD_DUMMY="${LOAD_DUMMY:-0}"
export DRY_RUN="${DRY_RUN:-0}"
# By default the simple suite uses the image-bundled /sgl-workspace/sglang.
# Set WORKSPACE_SGLANG_DIR explicitly only when you want to override it.
export WORKSPACE_SGLANG_DIR="${WORKSPACE_SGLANG_DIR:-}"

bash "${SCRIPT_DIR}/scripts/run_pd.slurm" 2>&1 | tee "${SCRIPT_DIR}/log_${MODEL_NAME}_xP${xP}_yD${yD}.log"
