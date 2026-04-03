#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROFILE="${PROFILE:-deepseek_v32_default}"
DEFAULT_DOCKER_IMAGE="${DEFAULT_DOCKER_IMAGE:-rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260331}"
case "${PROFILE}" in
    deepseek_v32_default)
        default_model_name="DeepSeek-V3.2"
        default_docker_image="${DEFAULT_DOCKER_IMAGE}"
        default_xp=1
        default_yd=1
        default_prefill_tp=8
        default_prefill_ep=true
        default_prefill_dp=false
        default_decode_tp=8
        default_decode_ep=true
        default_decode_dp=false
        default_decode_mtp=0
        default_bench_input=102400
        default_bench_output=1024
        default_bench_concurrency=128
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
        default_bench_input=102400
        default_bench_output=1024
        default_bench_concurrency=128
        ;;
    deepseek_r1_default)
        default_model_name="DeepSeek-R1"
        default_docker_image="${DEFAULT_DOCKER_IMAGE}"
        default_xp=1
        default_yd=1
        default_prefill_tp=8
        default_prefill_ep=true
        default_prefill_dp=false
        default_decode_tp=8
        default_decode_ep=true
        default_decode_dp=false
        default_decode_mtp=0
        default_bench_input=102400
        default_bench_output=1024
        default_bench_concurrency=128
        ;;
    deepseek_v32_fp4_default)
        default_model_name="DeepSeek-V3.2-mxfp4"
        default_docker_image="${DEFAULT_DOCKER_IMAGE}"
        default_xp=1
        default_yd=1
        default_prefill_tp=8
        default_prefill_ep=true
        default_prefill_dp=false
        default_decode_tp=8
        default_decode_ep=true
        default_decode_dp=false
        default_decode_mtp=0
        default_bench_input=61440
        default_bench_output=1024
        default_bench_concurrency=8
        ;;
    *)
        echo "Unsupported PROFILE: ${PROFILE}" >&2
        echo "Supported profiles: deepseek_v32_default, deepseek_v32_recommended, deepseek_r1_default, deepseek_v32_fp4_default" >&2
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
export BENCH_MODE="${BENCH_MODE:-slowdown}"
export BENCH_BACKEND="${BENCH_BACKEND:-sglang}"
export BENCH_BURSTINESS="${BENCH_BURSTINESS:-1.0}"
export BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-}"
export BENCH_NUM_WARMUPS="${BENCH_NUM_WARMUPS:-}"
export BENCH_RESULT_DIR="${BENCH_RESULT_DIR:-}"
export BENCH_RESULT_FILENAME="${BENCH_RESULT_FILENAME:-}"
export BENCH_POISSON_USE_SLOWDOWN_PHASE_CONTROL="${BENCH_POISSON_USE_SLOWDOWN_PHASE_CONTROL:-false}"
export SLOWDOWN_DURATION="${SLOWDOWN_DURATION:-60}"
export PREFILL_IDLE_TIMEOUT="${PREFILL_IDLE_TIMEOUT:-1200}"
export PREFILL_IDLE_REQUEST_TIMEOUT="${PREFILL_IDLE_REQUEST_TIMEOUT:-30}"
export PREFILL_IDLE_POLL_INTERVAL="${PREFILL_IDLE_POLL_INTERVAL:-5}"

export LOAD_DUMMY="${LOAD_DUMMY:-1}"
export DRY_RUN="${DRY_RUN:-0}"
export RESTART_CONTAINER_BEFORE_SERVER="${RESTART_CONTAINER_BEFORE_SERVER:-1}"
# By default the simple suite uses the image-bundled /sgl-workspace/sglang.
# Set WORKSPACE_SGLANG_DIR explicitly only when you want to override it.
export WORKSPACE_SGLANG_DIR="${WORKSPACE_SGLANG_DIR:-}"

case "${BENCH_MODE}" in
    slowdown|poisson) ;;
    *)
        echo "Unsupported BENCH_MODE: ${BENCH_MODE}" >&2
        echo "Supported BENCH_MODE values: slowdown, poisson" >&2
        exit 1
        ;;
esac

bash "${SCRIPT_DIR}/scripts/run_pd.slurm" 2>&1 | tee "${SCRIPT_DIR}/log_${MODEL_NAME}_xP${xP}_yD${yD}.log"
