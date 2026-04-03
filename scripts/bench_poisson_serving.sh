#!/bin/bash
set -euo pipefail

model_path="$1"
model_name="$2"
MODEL_PATH="${model_path}/${model_name}"
LOG_PATH="$3"
BENCH_INPUT_LEN="${4:-1024}"
BENCH_OUTPUT_LEN="${5:-256}"
MAX_CONCURRENCY="${6:-128}"
REQUEST_RATE="${7:-inf}"
RANDOM_RANGE_RATIO="${8:-1}"
NUM_PROMPTS_MULTIPLIER="${9:-10}"

BENCH_BACKEND="${BENCH_BACKEND:-sglang}"
BENCH_BURSTINESS="${BENCH_BURSTINESS:-1.0}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-}"
BENCH_NUM_WARMUPS="${BENCH_NUM_WARMUPS:-}"
BENCH_RESULT_DIR="${BENCH_RESULT_DIR:-${LOG_PATH}}"
BENCH_RESULT_FILENAME="${BENCH_RESULT_FILENAME:-poisson_${model_name}_$(date +%Y%m%d_%H%M%S).json}"
ROUTER_NODE="${ROUTER_NODE:-localhost}"
BENCH_POISSON_USE_SLOWDOWN_PHASE_CONTROL="${BENCH_POISSON_USE_SLOWDOWN_PHASE_CONTROL:-false}"
SLOWDOWN_DURATION="${SLOWDOWN_DURATION:-60}"
PREFILL_IDLE_TIMEOUT="${PREFILL_IDLE_TIMEOUT:-1200}"
PREFILL_IDLE_REQUEST_TIMEOUT="${PREFILL_IDLE_REQUEST_TIMEOUT:-30}"
PREFILL_IDLE_POLL_INTERVAL="${PREFILL_IDLE_POLL_INTERVAL:-5}"

if [[ -z "${BENCH_NUM_PROMPTS}" ]]; then
    BENCH_NUM_PROMPTS=$((MAX_CONCURRENCY * NUM_PROMPTS_MULTIPLIER))
fi

if [[ -z "${BENCH_NUM_WARMUPS}" ]]; then
    BENCH_NUM_WARMUPS=$((2 * MAX_CONCURRENCY))
fi

mkdir -p "${BENCH_RESULT_DIR}"

echo "Poisson benchmark configuration:"
echo "  model_path        : ${MODEL_PATH}"
echo "  backend           : ${BENCH_BACKEND}"
echo "  base_url          : http://${ROUTER_NODE}:30000"
echo "  input_len         : ${BENCH_INPUT_LEN}"
echo "  output_len        : ${BENCH_OUTPUT_LEN}"
echo "  random_range_ratio: ${RANDOM_RANGE_RATIO}"
echo "  num_prompts       : ${BENCH_NUM_PROMPTS}"
echo "  max_concurrency   : ${MAX_CONCURRENCY}"
echo "  request_rate      : ${REQUEST_RATE}"
echo "  burstiness        : ${BENCH_BURSTINESS}"
echo "  num_warmups       : ${BENCH_NUM_WARMUPS}"
echo "  result_file       : ${BENCH_RESULT_DIR}/${BENCH_RESULT_FILENAME}"

cmd=(
    python3 "${SGL_WS_PATH}/bench_serving/benchmark_serving.py"
    --model "${MODEL_PATH}"
    --backend "${BENCH_BACKEND}"
    --base-url "http://${ROUTER_NODE}:30000"
    --dataset-name random
    --random-input-len "${BENCH_INPUT_LEN}"
    --random-output-len "${BENCH_OUTPUT_LEN}"
    --random-range-ratio "${RANDOM_RANGE_RATIO}"
    --num-prompts "${BENCH_NUM_PROMPTS}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --request-rate "${REQUEST_RATE}"
    --burstiness "${BENCH_BURSTINESS}"
    --ignore-eos
    --save-result
    --num-warmups "${BENCH_NUM_WARMUPS}"
    --percentile-metrics ttft,tpot,itl,e2el
    --result-dir "${BENCH_RESULT_DIR}"
    --result-filename "${BENCH_RESULT_FILENAME}"
)

if [[ "${BENCH_BACKEND}" == "sglang" ]]; then
    cmd+=(--trust-remote-code --random-input-as-token-ids)
fi

if [[ "${BENCH_POISSON_USE_SLOWDOWN_PHASE_CONTROL}" == "true" ]]; then
    if [[ -z "${DECODE_HEAD_NODE:-}" || -z "${PREFILL_HEAD_NODE:-}" ]]; then
        echo "BENCH_POISSON_USE_SLOWDOWN_PHASE_CONTROL requires DECODE_HEAD_NODE and PREFILL_HEAD_NODE." >&2
        exit 1
    fi
    cmd+=(
        --phase-control-decode-slowdown
        --phase-control-decode-url "http://${DECODE_HEAD_NODE}:8000"
        --phase-control-prefill-url "http://${PREFILL_HEAD_NODE}:8000"
        --phase-control-slowdown-duration "${SLOWDOWN_DURATION}"
        --phase-control-prefill-idle-timeout "${PREFILL_IDLE_TIMEOUT}"
        --phase-control-prefill-idle-request-timeout "${PREFILL_IDLE_REQUEST_TIMEOUT}"
        --phase-control-prefill-idle-poll-interval "${PREFILL_IDLE_POLL_INTERVAL}"
    )
fi

set -x
"${cmd[@]}"
