#!/bin/bash
set -euo pipefail

n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
model_path=$5
model_name=$6
MODEL_PATH="${model_path}/${model_name}"
LOG_PATH=$7

BENCH_INPUT_LEN=${8:-900}
BENCH_OUTPUT_LEN=${9:-400}
BATCH_SIZE=${10:-512}
REQ_RATE=${11:-inf}
RANDOM_RANGE_RATIO=${12:-1}
NUM_PROMPTS_MULTIPLIER=${13:-10}

SLOWDOWN_DURATION=${SLOWDOWN_DURATION:-60}
ROUTER_NODE=${ROUTER_NODE:-localhost}
MAX_RUNNING_REQUESTS_THRESHOLD=${MAX_RUNNING_REQUESTS_THRESHOLD:-0}
BENCH_AUTO_CLAMP=${BENCH_AUTO_CLAMP:-true}
PREFILL_HEAD_NODE=${PREFILL_HEAD_NODE:-localhost}
PREFILL_IDLE_TIMEOUT=${PREFILL_IDLE_TIMEOUT:-1200}
PREFILL_IDLE_REQUEST_TIMEOUT=${PREFILL_IDLE_REQUEST_TIMEOUT:-30}
PREFILL_IDLE_POLL_INTERVAL=${PREFILL_IDLE_POLL_INTERVAL:-5}
SLOWDOWN_ACTIVE=0

if [[ "${MAX_RUNNING_REQUESTS_THRESHOLD}" =~ ^[0-9]+$ ]] && [[ "${MAX_RUNNING_REQUESTS_THRESHOLD}" -gt 0 ]] && [[ "${BATCH_SIZE}" -gt "${MAX_RUNNING_REQUESTS_THRESHOLD}" ]]; then
    if [[ "${BENCH_AUTO_CLAMP}" == "true" ]]; then
        echo "Warning: benchmark batch_size=${BATCH_SIZE} exceeds max_running_requests=${MAX_RUNNING_REQUESTS_THRESHOLD}; clamping."
        BATCH_SIZE="${MAX_RUNNING_REQUESTS_THRESHOLD}"
    else
        echo "Error: benchmark batch_size=${BATCH_SIZE} exceeds max_running_requests=${MAX_RUNNING_REQUESTS_THRESHOLD}" >&2
        exit 1
    fi
fi

mkdir -p "${LOG_PATH}"
echo "${BATCH_SIZE}" > "${LOG_PATH}/effective_batch_size.txt"
echo "Config input=${BENCH_INPUT_LEN} output=${BENCH_OUTPUT_LEN}; Concurrency=${BATCH_SIZE}; ReqRate=${REQ_RATE}"

release_slowdown() {
    if [[ "${SLOWDOWN_ACTIVE}" -eq 1 ]]; then
        echo "[$(date)] Releasing slow_down..."
        curl -fsS -H "Content-Type: application/json" \
            -d "{\"forward_sleep_time\": null}" \
            -X POST "http://${DECODE_HEAD_NODE}:8000/slow_down" || true
        SLOWDOWN_ACTIVE=0
    fi
}

trap release_slowdown EXIT INT TERM

sleep 60
echo "[$(date)] Starting slow_down..."
curl -fsS -H "Content-Type: application/json" \
    -d "{\"forward_sleep_time\": 90}" \
    -X POST "http://${DECODE_HEAD_NODE}:8000/slow_down"
echo "slow_down request sent successfully"
SLOWDOWN_ACTIVE=1

(
    DATASET_DIR="/mnt/nfs/minchsun/dataset"
    DATASET_FILE="${DATASET_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"

    if [[ ! -f "${DATASET_FILE}" ]]; then
        echo "Dataset file not found at ${DATASET_FILE}, downloading..."
        mkdir -p "${DATASET_DIR}"
        wget -O "${DATASET_FILE}" \
            "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    else
        echo "Using existing dataset file at ${DATASET_FILE}"
    fi

    if ! python3 -m pip show tabulate >/dev/null 2>&1; then
        python3 -m pip install tabulate
    fi

    set -x
    python3 -m sglang.bench_one_batch_server \
        --dataset-path "${DATASET_FILE}" \
        --model-path "${MODEL_PATH}" \
        --base-url "http://${ROUTER_NODE}:30000" \
        --batch-size "${BATCH_SIZE}" \
        --input-len "${BENCH_INPUT_LEN}" \
        --output-len "${BENCH_OUTPUT_LEN}" \
        --skip-warmup
) &
BENCHMARK_PID=$!
echo "[$(date)] Benchmark running with PID ${BENCHMARK_PID}"

echo "[$(date)] Waiting ${SLOWDOWN_DURATION}s before stopping slow_down..."
sleep "${SLOWDOWN_DURATION}"
if ! python3 "${SGL_WS_PATH}/utils/wait_for_prefill_idle.py" \
    --prefill_url "http://${PREFILL_HEAD_NODE}:8000" \
    --timeout "${PREFILL_IDLE_TIMEOUT}" \
    --request-timeout "${PREFILL_IDLE_REQUEST_TIMEOUT}" \
    --poll-interval "${PREFILL_IDLE_POLL_INTERVAL}"; then
    echo "Warning: prefill idle check failed or timed out; continuing to release slow_down."
fi

echo "[$(date)] Stopping slow_down..."
release_slowdown
echo "slow_down request sent successfully"

echo "[$(date)] Waiting for benchmark (PID ${BENCHMARK_PID}) to finish..."
wait "${BENCHMARK_PID}"
BENCHMARK_EXIT=$?

echo "[$(date)] Benchmark finished with exit code ${BENCHMARK_EXIT}"
exit "${BENCHMARK_EXIT}"
