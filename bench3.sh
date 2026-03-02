#!/bin/bash

SLOWDOWN_DURATION=${SLOWDOWN_DURATION:-600}  # seconds before stopping slow_down (10 minutes)
LOG_FILE="benchmark_$(date +%Y%m%d_%H%M%S).log"

# --- start_slow_down ---
echo "[$(date)] Starting slow_down..."
echo "will send slow_down request to DECODE_HEAD_NODE($DECODE_HEAD_NODE)"
curl -H "Content-Type: application/json" \
    -d "{\"forward_sleep_time\": 180}" \
    -X POST "http://$DECODE_HEAD_NODE:30003/slow_down"
echo "slow_down request sent successfully"

# --- benchmark ---
echo "[$(date)] Launching benchmark in background, logging to $LOG_FILE..."
(
    echo "start benchmark in docker"
    export ROUTER_NODE=$PREFILL_HEAD_NODE
    echo "make sure you have launched router on the ROUTER_NODE($ROUTER_NODE)"

    DATASET_DIR="${WORKSPACE}/../datasets"
    DATASET_FILE="${DATASET_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"

    if [[ ! -f "$DATASET_FILE" ]]; then
        echo "Dataset file not found at $DATASET_FILE, downloading..."
        mkdir -p "$DATASET_DIR"
        wget -O "$DATASET_FILE" \
            "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
        echo "Dataset downloaded successfully to $DATASET_FILE"
    else
        echo "Using existing dataset file at $DATASET_FILE"
    fi

    python3 -m sglang.bench_one_batch_server \
        --dataset-path "$DATASET_FILE" \
        --model-path deepseek-ai/DeepSeek-R1-0528 \
        --base-url http://$ROUTER_NODE:8000 \
        --batch-size 3200 \
        --input-len 1000 \
        --output-len 1000 \
        --skip-warmup
) > "$LOG_FILE" 2>&1 &
BENCHMARK_PID=$!
echo "[$(date)] Benchmark running with PID $BENCHMARK_PID"

# --- wait, then stop_slow_down ---
echo "[$(date)] Waiting ${SLOWDOWN_DURATION}s before stopping slow_down..."
sleep $SLOWDOWN_DURATION

echo "[$(date)] Stopping slow_down while benchmark is still running..."
echo "will send slow_down request to DECODE_HEAD_NODE($DECODE_HEAD_NODE)"
curl -H "Content-Type: application/json" \
    -d "{\"forward_sleep_time\": null}" \
    -X POST "http://$DECODE_HEAD_NODE:30003/slow_down"
echo "slow_down request sent successfully"

# --- wait for benchmark ---
echo "[$(date)] Waiting for benchmark (PID $BENCHMARK_PID) to finish..."
wait $BENCHMARK_PID
BENCHMARK_EXIT=$?

echo "[$(date)] Benchmark finished with exit code $BENCHMARK_EXIT"
