#!/bin/bash
# =============================================================================
# bench_throughput_with_slow_down.sh - Throughput Benchmark with Decode Slow-Down
# =============================================================================
# Primary benchmark script called by sglang_disagg_server.sh on NODE_RANK=0.
# It runs a concurrency-sweep benchmark against the router (port 30000) using
# the benchmark_serving tool from benchmark_lib.sh.
#
# Positional arguments (all required):
#   $1  n_prefill              - Number of prefill worker groups (xP)
#   $2  n_decode               - Number of decode worker groups (yD)
#   $3  prefill_gpus           - Total prefill GPUs (PREFILL_TP_SIZE * xP)
#   $4  decode_gpus            - Total decode GPUs (DECODE_TP_SIZE * yD)
#   $5  model_path             - Base model directory (MODEL_DIR inside Docker: /models)
#   $6  model_name             - Model subdirectory name
#   $7  log_path               - Directory to write benchmark result JSON files
#   $8  BENCH_INPUT_LEN        - Input sequence length (tokens)
#   $9  BENCH_OUTPUT_LEN       - Output sequence length (tokens)
#   $10 concurrency_list       - Concurrency levels, e.g. "1024" or "1024x512x128"
#   $11 req_rate               - Request rate (inf = send all at once)
#   $12 random_range_ratio     - Sequence length variance ratio (1 = fixed)
#   $13 num_prompts_multiplier - Total prompts = max_concurrency * this value
#
# Environment variables consumed:
#   IS_MTP         - "true" if speculative decoding (MTP) is active
#   DECODE_HEAD_NODE - IP of the first decode node (used for slow_down API)
#   SGL_WS_PATH    - Path to this repo inside Docker (/sglang_disagg)
# =============================================================================

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

echo "Config input=${BENCH_INPUT_LEN} output=${BENCH_OUTPUT_LEN}; Concurrency=${BATCH_SIZE}"

head_node="localhost"
head_port="30000"


SLOWDOWN_DURATION=${SLOWDOWN_DURATION:-60}  # seconds before stopping slow_down (10 minutes)
ROUTER_NODE=${ROUTER_NODE:-localhost}

# --- start_slow_down ---
# Send a slow_down request to the decode server to hold prefilled tokens in
# place while the benchmark fills its request queue. This simulates steady-state
# decode conditions before measurements begin.
sleep 90  # sleep to let server warm up finish
echo "[$(date)] Starting slow_down..."
echo "will send slow_down request to DECODE_HEAD_NODE($DECODE_HEAD_NODE)"
curl -H "Content-Type: application/json" \
    -d "{\"forward_sleep_time\": 90}" \
    -X POST "http://$DECODE_HEAD_NODE:8000/slow_down"
echo "slow_down request sent successfully"

# --- benchmark ---
# Launch the one-batch benchmark in the background while slow_down is active,
# so requests accumulate and are dispatched together when slow_down is released.
echo "[$(date)] Launching benchmark in background, output to console (captured by slurm)..."
(
    echo "start benchmark in docker"
    echo "make sure you have launched router on the ROUTER_NODE($ROUTER_NODE)"

    DATASET_DIR="/mnt/nfs/minchsun/dataset"
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

    set -x

    python3 -m sglang.bench_one_batch_server \
        --dataset-path "$DATASET_FILE" \
        --model-path $MODEL_PATH \
        --base-url http://$ROUTER_NODE:30000 \
        --batch-size $BATCH_SIZE \
        --input-len $BENCH_INPUT_LEN \
        --output-len $BENCH_OUTPUT_LEN \
        --skip-warmup
) &
BENCHMARK_PID=$!
echo "[$(date)] Benchmark running with PID $BENCHMARK_PID"

# --- wait, then stop_slow_down ---
# After the configured duration, wait for prefill to drain (no queued requests),
# then release the slow_down so decode proceeds at full speed.
echo "[$(date)] Waiting ${SLOWDOWN_DURATION}s before stopping slow_down..."
sleep $SLOWDOWN_DURATION
python $SGL_WS_PATH/utils/wait_for_prefill_idle.py --prefill_url http://${head_node}:8000

echo "[$(date)] Stopping slow_down while benchmark is still running..."
echo "will send slow_down request to DECODE_HEAD_NODE($DECODE_HEAD_NODE)"
curl -H "Content-Type: application/json" \
    -d "{\"forward_sleep_time\": null}" \
    -X POST "http://$DECODE_HEAD_NODE:8000/slow_down"
echo "slow_down request sent successfully"

# --- wait for benchmark ---
echo "[$(date)] Waiting for benchmark (PID $BENCHMARK_PID) to finish..."
wait $BENCHMARK_PID
BENCHMARK_EXIT=$?

echo "[$(date)] Benchmark finished with exit code $BENCHMARK_EXIT"
