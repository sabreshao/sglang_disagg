#!/bin/bash
set -x

n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
model_path=$5
model_name=$6
MODEL_PATH="${model_path}/${model_name}"
log_path=$7

chosen_isl=${8:-1024}
chosen_osl=${9:-1024}
concurrency_list=${10:-"512x1"}
chosen_req_rate=${11:-1}
random_range_ratio=${12:-0.8}
num_prompts_multiplier=${13:-10}

echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[0]}; ${chosen_req_rate}"

head_node="localhost"
head_port="30000"


SLOWDOWN_DURATION=${SLOWDOWN_DURATION:-60}  # seconds before stopping slow_down (10 minutes)
ROUTER_NODE=${ROUTER_NODE:-localhost}

# --- start_slow_down ---
echo "[$(date)] Starting slow_down..."
echo "will send slow_down request to DECODE_HEAD_NODE($DECODE_HEAD_NODE)"
curl -H "Content-Type: application/json" \
    -d "{\"forward_sleep_time\": 180}" \
    -X POST "http://$DECODE_HEAD_NODE:8000/slow_down"
echo "slow_down request sent successfully"

# --- benchmark ---
echo "[$(date)] Launching benchmark in background, output to console (captured by slurm)..."
(
    echo "start benchmark in docker"
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
        --model-path $MODEL_PATH \
        --base-url http://$ROUTER_NODE:30000 \
        --batch-size 3200 \
        --input-len 1000 \
        --output-len 1000 \
        --skip-warmup
) &
BENCHMARK_PID=$!
echo "[$(date)] Benchmark running with PID $BENCHMARK_PID"

# --- wait, then stop_slow_down ---
echo "[$(date)] Waiting ${SLOWDOWN_DURATION}s before stopping slow_down..."
sleep $SLOWDOWN_DURATION

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
