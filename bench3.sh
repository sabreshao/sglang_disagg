#!/bin/bash
set -x

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


SLOWDOWN_DURATION=${SLOWDOWN_DURATION:-600}  # seconds before stopping slow_down (10 minutes)
ROUTER_NODE=${ROUTER_NODE:-localhost}

# --- start_slow_down ---
sleep 90  # sleep to let server warm up finish
echo "[$(date)] Starting slow_down..."
echo "will send slow_down request to DECODE_HEAD_NODE($DECODE_HEAD_NODE)"
curl -H "Content-Type: application/json" \
    -d "{\"forward_sleep_time\": 90}" \
    -X POST "http://$DECODE_HEAD_NODE:8000/slow_down"
echo "slow_down request sent successfully"

# --- benchmark ---
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

    #sed -i 's/dp_size = server_info\.get("dp_size", None) or 1/dp_size = internal_state[0].get("dp_size", None) or 1/' /sgl-workspace/sglang/python/sglang/test/bench_one_batch_server_internal.py
    #sed -i 's| + "/get_server_info"|.replace(":30000", ":8000") + "/get_server_info"|g' /sgl-workspace/sglang/python/sglang/test/bench_one_batch_server_internal.py

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
