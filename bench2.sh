#!/bin/bash
# =============================================================================
# bench2.sh - Accuracy / Functional Benchmark
# =============================================================================
# This script tests model correctness (not throughput) by:
#   1. Sending a single chat completion request and printing the response
#   2. Running a GSM8K math benchmark (1319 questions) via SGLang's built-in tool
#   3. Running a concurrency sweep using benchmark_lib.sh for latency/throughput
#
# It is NOT called automatically by sglang_disagg_server.sh — run it manually
# after the servers are up to validate accuracy on a live deployment.
#
# Positional arguments:
#   $1  n_prefill              - Number of prefill worker groups (xP)
#   $2  n_decode               - Number of decode worker groups (yD)
#   $3  prefill_gpus           - Total prefill GPUs
#   $4  decode_gpus            - Total decode GPUs
#   $5  model_path             - Base model directory
#   $6  model_name             - Model subdirectory name
#   $7  log_path               - Directory for benchmark result JSON files
#   $8  chosen_isl             - Input sequence length (default: 1024)
#   $9  chosen_osl             - Output sequence length (default: 1024)
#   $10 concurrency_list       - Concurrency levels, e.g. "512x1" (x-separated)
#   $11 chosen_req_rate        - Request rate (default: 1)
#   $12 random_range_ratio     - Sequence length variance (default: 0.8)
#   $13 num_prompts_multiplier - Total prompts = max_concurrency * this (default: 10)
# =============================================================================

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

# Parse the x-separated concurrency list into an array
IFS='x' read -r -a chosen_concurrencies <<< "$concurrency_list"

echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[0]}; ${chosen_req_rate}"

head_node="localhost"
head_port="30000"


# =============================================================================
# Functional Smoke Test: Single Chat Completion
# =============================================================================
# Send one request to verify the server is responding correctly before
# running the full benchmark suite.

HOST=${head_node}
PORT=${head_port}
PROMPT="上联：莺莺燕燕翠翠红红处处融融洽洽。请出下联。"
MODEL=${model_name}

API_URL="http://${HOST}:${PORT}/v1/chat/completions"

echo "Sending request to: ${API_URL}"
echo "Prompt: ${PROMPT}"
echo "Model: ${MODEL}"
echo "---"

curl -X POST "${API_URL}" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"${MODEL}"'",
        "messages": [
            {
                "role": "user",
                "content": "'"${PROMPT}"'"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }' | jq '.'

echo ""
echo "Request completed."

# Brief pause before running the GSM8K accuracy benchmark
sleep 60

# =============================================================================
# GSM8K Accuracy Benchmark
# =============================================================================
# Runs SGLang's built-in GSM8K benchmark (1319 math word problems).
# Results are printed to stdout.
cd /sgl-workspace/sglang && python3 benchmark/gsm8k/bench_sglang.py --num-questions 1319 --host http://${HOST} --port ${PORT}

sleep 60

# =============================================================================
# Concurrency Sweep (benchmark_lib.sh)
# =============================================================================
profile_folder="${log_path}/sglang_isl_${chosen_isl}_osl_${chosen_osl}"
mkdir -p $profile_folder

source "$(dirname "$0")/benchmark_lib.sh"

for max_concurrency in ${chosen_concurrencies[@]}; do

    export_file="${profile_folder}/concurrency_${max_concurrency}_req_rate_${chosen_req_rate}_gpus_$((prefill_gpus+decode_gpus))_ctx_${prefill_gpus}_gen_${decode_gpus}"

    echo "profile_folder: $profile_folder"
    echo "max_concurrency: $max_concurrency"
    echo "chosen_req_rate: $chosen_req_rate"
    echo "MODEL_PATH: $MODEL_PATH"
    echo "head_port: $head_port"
    echo "chosen_isl: $chosen_isl"
    echo "chosen_osl: $chosen_osl"
    echo "export_file: $export_file"

    echo "-----------------------------------------"
done
