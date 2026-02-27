#!/bin/bash

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

IFS='x' read -r -a chosen_concurrencies <<< "$concurrency_list"

echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[0]}; ${chosen_req_rate}"

head_node="localhost"
head_port="30000"



# Simple script to send a prompt to an OpenAI-compatible LLM server

# Configuration
HOST=${head_node}
PORT=${head_port}
PROMPT="上联：莺莺燕燕翠翠红红处处融融洽洽。请出下联。"
MODEL=${model_name}

# API endpoint
API_URL="http://${HOST}:${PORT}/v1/chat/completions"

echo "Sending request to: ${API_URL}"
echo "Prompt: ${PROMPT}"
echo "Model: ${MODEL}"
echo "---"

# Send the request
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

sleep 60

cd /sgl-workspace/sglang && python3 benchmark/gsm8k/bench_sglang.py --num-questions 1319 --host http://${HOST} --port ${PORT}

sleep 60

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
