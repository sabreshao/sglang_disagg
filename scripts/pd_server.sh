#!/bin/bash
set -euo pipefail

NODE0_ADDR="${NODE0_ADDR:-localhost}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_NAME="${MODEL_NAME:-}"
SLURM_JOB_ID="${SLURM_JOB_ID:-0}"
SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-}"
SGL_WS_PATH="${SGL_WS_PATH:-/sglang_disagg_simple}"
HEADNODE_PORT="${HEADNODE_PORT:-20000}"

xP="${xP:-1}"
yD="${yD:-1}"
IPADDRS="${IPADDRS:-localhost}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP:-false}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-false}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
DECODE_ENABLE_EP="${DECODE_ENABLE_EP:-false}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-false}"
DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"

BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-1}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-512}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_AUTO_CLAMP="${BENCH_AUTO_CLAMP:-true}"
LOAD_DUMMY="${LOAD_DUMMY:-0}"
DRY_RUN="${DRY_RUN:-0}"

source "${SGL_WS_PATH}/scripts/set_env_vars.sh"

sync_workspace_sglang_checkout() {
    local runtime_sglang="/sgl-workspace/sglang"
    local workspace_sglang="/workspace_sglang"

    if [[ ! -d "${workspace_sglang}" ]]; then
        echo "Using image-bundled sglang at ${runtime_sglang}"
        return
    fi

    mkdir -p "/sgl-workspace"
    rm -rf "${runtime_sglang}"
    ln -s "${workspace_sglang}" "${runtime_sglang}"
    echo "Linked workspace sglang checkout: ${runtime_sglang} -> ${workspace_sglang}"
}

sync_workspace_sglang_checkout

PYTHONPATH_PARTS=()
if [[ -d "/sgl-workspace/sglang/python" ]]; then
    PYTHONPATH_PARTS+=("/sgl-workspace/sglang/python")
fi
if [[ -d "/sgl-workspace/aiter" ]]; then
    PYTHONPATH_PARTS+=("/sgl-workspace/aiter")
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
    PYTHONPATH_PARTS+=("${PYTHONPATH}")
fi
export PYTHONPATH
PYTHONPATH=$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")
export SGLANG_WS_PATH="${SGLANG_WS_PATH:-${SGL_WS_PATH}}"

host_ip="${SGLANG_HOST_IP}"
host_name=$(hostname)

if [[ -z "${MORI_RDMA_TC:-}" ]]; then
    echo "ERROR: MORI_RDMA_TC is not set." >&2
    exit 1
fi

trim_spaces() {
    echo "$*" | xargs
}

model_supported() {
    case "$1" in
        DeepSeek-V3.2|DeepSeek-R1) return 0 ;;
        *) return 1 ;;
    esac
}

if ! model_supported "${MODEL_NAME}"; then
    echo "Unsupported MODEL_NAME inside container: ${MODEL_NAME}" >&2
    exit 1
fi

build_model_base_config() {
    case "$1" in
        DeepSeek-V3.2)
            echo "--decode-log-interval 1 --watchdog-timeout 3600 --load-balance-method round_robin --attention-backend nsa --nsa-prefill-backend tilelang --nsa-decode-backend tilelang --disaggregation-transfer-backend mori"
            ;;
        DeepSeek-R1)
            echo "--decode-log-interval 1 --watchdog-timeout 3600 --ep-dispatch-algorithm static --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --disaggregation-transfer-backend mori"
            ;;
    esac
}

build_model_mtp_config() {
    local model_name="$1"
    local mtp_size="$2"
    if [[ "${mtp_size}" -le 0 ]]; then
        echo ""
        return
    fi

    case "${model_name}" in
        DeepSeek-V3.2)
            echo "--speculative-algorithm EAGLE --speculative-num-steps ${mtp_size} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((mtp_size + 1))"
            ;;
        DeepSeek-R1)
            echo "--speculative-algorithm NEXTN --speculative-num-steps ${mtp_size} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((mtp_size + 1))"
            ;;
    esac
}

build_model_dp_config() {
    case "$1" in
        DeepSeek-V3.2|DeepSeek-R1)
            echo "--moe-a2a-backend mori --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head"
            ;;
    esac
}

if [[ "${PREFILL_ENABLE_DP}" == "true" ]]; then
    prefill_cuda_graph_bs="$(seq -s ' ' 1 3)"
    prefill_max_running_requests=128
    prefill_chunked_prefill_size=$((MORI_MAX_DISPATCH_TOKENS_PREFILL * PREFILL_TP_SIZE))
else
    prefill_cuda_graph_bs="$(seq -s ' ' 1 128)"
    prefill_max_running_requests=128
    prefill_chunked_prefill_size=262144
fi

if [[ "${DECODE_ENABLE_DP}" == "true" ]]; then
    decode_cuda_graph_bs="$(seq -s ' ' 1 132)"
    decode_max_running_requests=4096
    decode_chunked_prefill_size=$((MORI_MAX_DISPATCH_TOKENS_DECODE * DECODE_TP_SIZE))
else
    decode_cuda_graph_bs="$(seq -s ' ' 1 32)"
    decode_max_running_requests=256
    decode_chunked_prefill_size=262144
fi

if [[ "${LOAD_DUMMY}" == "1" ]]; then
    LOAD_DUMMY_MODEL="--load-format dummy"
else
    LOAD_DUMMY_MODEL=""
fi

REQUESTED_BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY}"
if [[ "${REQUESTED_BENCH_MAX_CONCURRENCY}" == *x* ]]; then
    REQUESTED_BENCH_MAX_CONCURRENCY="${REQUESTED_BENCH_MAX_CONCURRENCY%%x*}"
    echo "Warning: BENCH_MAX_CONCURRENCY list detected, only the first value will be used: ${REQUESTED_BENCH_MAX_CONCURRENCY}"
fi

if ! [[ "${REQUESTED_BENCH_MAX_CONCURRENCY}" =~ ^[0-9]+$ ]]; then
    echo "Invalid BENCH_MAX_CONCURRENCY: ${BENCH_MAX_CONCURRENCY}" >&2
    exit 1
fi

EFFECTIVE_BENCH_MAX_CONCURRENCY="${REQUESTED_BENCH_MAX_CONCURRENCY}"
if [[ "${REQUESTED_BENCH_MAX_CONCURRENCY}" -gt "${decode_max_running_requests}" ]]; then
    if [[ "${BENCH_AUTO_CLAMP}" == "true" ]]; then
        echo "Warning: BENCH_MAX_CONCURRENCY=${REQUESTED_BENCH_MAX_CONCURRENCY} exceeds decode max-running-requests=${decode_max_running_requests}. Clamping benchmark concurrency."
        EFFECTIVE_BENCH_MAX_CONCURRENCY="${decode_max_running_requests}"
    else
        echo "BENCH_MAX_CONCURRENCY=${REQUESTED_BENCH_MAX_CONCURRENCY} exceeds decode max-running-requests=${decode_max_running_requests}" >&2
        exit 1
    fi
fi
export EFFECTIVE_BENCH_MAX_CONCURRENCY

build_model_prefill_config() {
    case "$1" in
        DeepSeek-V3.2)
            echo "--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size} --cuda-graph-bs ${prefill_cuda_graph_bs} --disable-radix-cache"
            ;;
        DeepSeek-R1)
            echo "--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size}  --disable-radix-cache"
            ;;
    esac
}

build_model_decode_config() {
    case "$1" in
        DeepSeek-V3.2)
            echo "--mem-fraction-static 0.85 --max-running-requests ${decode_max_running_requests} --chunked-prefill-size ${decode_chunked_prefill_size} --cuda-graph-bs ${decode_cuda_graph_bs} --prefill-round-robin-balance"
            ;;
        DeepSeek-R1)
            echo "--mem-fraction-static 0.85 --max-running-requests ${decode_max_running_requests} --chunked-prefill-size ${decode_chunked_prefill_size}  --prefill-round-robin-balance"
            ;;
    esac
}

build_server_config() {
    local mode="$1"
    local model_name="$2"
    local tp_size="$3"
    local enable_ep="$4"
    local enable_dp="$5"
    local decode_mtp_size="$6"

    local parallel_args="--tp-size ${tp_size}"
    if [[ "${enable_ep}" == "true" ]]; then
        parallel_args="${parallel_args} --ep-size ${tp_size}"
    fi
    if [[ "${enable_dp}" == "true" ]]; then
        parallel_args="${parallel_args} --dp-size ${tp_size}"
    fi

    local base_config
    base_config="$(build_model_base_config "${model_name}")"
    local mtp_config
    mtp_config="$(build_model_mtp_config "${model_name}" "${decode_mtp_size}")"
    local dp_config=""
    if [[ "${enable_dp}" == "true" ]]; then
        dp_config="$(build_model_dp_config "${model_name}")"
    fi

    local specific_config=""
    if [[ "${mode}" == "prefill" ]]; then
        specific_config="$(build_model_prefill_config "${model_name}")"
    else
        specific_config="$(build_model_decode_config "${model_name}")"
    fi

    trim_spaces "${parallel_args}" "${base_config}" "${mtp_config}" "${dp_config}" "${specific_config}"
}

print_launch_details() {
    local role="$1"
    local dispatch_tokens="$2"
    local launch_cmd="$3"

    echo "LAUNCH DETAIL =================================="
    echo "Role                : ${role}"
    echo "Host                : ${host_name}:${host_ip}"
    echo "Node Rank           : ${NODE_RANK}"
    echo "Model               : ${MODEL_DIR}/${MODEL_NAME}"
    echo "Topology            : xP=${xP}, yD=${yD}, IPADDRS=${IPADDRS}"
    echo "Parallelism         : PREFILL_TP=${PREFILL_TP_SIZE}, PREFILL_EP=${PREFILL_ENABLE_EP}, PREFILL_DP=${PREFILL_ENABLE_DP}, DECODE_TP=${DECODE_TP_SIZE}, DECODE_EP=${DECODE_ENABLE_EP}, DECODE_DP=${DECODE_ENABLE_DP}, DECODE_MTP=${DECODE_MTP_SIZE}"
    echo "Benchmark           : requested_concurrency=${BENCH_MAX_CONCURRENCY}, effective_concurrency=${EFFECTIVE_BENCH_MAX_CONCURRENCY}, request_rate=${BENCH_REQUEST_RATE}"
    echo "Network             : IBDEVICES=${IBDEVICES}, GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}, NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}, SGLANG_HOST_IP=${SGLANG_HOST_IP}"
    echo "MORI                : DISPATCH_TOKENS=${dispatch_tokens}, MORI_SHMEM_MODE=${MORI_SHMEM_MODE:-unset}, MORI_RDMA_TC=${MORI_RDMA_TC:-unset}, MORI_RDMA_SL=${MORI_RDMA_SL:-unset}"
    echo "Python Path         : ${PYTHONPATH:-unset}"
    echo "Workspace           : SGL_WS_PATH=${SGL_WS_PATH}, runtime_sglang=$(readlink -f /sgl-workspace/sglang 2>/dev/null || echo /sgl-workspace/sglang)"
    echo "Command             : ${launch_cmd}"
    echo "================================================"
}

PREFILL_SERVER_CONFIG=$(build_server_config "prefill" "${MODEL_NAME}" "${PREFILL_TP_SIZE}" "${PREFILL_ENABLE_EP}" "${PREFILL_ENABLE_DP}" "${DECODE_MTP_SIZE}")
DECODE_SERVER_CONFIG=$(build_server_config "decode" "${MODEL_NAME}" "${DECODE_TP_SIZE}" "${DECODE_ENABLE_EP}" "${DECODE_ENABLE_DP}" "${DECODE_MTP_SIZE}")

IFS=',' read -ra IP_ARRAY <<< "${IPADDRS}"
PREFILL_NODES_PER_WORKER=$(((PREFILL_TP_SIZE + 7) / 8))
DECODE_NODES_PER_WORKER=$(((DECODE_TP_SIZE + 7) / 8))
NODE_OFFSET=$((PREFILL_NODES_PER_WORKER * xP))

PREFILL_HEADNODE_URLS=()
PREFILL_ARGS=""
for i in $(seq 0 $((xP - 1))); do
    prefill_idx=$((i * PREFILL_NODES_PER_WORKER))
    PREFILL_HEADNODE_URLS[$i]="${IP_ARRAY[$prefill_idx]}:${HEADNODE_PORT}"
    PREFILL_ARGS="${PREFILL_ARGS} --prefill http://${IP_ARRAY[$prefill_idx]}:8000"
done

DECODE_HEADNODE_URLS=()
DECODE_ARGS=""
for i in $(seq 0 $((yD - 1))); do
    decode_idx=$((i * DECODE_NODES_PER_WORKER + NODE_OFFSET))
    DECODE_HEADNODE_URLS[$i]="${IP_ARRAY[$decode_idx]}:${HEADNODE_PORT}"
    DECODE_ARGS="${DECODE_ARGS} --decode http://${IP_ARRAY[$decode_idx]}:8000"
done

echo "Waiting at the container creation barrier on ${host_name}"
python "${SGL_WS_PATH}/utils/socket_barrier.py" \
    --local-ip "${host_ip}" \
    --local-port 5000 \
    --enable-port \
    --node-ips "${IPADDRS}" \
    --node-ports 5000 \
    --wait-for-all-ports \
    --timeout 300

if [[ "${NODE_RANK}" -eq 0 ]]; then
    PREFILL_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_PREFILL} python3 -m sglang.launch_server --model-path ${MODEL_DIR}/${MODEL_NAME} ${LOAD_DUMMY_MODEL} --disaggregation-mode prefill --disaggregation-ib-device ${IBDEVICES} --host 0.0.0.0 --port 8000 --trust-remote-code ${PREFILL_SERVER_CONFIG}"
    if [[ "${PREFILL_NODES_PER_WORKER}" -gt 1 ]]; then
        PREFILL_CMD="${PREFILL_CMD} --dist-init-addr ${PREFILL_HEADNODE_URLS[0]} --nnodes ${PREFILL_NODES_PER_WORKER} --node-rank 0"
    fi
    print_launch_details "prefill-head" "${MORI_MAX_DISPATCH_TOKENS_PREFILL}" "${PREFILL_CMD}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "DRY RUN: ${PREFILL_CMD}"
    else
        eval "${PREFILL_CMD}" 2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log &
        prefill_pid=$!
    fi

    python "${SGL_WS_PATH}/utils/socket_barrier.py" \
        --node-ips "${IPADDRS}" \
        --node-ports 8000 \
        --wait-for-all-ports \
        --timeout 1800

    ROUTER_CMD="python -m sglang_router.launch_router --pd-disaggregation --mini-lb --port 30000 --policy random --prefill-policy random --decode-policy random ${PREFILL_ARGS} ${DECODE_ARGS}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "DRY RUN: ${ROUTER_CMD}"
    else
        eval "${ROUTER_CMD}" 2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log &
        proxy_pid=$!
        python "${SGL_WS_PATH}/utils/socket_barrier.py" \
            --node-ips "${NODE0_ADDR}" \
            --node-ports 30000 \
            --wait-for-all-health \
            --health-endpoint /health \
            --timeout 1800
    fi

    echo "Router is ready for benchmarking"
    export IS_MTP=false
    if [[ "${DECODE_MTP_SIZE}" -gt 0 ]]; then
        export IS_MTP=true
    fi
    export DECODE_HEAD_NODE="${IP_ARRAY[$NODE_OFFSET]}"
    export PREFILL_HEAD_NODE="${IP_ARRAY[0]}"
    export MAX_RUNNING_REQUESTS_THRESHOLD="${decode_max_running_requests}"

    BENCH_CMD="bash ${SGL_WS_PATH}/scripts/bench_throughput_with_slow_down.sh ${xP} ${yD} $((PREFILL_TP_SIZE * xP)) $((DECODE_TP_SIZE * yD)) ${MODEL_DIR} ${MODEL_NAME} /run_logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} ${BENCH_OUTPUT_LEN} ${EFFECTIVE_BENCH_MAX_CONCURRENCY} ${BENCH_REQUEST_RATE} ${BENCH_RANDOM_RANGE_RATIO} ${BENCH_NUM_PROMPTS_MULTIPLIER}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "DRY RUN: ${BENCH_CMD}"
    else
        cd "${SGL_WS_PATH}"
        eval "${BENCH_CMD}"
    fi

    mkdir -p "${SGL_WS_PATH}/logs/slurm_job-${SLURM_JOB_ID}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        cp /run_logs/slurm_job-${SLURM_JOB_ID}/* "${SGL_WS_PATH}/logs/slurm_job-${SLURM_JOB_ID}/" || true
        kill "${proxy_pid}" || true
        kill "${prefill_pid}" || true
    fi

elif [[ "${NODE_RANK}" -gt 0 && "${NODE_RANK}" -lt "${NODE_OFFSET}" ]]; then
    PREFILL_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_PREFILL} python3 -m sglang.launch_server --model-path ${MODEL_DIR}/${MODEL_NAME} ${LOAD_DUMMY_MODEL} --disaggregation-mode prefill --disaggregation-ib-device ${IBDEVICES} --host 0.0.0.0 --port 8000 --trust-remote-code ${PREFILL_SERVER_CONFIG}"
    if [[ "${PREFILL_NODES_PER_WORKER}" -gt 1 ]]; then
        rank=$((NODE_RANK % PREFILL_NODES_PER_WORKER))
        PREFILL_CMD="${PREFILL_CMD} --dist-init-addr ${PREFILL_HEADNODE_URLS[0]} --nnodes ${PREFILL_NODES_PER_WORKER} --node-rank ${rank}"
    fi
    print_launch_details "prefill-worker" "${MORI_MAX_DISPATCH_TOKENS_PREFILL}" "${PREFILL_CMD}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "DRY RUN: ${PREFILL_CMD}"
    else
        eval "${PREFILL_CMD}" 2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log &
        prefill_pid=$!
        python "${SGL_WS_PATH}/utils/socket_barrier.py" \
            --node-ips "${NODE0_ADDR}" \
            --node-ports 30000 \
            --wait-for-all-ports \
            --timeout 1800
        python "${SGL_WS_PATH}/utils/socket_wait.py" \
            --remote-ip "${NODE0_ADDR}" \
            --remote-port 30000
        kill "${prefill_pid}" || true
    fi

else
    RANK=$((NODE_RANK - xP * PREFILL_NODES_PER_WORKER))
    DECODE_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_DECODE} python3 -m sglang.launch_server --model-path ${MODEL_DIR}/${MODEL_NAME} ${LOAD_DUMMY_MODEL} --disaggregation-mode decode --disaggregation-ib-device ${IBDEVICES} --host 0.0.0.0 --port 8000 --trust-remote-code ${DECODE_SERVER_CONFIG}"
    if [[ "${DECODE_NODES_PER_WORKER}" -gt 1 ]]; then
        rank=$((RANK % DECODE_NODES_PER_WORKER))
        decode_idx=$((RANK / DECODE_NODES_PER_WORKER))
        DECODE_CMD="${DECODE_CMD} --dist-init-addr ${DECODE_HEADNODE_URLS[$decode_idx]} --nnodes ${DECODE_NODES_PER_WORKER} --node-rank ${rank}"
    fi
    print_launch_details "decode" "${MORI_MAX_DISPATCH_TOKENS_DECODE}" "${DECODE_CMD}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "DRY RUN: ${DECODE_CMD}"
    else
        eval "${DECODE_CMD}" 2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/decode_NODE${NODE_RANK}.log &
        decode_pid=$!
        python "${SGL_WS_PATH}/utils/socket_barrier.py" \
            --node-ips "${NODE0_ADDR}" \
            --node-ports 30000 \
            --wait-for-all-ports \
            --timeout 1800
        python "${SGL_WS_PATH}/utils/socket_wait.py" \
            --remote-ip "${NODE0_ADDR}" \
            --remote-port 30000
        kill "${decode_pid}" || true
    fi

    if [[ "${RANK}" -eq 0 ]]; then
        mkdir -p "${SGL_WS_PATH}/logs/slurm_job-${SLURM_JOB_ID}"
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            cp /run_logs/slurm_job-${SLURM_JOB_ID}/* "${SGL_WS_PATH}/logs/slurm_job-${SLURM_JOB_ID}/" || true
        fi

        if result=$(python "${SGL_WS_PATH}/utils/parse_decode_log.py" "/run_logs/slurm_job-${SLURM_JOB_ID}/decode_NODE${NODE_RANK}.log" "${BENCH_OUTPUT_LEN}" "${EFFECTIVE_BENCH_MAX_CONCURRENCY}"); then
            tpot=$(echo "${result}" | sed -n '1p')
            output_throughput=$(echo "${result}" | sed -n '2p')
            echo "Batch size = ${EFFECTIVE_BENCH_MAX_CONCURRENCY}"
            echo "Input len  = ${BENCH_INPUT_LEN}"
            echo "Output len = ${BENCH_OUTPUT_LEN}"
            echo "TPOT       = ${tpot} ms"
            echo "Output throughput = ${output_throughput} tokens/s"
        else
            echo "Decode log metrics are invalid; see stderr above."
        fi
    fi
fi

echo "Script completed successfully"
