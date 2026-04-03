#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/config.sh}"
RECIPE_CONTAINER_DIR_DEFAULT="/customer_recipe"
WORKSPACE_SGLANG_MOUNT="/workspace_sglang"
CONTAINER_LOG_ROOT="/run_logs"

fail() {
    echo "Error: $*" >&2
    exit 1
}

log() {
    echo "[$(date '+%F %T')] $*"
}

is_true() {
    [[ "$1" == "1" || "$1" == "true" || "$1" == "True" ]]
}

require_non_empty() {
    local name="$1"
    local value="$2"
    if [[ -z "${value}" ]]; then
        fail "Required variable ${name} is empty."
    fi
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
        fail "${name} must be a positive integer, got: ${value}"
    fi
}

nodes_per_worker() {
    local tp_size="$1"
    echo $(((tp_size + GPUS_PER_NODE - 1) / GPUS_PER_NODE))
}

join_array_with_colon() {
    local IFS=":"
    echo "$*"
}

cmd_to_string() {
    local out=""
    local arg
    for arg in "$@"; do
        printf -v out '%s%q ' "${out}" "${arg}"
    done
    echo "${out% }"
}

init_run_timestamp() {
    RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S_%3N')}"
    export RUN_TIMESTAMP
}

source_config() {
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        fail "Config file not found: ${CONFIG_FILE}"
    fi
    CONFIG_FILE=$(readlink -f "${CONFIG_FILE}")
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"

    if ! declare -p MODEL_BASE_ARGS >/dev/null 2>&1; then
        if declare -p COMMON_SERVER_ARGS >/dev/null 2>&1; then
            MODEL_BASE_ARGS=("${COMMON_SERVER_ARGS[@]}")
        else
            MODEL_BASE_ARGS=()
        fi
    fi
    if ! declare -p MODEL_DP_ARGS >/dev/null 2>&1; then
        MODEL_DP_ARGS=()
    fi

    if [[ "${BENCH_MODE}" != "slowdown" ]]; then
        fail "Only BENCH_MODE=slowdown is supported in customer_recipe."
    fi
    if [[ "${BENCH_POISSON_USE_SLOWDOWN_PHASE_CONTROL}" != "false" ]]; then
        fail "BENCH_POISSON_USE_SLOWDOWN_PHASE_CONTROL must stay false."
    fi
}

validate_config() {
    require_non_empty "MODEL_NAME" "${MODEL_NAME}"
    require_non_empty "MODEL_DIR" "${MODEL_DIR}"
    require_non_empty "DOCKER_IMAGE_NAME" "${DOCKER_IMAGE_NAME}"
    require_positive_integer "xP" "${xP}"
    require_positive_integer "yD" "${yD}"
    require_positive_integer "GPUS_PER_NODE" "${GPUS_PER_NODE}"
    require_positive_integer "PREFILL_TP_SIZE" "${PREFILL_TP_SIZE}"
    require_positive_integer "DECODE_TP_SIZE" "${DECODE_TP_SIZE}"
    require_positive_integer "BENCH_INPUT_LEN" "${BENCH_INPUT_LEN}"
    require_positive_integer "BENCH_OUTPUT_LEN" "${BENCH_OUTPUT_LEN}"
    require_positive_integer "BENCH_MAX_CONCURRENCY" "${BENCH_MAX_CONCURRENCY}"
    require_positive_integer "PREFILL_MAX_RUNNING_REQUESTS" "${PREFILL_MAX_RUNNING_REQUESTS}"
    require_positive_integer "DECODE_MAX_RUNNING_REQUESTS" "${DECODE_MAX_RUNNING_REQUESTS}"
    require_positive_integer "HEADNODE_PORT" "${HEADNODE_PORT}"

    if [[ "$(dirname "${CONFIG_FILE}")" != "${SCRIPT_DIR}" ]]; then
        fail "CONFIG_FILE must live in ${SCRIPT_DIR} so it can be mounted into the container."
    fi

    if [[ -n "${WORKSPACE_SGLANG_DIR}" ]] && [[ ! -d "${WORKSPACE_SGLANG_DIR}" ]]; then
        fail "WORKSPACE_SGLANG_DIR does not exist: ${WORKSPACE_SGLANG_DIR}"
    fi
}

derive_topology() {
    PREFILL_NODES_PER_WORKER=$(nodes_per_worker "${PREFILL_TP_SIZE}")
    DECODE_NODES_PER_WORKER=$(nodes_per_worker "${DECODE_TP_SIZE}")
    NODE_OFFSET=$((xP * PREFILL_NODES_PER_WORKER))
    EXPECTED_NUM_NODES=$((NODE_OFFSET + yD * DECODE_NODES_PER_WORKER))

    if [[ -z "${NUM_NODES:-}" ]]; then
        NUM_NODES="${EXPECTED_NUM_NODES}"
    elif [[ "${NUM_NODES}" -ne "${EXPECTED_NUM_NODES}" ]]; then
        fail "NUM_NODES=${NUM_NODES} does not match expected topology ${EXPECTED_NUM_NODES}."
    fi

    REQUESTED_BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY}"
    EFFECTIVE_BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY}"
    if (( REQUESTED_BENCH_MAX_CONCURRENCY > DECODE_MAX_RUNNING_REQUESTS )); then
        if is_true "${BENCH_AUTO_CLAMP}"; then
            log "Warning: BENCH_MAX_CONCURRENCY=${REQUESTED_BENCH_MAX_CONCURRENCY} exceeds decode max-running-requests=${DECODE_MAX_RUNNING_REQUESTS}; clamping."
            EFFECTIVE_BENCH_MAX_CONCURRENCY="${DECODE_MAX_RUNNING_REQUESTS}"
        else
            fail "BENCH_MAX_CONCURRENCY=${REQUESTED_BENCH_MAX_CONCURRENCY} exceeds decode max-running-requests=${DECODE_MAX_RUNNING_REQUESTS}"
        fi
    fi
}

detect_node_env() {
    local node_name
    local default_ifname
    local default_ibdevices
    local nd_prio
    local nd_dscp

    node_name=$(hostname)
    case "${node_name}" in
        GPU*|smci355-ccs-aus*)
            default_ibdevices="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
            default_ifname=$(ip route show default | awk 'NR == 1 {print $5}')
            ;;
        node*)
            default_ibdevices="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
            default_ifname="enp193s0f1np1"
            ;;
        mia1*)
            default_ibdevices="rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7"
            default_ifname=$(ip route show default | awk 'NR == 1 {print $5}')
            ;;
        *)
            fail "unable to infer network config from hostname: ${node_name}"
            ;;
    esac

    IBDEVICES="${IBDEVICES_OVERRIDE:-${default_ibdevices}}"
    GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME_OVERRIDE:-${default_ifname}}"
    NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME_OVERRIDE:-${default_ifname}}"
    SGLANG_HOST_IP="${SGLANG_HOST_IP_OVERRIDE:-$(ip -4 addr show dev "${GLOO_SOCKET_IFNAME}" | awk '/inet / {ip=$2; sub(/\/.*/, "", ip); if (first == "") first=ip} END {print first}')}"
    require_non_empty "SGLANG_HOST_IP" "${SGLANG_HOST_IP}"

    NCCL_IB_HCA="${NCCL_IB_HCA_OVERRIDE:-${IBDEVICES}}"

    if [[ -n "${MORI_RDMA_SL_OVERRIDE}" ]]; then
        MORI_RDMA_SL="${MORI_RDMA_SL_OVERRIDE}"
    else
        nd_prio=$(nicctl show qos 2>/dev/null | awk '/PFC no-drop priorities/ {value=$NF} END {print value}')
        require_non_empty "MORI_RDMA_SL" "${nd_prio}"
        MORI_RDMA_SL="${nd_prio}"
    fi

    if [[ -n "${MORI_RDMA_TC_OVERRIDE}" ]]; then
        MORI_RDMA_TC="${MORI_RDMA_TC_OVERRIDE}"
    else
        nd_dscp=$(nicctl show qos 2>/dev/null | awk -v p="${MORI_RDMA_SL}" '$1 == "DSCP" && $2 == ":" && $NF == p {value=$3} END {print value}')
        require_non_empty "MORI_RDMA_TC" "${nd_dscp}"
        MORI_RDMA_TC=$((4 * nd_dscp))
    fi
}

get_ip_for_node() {
    local node_name="$1"
    srun --input=none --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="${node_name}" bash -lc \
        "ip route get ${ROUTE_PROBE_TARGET} | awk '{for (i = 1; i <= NF; ++i) if (\$i == \"src\") {print \$(i + 1); exit}}'"
}

select_nodes_from_slurm() {
    require_non_empty "SLURM_JOB_NODELIST" "${SLURM_JOB_NODELIST:-}"

    FULL_NODELIST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    TOTAL_ALLOCATED=$(echo "${FULL_NODELIST}" | wc -l)
    if (( NUM_NODES > TOTAL_ALLOCATED )); then
        fail "Need ${NUM_NODES} nodes but only ${TOTAL_ALLOCATED} allocated."
    fi

    SELECTED_NODES=$(echo "${FULL_NODELIST}" | head -n "${NUM_NODES}")
    SELECTED_NODELIST_SRUN=$(echo "${SELECTED_NODES}" | paste -sd,)
    MASTER_NODE=$(echo "${SELECTED_NODES}" | head -n 1)

    log "Selected nodes:"
    echo "${SELECTED_NODES}"

    local model_path="${MODEL_DIR}/${MODEL_NAME}"
    log "Checking model availability on selected nodes..."
    srun --nodelist="${SELECTED_NODELIST_SRUN}" --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" /bin/bash -lc "
        if [[ -d '${model_path}' ]]; then
            echo \"\$(hostname): found ${model_path}\"
        else
            echo \"\$(hostname): missing ${model_path}\" >&2
            exit 1
        fi
    "

    NODE0_ADDR=$(get_ip_for_node "${MASTER_NODE}")
    require_non_empty "NODE0_ADDR" "${NODE0_ADDR}"

    IPS=()
    local selected_node_names=()
    mapfile -t selected_node_names <<< "${SELECTED_NODES}"
    for node_name in "${selected_node_names[@]}"; do
        local ip
        ip=$(get_ip_for_node "${node_name}")
        require_non_empty "IP for ${node_name}" "${ip}"
        echo "${node_name} ${ip}"
        IPS+=("${ip}")
    done
    IPADDRS=$(IFS=,; echo "${IPS[*]}")
}

prepare_cluster_urls() {
    local i
    local prefill_idx
    local decode_idx
    local total_ips

    IFS=',' read -r -a IP_ARRAY <<< "${IPADDRS}"
    total_ips=${#IP_ARRAY[@]}
    if (( total_ips < NUM_NODES )); then
        fail "Expected ${NUM_NODES} node IPs but collected ${total_ips}. IPADDRS=${IPADDRS}"
    fi

    PREFILL_HEADNODE_URLS=()
    DECODE_HEADNODE_URLS=()
    ROUTER_PREFILL_ARGS=()
    ROUTER_DECODE_ARGS=()

    for ((i = 0; i < xP; ++i)); do
        prefill_idx=$((i * PREFILL_NODES_PER_WORKER))
        PREFILL_HEADNODE_URLS+=("${IP_ARRAY[prefill_idx]}:${HEADNODE_PORT}")
        ROUTER_PREFILL_ARGS+=(--prefill "http://${IP_ARRAY[prefill_idx]}:8000")
    done

    for ((i = 0; i < yD; ++i)); do
        decode_idx=$((i * DECODE_NODES_PER_WORKER + NODE_OFFSET))
        DECODE_HEADNODE_URLS+=("${IP_ARRAY[decode_idx]}:${HEADNODE_PORT}")
        ROUTER_DECODE_ARGS+=(--decode "http://${IP_ARRAY[decode_idx]}:8000")
    done

    PREFILL_HEAD_NODE="${IP_ARRAY[0]}"
    DECODE_HEAD_NODE="${IP_ARRAY[NODE_OFFSET]}"
}

build_mtp_args() {
    MTP_ARGS=()
    if (( DECODE_MTP_SIZE <= 0 )); then
        return
    fi

    case "${MODEL_NAME}" in
        DeepSeek-V3.2|DeepSeek-V3.2-mxfp4)
            MTP_ARGS=(
                --speculative-algorithm EAGLE
                --speculative-num-steps "${DECODE_MTP_SIZE}"
                --speculative-eagle-topk 1
                --speculative-num-draft-tokens "$((DECODE_MTP_SIZE + 1))"
            )
            ;;
        DeepSeek-R1)
            MTP_ARGS=(
                --speculative-algorithm NEXTN
                --speculative-num-steps "${DECODE_MTP_SIZE}"
                --speculative-eagle-topk 1
                --speculative-num-draft-tokens "$((DECODE_MTP_SIZE + 1))"
            )
            ;;
        *)
            fail "Unsupported MODEL_NAME for MTP: ${MODEL_NAME}"
            ;;
    esac
}

build_server_command() {
    local role="$1"
    local tp_size="$2"
    local enable_ep="$3"
    local enable_dp="$4"
    local nodes_in_group="$5"
    local node_rank_in_group="$6"
    local dist_init_addr="$7"

    CURRENT_CMD=(
        python3 -m sglang.launch_server
        --model-path "${MODEL_DIR}/${MODEL_NAME}"
    )

    if is_true "${LOAD_DUMMY}"; then
        CURRENT_CMD+=(--load-format dummy)
    fi

    CURRENT_CMD+=(
        --disaggregation-mode "${role}"
        --disaggregation-ib-device "${IBDEVICES}"
        --host 0.0.0.0
        --port 8000
        --trust-remote-code
        --tp-size "${tp_size}"
    )

    if is_true "${enable_ep}"; then
        CURRENT_CMD+=(--ep-size "${tp_size}")
    fi

    if is_true "${enable_dp}"; then
        CURRENT_CMD+=(--dp-size "${tp_size}")
    fi

    build_mtp_args
    CURRENT_CMD+=("${MODEL_BASE_ARGS[@]}")
    CURRENT_CMD+=("${MTP_ARGS[@]}")
    if is_true "${enable_dp}"; then
        CURRENT_CMD+=("${MODEL_DP_ARGS[@]}")
    fi

    if [[ "${role}" == "prefill" ]]; then
        CURRENT_CMD+=("${PREFILL_ROLE_ARGS[@]}")
        CURRENT_CMD+=("${PREFILL_EXTRA_ARGS[@]}")
    else
        CURRENT_CMD+=("${DECODE_ROLE_ARGS[@]}")
        CURRENT_CMD+=("${DECODE_EXTRA_ARGS[@]}")
    fi

    if (( nodes_in_group > 1 )); then
        CURRENT_CMD+=(
            --dist-init-addr "${dist_init_addr}"
            --nnodes "${nodes_in_group}"
            --node-rank "${node_rank_in_group}"
        )
    fi
}

container_log_dir() {
    echo "${CONTAINER_LOG_ROOT}/slurm_job-${SLURM_JOB_ID}/${RUN_TIMESTAMP}"
}

recipe_log_dir() {
    echo "${SCRIPT_DIR}/logs/slurm_job-${SLURM_JOB_ID}/${RUN_TIMESTAMP}"
}

copy_logs_back() {
    local src_dir
    local dst_dir

    src_dir=$(container_log_dir)
    dst_dir=$(recipe_log_dir)
    mkdir -p "${dst_dir}"
    cp "${src_dir}"/* "${dst_dir}/" 2>/dev/null || true
}

wait_for_router_shutdown() {
    python "${SCRIPT_DIR}/utils/socket_barrier.py" \
        --node-ips "${NODE0_ADDR}" \
        --node-ports 30000 \
        --wait-for-all-ports \
        --timeout 1800

    python "${SCRIPT_DIR}/utils/socket_wait.py" \
        --remote-ip "${NODE0_ADDR}" \
        --remote-port 30000
}

prepare_runtime_pythonpath() {
    local runtime_sglang="/sgl-workspace/sglang"
    local pythonpath_parts=()

    if [[ -d "${WORKSPACE_SGLANG_MOUNT:-}" ]]; then
        mkdir -p /sgl-workspace
        rm -rf "${runtime_sglang}"
        ln -s "${WORKSPACE_SGLANG_MOUNT}" "${runtime_sglang}"
        log "Linked workspace sglang checkout: ${runtime_sglang} -> ${WORKSPACE_SGLANG_MOUNT}"
    else
        log "Using image-bundled sglang at ${runtime_sglang}"
    fi

    if [[ -d "${runtime_sglang}/python" ]]; then
        pythonpath_parts+=("${runtime_sglang}/python")
    fi
    if [[ -d "/sgl-workspace/aiter" ]]; then
        pythonpath_parts+=("/sgl-workspace/aiter")
    fi
    if [[ -n "${PYTHONPATH:-}" ]]; then
        pythonpath_parts+=("${PYTHONPATH}")
    fi

    PYTHONPATH=$(join_array_with_colon "${pythonpath_parts[@]}")
    export PYTHONPATH
    export SGLANG_WS_PATH="${SCRIPT_DIR}"
}

print_launch_details() {
    local role="$1"
    local dispatch_tokens="$2"
    shift 2

    echo "LAUNCH DETAIL =================================="
    echo "Role                : ${role}"
    echo "Host                : $(hostname):${SGLANG_HOST_IP}"
    echo "Node Rank           : ${NODE_RANK}"
    echo "Model               : ${MODEL_DIR}/${MODEL_NAME}"
    echo "Topology            : xP=${xP}, yD=${yD}, IPADDRS=${IPADDRS}"
    echo "Parallelism         : PREFILL_TP=${PREFILL_TP_SIZE}, PREFILL_EP=${PREFILL_ENABLE_EP}, PREFILL_DP=${PREFILL_ENABLE_DP}, DECODE_TP=${DECODE_TP_SIZE}, DECODE_EP=${DECODE_ENABLE_EP}, DECODE_DP=${DECODE_ENABLE_DP}, DECODE_MTP=${DECODE_MTP_SIZE}"
    echo "Benchmark           : mode=${BENCH_MODE}, requested_concurrency=${BENCH_MAX_CONCURRENCY}, effective_concurrency=${EFFECTIVE_BENCH_MAX_CONCURRENCY}"
    echo "Network             : IBDEVICES=${IBDEVICES}, GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}, NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}, SGLANG_HOST_IP=${SGLANG_HOST_IP}"
    echo "MORI                : DISPATCH_TOKENS=${dispatch_tokens}, MORI_SHMEM_MODE=${MORI_SHMEM_MODE}, MORI_RDMA_TC=${MORI_RDMA_TC}, MORI_RDMA_SL=${MORI_RDMA_SL}"
    echo "Python Path         : ${PYTHONPATH:-unset}"
    echo "Command             : $(cmd_to_string "$@")"
    echo "================================================"
}

run_slowdown_benchmark() {
    local log_dir
    local slowdown_active=0
    local benchmark_pid
    local benchmark_exit

    log_dir=$(container_log_dir)
    mkdir -p "${log_dir}"
    echo "${EFFECTIVE_BENCH_MAX_CONCURRENCY}" > "${log_dir}/effective_batch_size.txt"
    echo "Config input=${BENCH_INPUT_LEN} output=${BENCH_OUTPUT_LEN}; Concurrency=${EFFECTIVE_BENCH_MAX_CONCURRENCY}"

    release_slowdown() {
        if [[ "${slowdown_active}" -eq 1 ]]; then
            log "Releasing slow_down..."
            curl -fsS -H "Content-Type: application/json" \
                -d "{\"forward_sleep_time\": null}" \
                -X POST "http://${DECODE_HEAD_NODE}:8000/slow_down" || true
            slowdown_active=0
        fi
    }

    trap release_slowdown EXIT INT TERM

    sleep "${BENCH_STARTUP_WAIT_SECONDS}"
    log "Starting slow_down..."
    curl -fsS -H "Content-Type: application/json" \
        -d "{\"forward_sleep_time\": ${BENCH_FORWARD_SLEEP_TIME}}" \
        -X POST "http://${DECODE_HEAD_NODE}:8000/slow_down"
    slowdown_active=1

    (
        if [[ ! -f "${DATASET_FILE}" ]]; then
            echo "Dataset file not found at ${DATASET_FILE}, downloading..."
            mkdir -p "${DATASET_DIR}"
            wget -O "${DATASET_FILE}" "${DATASET_URL}"
        else
            echo "Using existing dataset file at ${DATASET_FILE}"
        fi

        if ! python3 -m pip show tabulate >/dev/null 2>&1; then
            python3 -m pip install tabulate
        fi

        python3 -m sglang.bench_one_batch_server \
            --dataset-path "${DATASET_FILE}" \
            --model-path "${MODEL_DIR}/${MODEL_NAME}" \
            --base-url "http://localhost:30000" \
            --batch-size "${EFFECTIVE_BENCH_MAX_CONCURRENCY}" \
            --input-len "${BENCH_INPUT_LEN}" \
            --output-len "${BENCH_OUTPUT_LEN}" \
            --skip-warmup
    ) &
    benchmark_pid=$!
    log "Benchmark running with PID ${benchmark_pid}"

    log "Waiting ${SLOWDOWN_DURATION}s before stopping slow_down..."
    sleep "${SLOWDOWN_DURATION}"
    if ! python3 "${SCRIPT_DIR}/utils/wait_for_prefill_idle.py" \
        --prefill_url "http://${PREFILL_HEAD_NODE}:8000" \
        --timeout "${PREFILL_IDLE_TIMEOUT}" \
        --request-timeout "${PREFILL_IDLE_REQUEST_TIMEOUT}" \
        --poll-interval "${PREFILL_IDLE_POLL_INTERVAL}"; then
        log "Warning: prefill idle check failed or timed out; continuing to release slow_down."
    fi

    log "Stopping slow_down..."
    release_slowdown

    log "Waiting for benchmark (PID ${benchmark_pid}) to finish..."
    wait "${benchmark_pid}"
    benchmark_exit=$?
    log "Benchmark finished with exit code ${benchmark_exit}"
    trap - EXIT INT TERM
    return "${benchmark_exit}"
}

run_prefill_rank() {
    local group_idx="$1"
    local group_rank="$2"
    local launch_pid=""
    local benchmark_status=0

    build_server_command \
        "prefill" \
        "${PREFILL_TP_SIZE}" \
        "${PREFILL_ENABLE_EP}" \
        "${PREFILL_ENABLE_DP}" \
        "${PREFILL_NODES_PER_WORKER}" \
        "${group_rank}" \
        "${PREFILL_HEADNODE_URLS[group_idx]}"

    print_launch_details "prefill" "${MORI_MAX_DISPATCH_TOKENS_PREFILL}" "${CURRENT_CMD[@]}"

    if is_true "${DRY_RUN}"; then
        echo "DRY RUN: $(cmd_to_string "${CURRENT_CMD[@]}")"
    else
        (
            export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${MORI_MAX_DISPATCH_TOKENS_PREFILL}"
            "${CURRENT_CMD[@]}"
        ) 2>&1 | tee "$(container_log_dir)/prefill_NODE${NODE_RANK}.log" &
        launch_pid=$!
    fi

    if [[ "${NODE_RANK}" -eq 0 ]]; then
        local router_cmd=(
            python -m sglang_router.launch_router
            --pd-disaggregation
            --mini-lb
            --port 30000
        )
        router_cmd+=("${ROUTER_EXTRA_ARGS[@]}")
        router_cmd+=("${ROUTER_PREFILL_ARGS[@]}")
        router_cmd+=("${ROUTER_DECODE_ARGS[@]}")

        if is_true "${DRY_RUN}"; then
            echo "DRY RUN: $(cmd_to_string "${router_cmd[@]}")"
            return 0
        fi

        python "${SCRIPT_DIR}/utils/socket_barrier.py" \
            --node-ips "${IPADDRS}" \
            --node-ports 8000 \
            --wait-for-all-ports \
            --timeout 1800

        local router_pid=""
        "${router_cmd[@]}" 2>&1 | tee "$(container_log_dir)/router_NODE${NODE_RANK}.log" &
        router_pid=$!

        python "${SCRIPT_DIR}/utils/socket_barrier.py" \
            --node-ips "${NODE0_ADDR}" \
            --node-ports 30000 \
            --wait-for-all-health \
            --health-endpoint /health \
            --timeout 1800

        run_slowdown_benchmark || benchmark_status=$?
        kill "${router_pid}" >/dev/null 2>&1 || true
        kill "${launch_pid}" >/dev/null 2>&1 || true
        copy_logs_back
        return "${benchmark_status}"
    elif ! is_true "${DRY_RUN}"; then
        wait_for_router_shutdown
        kill "${launch_pid}" >/dev/null 2>&1 || true
    fi
}

run_decode_rank() {
    local decode_rank="$1"
    local group_idx="$2"
    local group_rank="$3"
    local launch_pid=""

    build_server_command \
        "decode" \
        "${DECODE_TP_SIZE}" \
        "${DECODE_ENABLE_EP}" \
        "${DECODE_ENABLE_DP}" \
        "${DECODE_NODES_PER_WORKER}" \
        "${group_rank}" \
        "${DECODE_HEADNODE_URLS[group_idx]}"

    print_launch_details "decode" "${MORI_MAX_DISPATCH_TOKENS_DECODE}" "${CURRENT_CMD[@]}"

    if is_true "${DRY_RUN}"; then
        echo "DRY RUN: $(cmd_to_string "${CURRENT_CMD[@]}")"
    else
        (
            export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${MORI_MAX_DISPATCH_TOKENS_DECODE}"
            "${CURRENT_CMD[@]}"
        ) 2>&1 | tee "$(container_log_dir)/decode_NODE${NODE_RANK}.log" &
        launch_pid=$!

        wait_for_router_shutdown
        kill "${launch_pid}" >/dev/null 2>&1 || true
    fi

    if [[ "${decode_rank}" -eq 0 ]]; then
        copy_logs_back
        if ! is_true "${DRY_RUN}"; then
            local result
            if result=$(python "${SCRIPT_DIR}/utils/parse_decode_log.py" \
                "$(container_log_dir)/decode_NODE${NODE_RANK}.log" \
                "${BENCH_OUTPUT_LEN}" \
                "${EFFECTIVE_BENCH_MAX_CONCURRENCY}"); then
                local tpot
                local output_throughput
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
}

container_main() {
    source_config
    init_run_timestamp
    derive_topology
    prepare_cluster_urls
    prepare_runtime_pythonpath

    export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT
    export SGLANG_DISAGGREGATION_WAITING_TIMEOUT
    export MORI_SHMEM_MODE
    export SGLANG_MORI_FP8_DISP
    export MORI_EP_LAUNCH_CONFIG_MODE
    export MORI_APP_LOG_LEVEL
    export MORI_RDMA_SL
    export MORI_RDMA_TC
    export IBDEVICES
    export GLOO_SOCKET_IFNAME
    export NCCL_SOCKET_IFNAME
    export NCCL_IB_HCA
    export SGLANG_HOST_IP

    mkdir -p "$(container_log_dir)"

    if ! is_true "${DRY_RUN}"; then
        log "Waiting at the container creation barrier on $(hostname)"
        python "${SCRIPT_DIR}/utils/socket_barrier.py" \
            --local-ip "${SGLANG_HOST_IP}" \
            --local-port 5000 \
            --enable-port \
            --node-ips "${IPADDRS}" \
            --node-ports 5000 \
            --wait-for-all-ports \
            --timeout 300
    fi

    if (( NODE_RANK < NODE_OFFSET )); then
        run_prefill_rank "$((NODE_RANK / PREFILL_NODES_PER_WORKER))" "$((NODE_RANK % PREFILL_NODES_PER_WORKER))"
    else
        local decode_rank=$((NODE_RANK - NODE_OFFSET))
        run_decode_rank \
            "${decode_rank}" \
            "$((decode_rank / DECODE_NODES_PER_WORKER))" \
            "$((decode_rank % DECODE_NODES_PER_WORKER))"
    fi

    echo "Script completed successfully"
}

run_node_container() {
    source_config
    init_run_timestamp
    validate_config
    derive_topology
    detect_node_env

    local container_name="${DOCKER_CONT_BASENAME}_${SLURM_PROCID}"
    local container_config_file="${RECIPE_CONTAINER_DIR_DEFAULT}/$(basename "${CONFIG_FILE}")"
    local nicctl_bin
    nicctl_bin=$(command -v nicctl)
    require_non_empty "nicctl" "${nicctl_bin}"

    sudo docker rm -f "${container_name}" >/dev/null 2>&1 || true

    local -a docker_args=(
        --rm
        --init
        --stop-timeout 10
        --device /dev/dri
        --device /dev/kfd
        --device /dev/infiniband
        --device /dev/infiniband/rdma_cm
        --device /dev/infiniband/uverbs0
        --device /dev/infiniband/uverbs1
        --device /dev/infiniband/uverbs2
        --device /dev/infiniband/uverbs3
        --device /dev/infiniband/uverbs4
        --device /dev/infiniband/uverbs5
        --device /dev/infiniband/uverbs6
        --device /dev/infiniband/uverbs7
        --ulimit memlock=-1
        --ulimit stack=67108864
        --network host
        --ipc host
        --group-add video
        --cap-add SYS_PTRACE
        --security-opt seccomp=unconfined
        --privileged
        --shm-size 128G
        -v "${MODEL_DIR}:/models"
        -v "/tmp:${CONTAINER_LOG_ROOT}"
        -v "${SCRIPT_DIR}:${RECIPE_CONTAINER_DIR_DEFAULT}"
        -v "${nicctl_bin}:/usr/sbin/nicctl"
        --name "${container_name}"
    )

    local mount_spec
    for mount_spec in "${EXTRA_CONTAINER_MOUNTS[@]}"; do
        docker_args+=(-v "${mount_spec}")
    done

    if [[ -n "${WORKSPACE_SGLANG_DIR}" ]]; then
        docker_args+=(-v "${WORKSPACE_SGLANG_DIR}:${WORKSPACE_SGLANG_MOUNT}:ro")
    fi

    local -a pass_env_vars=(
        "CONFIG_FILE=${container_config_file}"
        "SLURM_JOB_ID=${SLURM_JOB_ID}"
        "RUN_TIMESTAMP=${RUN_TIMESTAMP}"
        "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
        "NODE_RANK=${SLURM_PROCID}"
        "NODE0_ADDR=${NODE0_ADDR}"
        "IPADDRS=${IPADDRS}"
        "NUM_NODES=${NUM_NODES}"
        "MODEL_DIR=/models"
        "MODEL_NAME=${MODEL_NAME}"
        "LOAD_DUMMY=${LOAD_DUMMY}"
        "xP=${xP}"
        "yD=${yD}"
        "GPUS_PER_NODE=${GPUS_PER_NODE}"
        "PREFILL_TP_SIZE=${PREFILL_TP_SIZE}"
        "PREFILL_ENABLE_EP=${PREFILL_ENABLE_EP}"
        "PREFILL_ENABLE_DP=${PREFILL_ENABLE_DP}"
        "PREFILL_MEM_FRACTION_STATIC=${PREFILL_MEM_FRACTION_STATIC}"
        "PREFILL_MAX_RUNNING_REQUESTS=${PREFILL_MAX_RUNNING_REQUESTS}"
        "PREFILL_CHUNKED_PREFILL_SIZE=${PREFILL_CHUNKED_PREFILL_SIZE}"
        "DECODE_TP_SIZE=${DECODE_TP_SIZE}"
        "DECODE_ENABLE_EP=${DECODE_ENABLE_EP}"
        "DECODE_ENABLE_DP=${DECODE_ENABLE_DP}"
        "DECODE_MTP_SIZE=${DECODE_MTP_SIZE}"
        "DECODE_MEM_FRACTION_STATIC=${DECODE_MEM_FRACTION_STATIC}"
        "DECODE_MAX_RUNNING_REQUESTS=${DECODE_MAX_RUNNING_REQUESTS}"
        "DECODE_CHUNKED_PREFILL_SIZE=${DECODE_CHUNKED_PREFILL_SIZE}"
        "BENCH_MODE=${BENCH_MODE}"
        "BENCH_INPUT_LEN=${BENCH_INPUT_LEN}"
        "BENCH_OUTPUT_LEN=${BENCH_OUTPUT_LEN}"
        "BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY}"
        "BENCH_AUTO_CLAMP=${BENCH_AUTO_CLAMP}"
        "BENCH_STARTUP_WAIT_SECONDS=${BENCH_STARTUP_WAIT_SECONDS}"
        "BENCH_FORWARD_SLEEP_TIME=${BENCH_FORWARD_SLEEP_TIME}"
        "SLOWDOWN_DURATION=${SLOWDOWN_DURATION}"
        "PREFILL_IDLE_TIMEOUT=${PREFILL_IDLE_TIMEOUT}"
        "PREFILL_IDLE_REQUEST_TIMEOUT=${PREFILL_IDLE_REQUEST_TIMEOUT}"
        "PREFILL_IDLE_POLL_INTERVAL=${PREFILL_IDLE_POLL_INTERVAL}"
        "DATASET_DIR=${DATASET_DIR}"
        "DATASET_FILE=${DATASET_FILE}"
        "DATASET_URL=${DATASET_URL}"
        "HEADNODE_PORT=${HEADNODE_PORT}"
        "DRY_RUN=${DRY_RUN}"
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT}"
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=${SGLANG_DISAGGREGATION_WAITING_TIMEOUT}"
        "MORI_SHMEM_MODE=${MORI_SHMEM_MODE}"
        "SGLANG_MORI_FP8_DISP=${SGLANG_MORI_FP8_DISP}"
        "MORI_EP_LAUNCH_CONFIG_MODE=${MORI_EP_LAUNCH_CONFIG_MODE}"
        "MORI_MAX_DISPATCH_TOKENS_PREFILL=${MORI_MAX_DISPATCH_TOKENS_PREFILL}"
        "MORI_MAX_DISPATCH_TOKENS_DECODE=${MORI_MAX_DISPATCH_TOKENS_DECODE}"
        "MORI_APP_LOG_LEVEL=${MORI_APP_LOG_LEVEL}"
        "IBDEVICES=${IBDEVICES}"
        "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
        "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
        "SGLANG_HOST_IP=${SGLANG_HOST_IP}"
        "NCCL_IB_HCA=${NCCL_IB_HCA}"
        "MORI_RDMA_SL=${MORI_RDMA_SL}"
        "MORI_RDMA_TC=${MORI_RDMA_TC}"
        "WORKSPACE_SGLANG_MOUNT=${WORKSPACE_SGLANG_MOUNT}"
    )

    local env_spec
    for env_spec in "${pass_env_vars[@]}" "${CONTAINER_ENV_OVERRIDES[@]}"; do
        docker_args+=(-e "${env_spec}")
    done

    docker_args+=("${EXTRA_DOCKER_ARGS[@]}")

    log "Rank ${SLURM_PROCID} on $(hostname) -> ${container_name}"

    local log_dir
    log_dir=$(container_log_dir)

    exec sudo docker run \
        "${docker_args[@]}" \
        "${DOCKER_IMAGE_NAME}" \
        bash -lc "mkdir -p '${log_dir}' && bash '${RECIPE_CONTAINER_DIR_DEFAULT}/run.sh' __inside_container 2>&1 | tee '${log_dir}/pd_server_NODE${SLURM_PROCID}.log'"
}

host_main() {
    source_config
    init_run_timestamp
    validate_config
    derive_topology
    select_nodes_from_slurm

    log "Using config: $(basename "${CONFIG_FILE}")"
    log "Run timestamp: ${RUN_TIMESTAMP}"

    log "Refreshing NFS caches on selected nodes..."
    srun --nodelist="${SELECTED_NODELIST_SRUN}" bash -lc "
        sync
        ls -la '${SCRIPT_DIR}' >/dev/null 2>&1
        stat '${SCRIPT_DIR}/run.sh' >/dev/null 2>&1
        echo 'NFS cache refreshed on' \$(hostname)
    "

    SANITIZED_USER=$(whoami | tr -c 'a-zA-Z0-9_.-' '_')
    DOCKER_CONT_BASENAME="customer_pd_${SANITIZED_USER}_${MODEL_NAME}_${SLURM_JOB_ID}"
    export DOCKER_CONT_BASENAME NODE0_ADDR IPADDRS NUM_NODES RUN_TIMESTAMP SLURM_JOB_NODELIST SLURM_JOB_ID

    log "Launching one container per node..."
    srun \
        --nodes="${NUM_NODES}" \
        --ntasks="${NUM_NODES}" \
        --nodelist="${SELECTED_NODELIST_SRUN}" \
        --kill-on-bad-exit=1 \
        --unbuffered \
        bash "${SCRIPT_DIR}/run.sh" __node_launcher
}

main() {
    local mode="${1:-__host_main}"
    case "${mode}" in
        __host_main)
            host_main
            ;;
        __node_launcher)
            run_node_container
            ;;
        __inside_container)
            container_main
            ;;
        *)
            fail "Unsupported mode: ${mode}"
            ;;
    esac
}

main "${1:-__host_main}"
