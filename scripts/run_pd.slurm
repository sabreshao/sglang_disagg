#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SGL_WS_PATH="/sglang_disagg_simple"

DRY_RUN="${DRY_RUN:-0}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_DIR="${MODEL_DIR:-}"
DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-}"
WORKSPACE_SGLANG_DIR="${WORKSPACE_SGLANG_DIR:-}"

xP="${xP:-1}"
yD="${yD:-1}"
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
BENCH_MODE="${BENCH_MODE:-slowdown}"
BENCH_BACKEND="${BENCH_BACKEND:-sglang}"
BENCH_BURSTINESS="${BENCH_BURSTINESS:-1.0}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-}"
BENCH_NUM_WARMUPS="${BENCH_NUM_WARMUPS:-}"
BENCH_RESULT_DIR="${BENCH_RESULT_DIR:-}"
BENCH_RESULT_FILENAME="${BENCH_RESULT_FILENAME:-}"
LOAD_DUMMY="${LOAD_DUMMY:-0}"

echo "=== Job Start Time ==="
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
echo "======================="
echo ""

SUPPORTED_MODELS=("DeepSeek-V3.2" "DeepSeek-V3.2-mxfp4" "DeepSeek-R1")

validate_model_name() {
    local model
    for model in "${SUPPORTED_MODELS[@]}"; do
        if [[ "${MODEL_NAME}" == "${model}" ]]; then
            return 0
        fi
    done

    echo "Unsupported MODEL_NAME: ${MODEL_NAME}" >&2
    printf 'Supported models:\n' >&2
    printf '  - %s\n' "${SUPPORTED_MODELS[@]}" >&2
    exit 1
}

nodes_per_worker() {
    local tp_size="$1"
    echo $(((tp_size + 7) / 8))
}

require_non_empty() {
    local name="$1"
    local value="$2"
    if [[ -z "${value}" ]]; then
        echo "Required variable ${name} is empty." >&2
        exit 1
    fi
}

validate_model_name
require_non_empty "MODEL_DIR" "${MODEL_DIR}"
require_non_empty "DOCKER_IMAGE_NAME" "${DOCKER_IMAGE_NAME}"

PREFILL_NODES_PER_WORKER=$(nodes_per_worker "${PREFILL_TP_SIZE}")
DECODE_NODES_PER_WORKER=$(nodes_per_worker "${DECODE_TP_SIZE}")
EXPECTED_NUM_NODES=$((xP * PREFILL_NODES_PER_WORKER + yD * DECODE_NODES_PER_WORKER))

if [[ -z "${NUM_NODES:-}" ]]; then
    NUM_NODES="${EXPECTED_NUM_NODES}"
elif [[ "${NUM_NODES}" -ne "${EXPECTED_NUM_NODES}" ]]; then
    echo "NUM_NODES=${NUM_NODES} does not match expected topology ${EXPECTED_NUM_NODES}." >&2
    exit 1
fi

if [[ -n "${WORKSPACE_SGLANG_DIR}" ]] && [[ ! -d "${WORKSPACE_SGLANG_DIR}" ]]; then
    echo "WORKSPACE_SGLANG_DIR does not exist: ${WORKSPACE_SGLANG_DIR}" >&2
    exit 1
fi

FULL_NODELIST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")
TOTAL_ALLOCATED=$(echo "${FULL_NODELIST}" | wc -l)
if [[ "${NUM_NODES}" -gt "${TOTAL_ALLOCATED}" ]]; then
    echo "Need ${NUM_NODES} nodes but only ${TOTAL_ALLOCATED} allocated." >&2
    exit 1
fi

SELECTED_NODES=$(echo "${FULL_NODELIST}" | head -n "${NUM_NODES}")
SELECTED_NODELIST_SRUN=$(echo "${SELECTED_NODES}" | paste -sd,)

echo "Selected nodes:"
echo "${SELECTED_NODES}"

check_model_path() {
    local path="$1"
    srun --nodelist="${SELECTED_NODELIST_SRUN}" --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" /bin/bash -lc "
        if [[ -d '${path}' ]]; then
            echo \"\$(hostname): found ${path}\"
        else
            echo \"\$(hostname): missing ${path}\" >&2
            exit 1
        fi
    "
}

echo "Checking model availability on selected nodes..."
check_model_path "${MODEL_DIR}/${MODEL_NAME}"

MASTER_NODE=$(echo "${SELECTED_NODES}" | head -n 1)
NODE0_ADDR=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="${MASTER_NODE}" bash -lc "ip route get 10.2.224.0 | awk '/src/ {print \$5}'")

IPS=()
for NODE in ${SELECTED_NODES}; do
    IP=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="${NODE}" bash -lc "ip route get 10.2.224.0 | awk '/src/ {print \$5}'")
    echo "${NODE} ${IP}"
    IPS+=("${IP}")
done
IPADDRS=$(IFS=,; echo "${IPS[*]}")

if ! command -v nicctl >/dev/null 2>&1; then
    echo "Error: nicctl not found on host." >&2
    exit 1
fi

echo "Refreshing NFS caches on selected nodes..."
srun --nodelist="${SELECTED_NODELIST_SRUN}" bash -lc "
    sync
    ls -la '${REPO_DIR}' >/dev/null 2>&1
    stat '${REPO_DIR}/scripts/pd_server.sh' >/dev/null 2>&1
    echo 'NFS cache refreshed on' \$(hostname)
"

SANITIZED_USER=$(whoami | tr -c 'a-zA-Z0-9_.-' '_')
DOCKER_CONT_BASENAME="simple_pd_${SANITIZED_USER}_${MODEL_NAME}_${SLURM_JOB_ID}"

export REPO_DIR
export SGL_WS_PATH
export NODE0_ADDR
export IPADDRS
export NUM_NODES
export DOCKER_CONT_BASENAME

export MODEL_NAME MODEL_DIR DOCKER_IMAGE_NAME WORKSPACE_SGLANG_DIR
export xP yD
export PREFILL_TP_SIZE PREFILL_ENABLE_EP PREFILL_ENABLE_DP
export DECODE_TP_SIZE DECODE_ENABLE_EP DECODE_ENABLE_DP DECODE_MTP_SIZE
export BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_RANDOM_RANGE_RATIO
export BENCH_NUM_PROMPTS_MULTIPLIER BENCH_MAX_CONCURRENCY BENCH_REQUEST_RATE
export BENCH_AUTO_CLAMP BENCH_MODE BENCH_BACKEND BENCH_BURSTINESS
export BENCH_NUM_PROMPTS BENCH_NUM_WARMUPS BENCH_RESULT_DIR BENCH_RESULT_FILENAME
export LOAD_DUMMY DRY_RUN RESTART_CONTAINER_BEFORE_SERVER

echo "Launching one container per node..."
srun \
    --nodes="${NUM_NODES}" \
    --ntasks="${NUM_NODES}" \
    --nodelist="${SELECTED_NODELIST_SRUN}" \
    --kill-on-bad-exit=1 \
    --unbuffered \
    bash "${SCRIPT_DIR}/node_docker.sh"
