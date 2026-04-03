#!/bin/bash
set -euo pipefail

CONTAINER_NAME="${DOCKER_CONT_BASENAME}_${SLURM_PROCID}"
NICCTL_BIN=$(command -v nicctl)

sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker_args=(
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
    -v "/tmp:/run_logs"
    -v "${REPO_DIR}:${SGL_WS_PATH}"
    -v "/mnt/nfs:/mnt/nfs"
    -v "/opt/amd:/opt/amd"
    -v "${NICCTL_BIN}:/usr/sbin/nicctl"
    -e PYTHONUNBUFFERED=1
    -e SLURM_JOB_ID="${SLURM_JOB_ID}"
    -e SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST}"
    -e NODE_RANK="${SLURM_PROCID}"
    -e NODE0_ADDR="${NODE0_ADDR}"
    -e IPADDRS="${IPADDRS}"
    -e MODEL_DIR=/models
    -e MODEL_NAME="${MODEL_NAME}"
    -e LOAD_DUMMY="${LOAD_DUMMY}"
    -e SGL_WS_PATH="${SGL_WS_PATH}"
    -e SGLANG_WS_PATH="${SGL_WS_PATH}"
    -e xP="${xP}"
    -e yD="${yD}"
    -e PREFILL_TP_SIZE="${PREFILL_TP_SIZE}"
    -e PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP}"
    -e PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP}"
    -e DECODE_TP_SIZE="${DECODE_TP_SIZE}"
    -e DECODE_ENABLE_EP="${DECODE_ENABLE_EP}"
    -e DECODE_ENABLE_DP="${DECODE_ENABLE_DP}"
    -e DECODE_MTP_SIZE="${DECODE_MTP_SIZE}"
    -e BENCH_INPUT_LEN="${BENCH_INPUT_LEN}"
    -e BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN}"
    -e BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO}"
    -e BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER}"
    -e BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY}"
    -e BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE}"
    -e BENCH_AUTO_CLAMP="${BENCH_AUTO_CLAMP}"
    -e BENCH_MODE="${BENCH_MODE}"
    -e BENCH_BACKEND="${BENCH_BACKEND}"
    -e BENCH_BURSTINESS="${BENCH_BURSTINESS}"
    -e BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS}"
    -e BENCH_NUM_WARMUPS="${BENCH_NUM_WARMUPS}"
    -e BENCH_RESULT_DIR="${BENCH_RESULT_DIR}"
    -e BENCH_RESULT_FILENAME="${BENCH_RESULT_FILENAME}"
    -e BENCH_POISSON_USE_SLOWDOWN_PHASE_CONTROL="${BENCH_POISSON_USE_SLOWDOWN_PHASE_CONTROL:-false}"
    -e SLOWDOWN_DURATION="${SLOWDOWN_DURATION:-}"
    -e PREFILL_IDLE_TIMEOUT="${PREFILL_IDLE_TIMEOUT:-}"
    -e PREFILL_IDLE_REQUEST_TIMEOUT="${PREFILL_IDLE_REQUEST_TIMEOUT:-}"
    -e PREFILL_IDLE_POLL_INTERVAL="${PREFILL_IDLE_POLL_INTERVAL:-}"
    -e DRY_RUN="${DRY_RUN}"
    --name "${CONTAINER_NAME}"
)

if [[ -n "${WORKSPACE_SGLANG_DIR:-}" ]]; then
    docker_args+=(-v "${WORKSPACE_SGLANG_DIR}:/workspace_sglang:ro")
fi

echo "Rank ${SLURM_PROCID} on $(hostname) -> ${CONTAINER_NAME}"

cleanup() {
    sudo docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

sudo docker run -d \
    "${docker_args[@]}" \
    "${DOCKER_IMAGE_NAME}" \
    bash -lc 'trap "exit 0" TERM INT; while true; do sleep 3600; done' >/dev/null

if [[ "${RESTART_CONTAINER_BEFORE_SERVER:-1}" == "1" ]]; then
    echo "Restarting container ${CONTAINER_NAME} before launching pd_server.sh"
    sudo docker restart "${CONTAINER_NAME}" >/dev/null
fi

sudo docker exec "${CONTAINER_NAME}" bash -lc '
    mkdir -p /run_logs/slurm_job-${SLURM_JOB_ID}
    bash '"${SGL_WS_PATH}"'/scripts/pd_server.sh 2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/pd_server_NODE${NODE_RANK}.log
'
