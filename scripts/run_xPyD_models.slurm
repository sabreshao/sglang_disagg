#!/bin/bash
# =============================================================================
# run_xPyD_models.slurm - SLURM Orchestration Script
# =============================================================================
# This script is the core orchestration layer for disaggregated inference.
# It can be submitted as a batch job via sbatch, or called directly by the
# interactive entrypoints (run_interactive_disagg.sh / run_interactive_disagg_qwen3.sh)
# after nodes have been reserved with salloc.
#
# Responsibilities:
#   1. Validate MODEL_NAME against the supported model list
#   2. Verify model weights are accessible on all allocated nodes
#   3. Trim the SLURM allocation down to NUM_NODES nodes
#   4. Resolve IP addresses for all selected nodes
#   5. On each node, start a Docker container and run sglang_disagg_server.sh
#      (which assigns each node its prefill/decode/router role)
#   6. Clean up Docker containers after completion
#
# All per-node server logic (prefill/decode/router/benchmark) lives inside
# sglang_disagg_server.sh, which runs inside Docker on each node.
# =============================================================================

# Default SLURM directives (only active when submitted via sbatch).
# For interactive (salloc) use, these are ignored.
#SBATCH --job-name=1p2d_bench-serving    # Specify a custom string for your slurm batch job
#SBATCH -N 3            # CHECK this to be right in batch jobs
#SBATCH -n 3          # CHECK this to be right in batch jobs
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job
#SBATCH --gres=gpu:8      # Request 8 GPUs and 8 NICs (use --gres if specific GPU resources are needed)
#SBATCH --time=24:00:00         # Set a time limit for the job (HH:MM:SS)
#SBATCH --output="slurm_job-%j.out"
#SBATCH --error="slurm_job-%j.err"

# DRY_RUN=1: print Docker/srun commands without running them (useful for debug)
DRY_RUN="${DRY_RUN:-0}"

# ------------------------
# Print current time in UTC and PST formats
# ------------------------
echo "=== Job Start Time ==="
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
echo "======================="
echo ""

# =============================================================================
# Model Validation
# =============================================================================
# Only models in VALID_MODELS are supported. MODEL_RUNFILES maps each model
# to the script that will be executed inside Docker on each node.

VALID_MODELS=( \
    "DeepSeek-V3" \
    "DeepSeek-V3-0324" \
    "DeepSeek-R1" \
    "DeepSeek-R1-0528-MXFP4-Preview" \
    "Qwen3-235B"
)

# Each model maps to its per-node server launcher script
declare -A MODEL_RUNFILES=(
    ["DeepSeek-V3"]="scripts/sglang_disagg_server.sh"
    ["DeepSeek-V3-0324"]="scripts/sglang_disagg_server.sh"
    ["DeepSeek-R1"]="scripts/sglang_disagg_server.sh"
    ["DeepSeek-R1-0528-MXFP4-Preview"]="scripts/sglang_disagg_server.sh"
    ["Qwen3-235B"]="scripts/sglang_disagg_server.sh"
)

# Resolve the run file for the requested model
if [[ -n "${MODEL_RUNFILES[$MODEL_NAME]}" ]]; then
    RUN_FILE="${MODEL_RUNFILES[$MODEL_NAME]}"
    echo "Model found: $MODEL_NAME"
    echo "Runfile set: $RUN_FILE"
else
    echo "Error: Model '$MODEL_NAME' not found in MODEL_RUNFILES"
    echo "Available models: ${!MODEL_RUNFILES[@]}"
    exit 1
fi

# =============================================================================
# Docker Image Selection
# =============================================================================
# The default image supports Qwen3 + MORI; override DOCKER_IMAGE_NAME in the
# entrypoint script to pin a different image for specific models.
#export DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-1224}"
#export DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-rocm/sgl-dev:sglang-0.5.8-rocm700-mi35x-mori-0210}"
#export DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-rocm/ali-private:sglang-0.5.8-rocm700-mi35x-mori-0210-qwen3-moe-0228}"
#export DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-lmsysorg/sglang:v0.5.10rc0-rocm700-mi35x}"
#export DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260330}"
export DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-lmsysorg/sglang:v0.5.9-rocm700-mi35x}"
#export DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-rocm/sgl-dev:v0.5.8-rocm700-mi35x-20260202}"
if test -z "${DOCKER_IMAGE_NAME}"; then
  echo "Error: DOCKER_IMAGE_NAME is not set or empty."
  exit 1
fi

# DI_REPO_DIR is the directory containing this script and all helper scripts.
# It gets bind-mounted into Docker at /sglang_disagg.
export DI_REPO_DIR=$(pwd)

xP="${xP:-1}"  # Number of prefill worker groups
yD="${yD:-1}"  # Number of decode worker groups

# Parallelism Configuration with defaults
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP:-true}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-true}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
DECODE_ENABLE_EP="${DECODE_ENABLE_EP:-true}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-true}"
DECODE_MTP_SIZE=${DECODE_MTP_SIZE:-0}  # 0 = MTP/speculative decoding disabled

# Benchmark Configuration with defaults
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-1}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-512}"

MODEL_NAME="${MODEL_NAME:-None}"
MODEL_DIR="${MODEL_DIR:-"/nfsdata"}"
LOAD_DUMMY="${LOAD_DUMMY:-"0"}"
SGLANG_DIR="${SGLANG_DIR:-""}"  # Optional: path to an external SGLang checkout

# Validate that MODEL_NAME is in the supported list
validate_model_name() {
  local is_valid_model=false

  for model in "${VALID_MODELS[@]}"; do
    if [[ "$MODEL_NAME" == "$model" ]]; then
      is_valid_model=true
      break
    fi
  done

  if ! $is_valid_model; then
    printf "Error: Invalid MODEL_NAME: '%s'\nValid models are:\n" "$MODEL_NAME"
    for model in "${VALID_MODELS[@]}"; do
      printf "  - %s\n" "$model"
    done
    exit 1
  fi

  echo "MODEL_NAME '$MODEL_NAME' is valid."
  return 0
}

validate_model_name "${MODEL_NAME}"


# =============================================================================
# Model Path Validation Across All Nodes
# =============================================================================
# Model weights must be accessible on every allocated node (typically via NFS).
# We run a parallel check using srun to confirm availability before launching.
echo "Looking for model: $MODEL_NAME"
echo "Checking model availability across all allocated nodes..."

ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
TOTAL_NODES=$(echo "$ALL_NODES" | wc -l)

echo "Total allocated nodes: $TOTAL_NODES"
echo "Nodes: $(echo "$ALL_NODES" | tr '\n' ' ')"

# check_model_path: Runs a directory existence check on all allocated nodes in
# parallel via srun. Returns 0 if all nodes can find the path, 1 otherwise.
check_model_path() {
    local path=$1
    local check_name=$2

    echo "Checking $check_name: $path"

    srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES /bin/bash -c "
        if [ -d '$path' ]; then
            echo \"\$(hostname): ✓ Found $path\"
            exit 0
        else
            echo \"\$(hostname): ✗ Missing $path\"
            exit 1
        fi
    "

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ $check_name available on ALL nodes"
        return 0
    else
        echo "✗ $check_name NOT available on all nodes"
        return 1
    fi
}

# Abort if model weights are missing on any node
if check_model_path "$MODEL_DIR/$MODEL_NAME" "$MODEL_DIR"; then
    MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
    echo ""
    echo "✓ Selected MODEL_PATH: $MODEL_PATH (available on all nodes)"
else
    echo ""
    echo "✗ FATAL ERROR: Model '$MODEL_NAME' not found on ALL allocated nodes in the following:"
    echo "  - $MODEL_DIR/$MODEL_NAME"
    echo ""
    echo "Model must be accessible from all nodes for distributed execution."
    echo "Please ensure the model is available on all allocated nodes."
    exit 1
fi

echo "Final MODEL_PATH: $MODEL_PATH"
echo ""

NUM_NODES="${NUM_NODES}"


# =============================================================================
# External SGLang Checkout (optional)
# =============================================================================
# If SGLANG_DIR is set, bind-mount it into the container at /sgl-workspace/sglang
# to override the SGLang version baked into the Docker image.
if [[ -z "${SGLANG_DIR}" ]]; then
    echo "Using internal sglang"
    SGLANG_MAP=" "
else
    echo "Using external sglang"
    SGLANG_MAP="-v ${SGLANG_DIR}:/sgl-workspace/sglang "
fi

# =============================================================================
# Node Selection and SLURM Environment Setup
# =============================================================================
# When salloc allocates more nodes than NUM_NODES, we trim the list to the
# first NUM_NODES and update all SLURM env vars accordingly. This allows users
# to hold a larger pool and only use a subset for a given run.
echo "Original SLURM allocation:"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"

FULL_NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
SELECTED_NODES=$(echo "$FULL_NODELIST" | head -n $NUM_NODES)
SELECTED_NODELIST_STR=$(echo "$SELECTED_NODES" | tr '\n' ',' | sed 's/,$//')

NEW_SLURM_NODELIST=$(echo "$SELECTED_NODES" | paste -sd, | sed 's/,/,/g')

# Override SLURM env vars to reflect the trimmed node set
export SLURM_NNODES=$NUM_NODES
export SLURM_NTASKS=$NUM_NODES
export SLURM_JOB_NUM_NODES=$NUM_NODES
export SLURM_NPROCS=$NUM_NODES
export SLURM_JOB_NODELIST="$NEW_SLURM_NODELIST"
export SLURM_NODELIST="$NEW_SLURM_NODELIST"

# Keep other SLURM variables as they were or set defaults
export SLURM_TASKS_PER_NODE="1(x$NUM_NODES)"
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/home/bill}"
export SLURM_CLUSTER_NAME="${SLURM_CLUSTER_NAME:-m2m}"
export SLURM_JOB_CPUS_PER_NODE="${SLURM_JOB_CPUS_PER_NODE}"
export SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-amd-rccl}"
export SLURM_JOBID="${SLURM_JOBID:-$SLURM_JOB_ID}"
export SLURM_JOB_QOS="${SLURM_JOB_QOS:-normal}"
export SLURM_JOB_ACCOUNT="${SLURM_JOB_ACCOUNT:-amd-rccl}"
export SLURM_NTASKS_PER_NODE=1
export SLURM_SUBMIT_HOST="${SLURM_SUBMIT_HOST}"
export SLURM_JOB_ID="${SLURM_JOB_ID}"
export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-1p1d_bench-serving}"

echo ""
echo "Updated SLURM Environment Variables:"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_QOS: $SLURM_JOB_QOS"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_ACCOUNT: $SLURM_JOB_ACCOUNT"
echo "SLURM_NPROCS: $SLURM_NPROCS"
echo "SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "SLURM_CONF: $SLURM_CONF"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "SLURM_CLUSTER_NAME: $SLURM_CLUSTER_NAME"
echo "ulimit: $(ulimit -a)"
echo ""
echo "Selected nodes for execution:"
echo "$SELECTED_NODES"
echo ""

# =============================================================================
# IP Address Resolution
# =============================================================================
# Resolve the primary IP of each selected node using its route to the fabric
# subnet (10.2.224.0). IPADDRS is passed into Docker containers so each node
# can identify its peers.
USER_NAME=$(whoami)
MASTER_NODE=$(echo "$SELECTED_NODES" | head -n 1)
NODE0_ADDR=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$MASTER_NODE" bash -c 'ip route get 10.2.224.0')
NODE0_ADDR=$(echo "$NODE0_ADDR" | awk '/src/ {print $5}')

IPS=()

GW_NIC=enp193s0f1np1
for NODE in $SELECTED_NODES; do
    IP=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$NODE" bash -c 'ip route get 10.2.224.0')
    IP=$(echo "$IP" | awk '/src/ {print $5}')
    echo "$NODE $IP"
    IPS+=("$IP")
done

echo "Selected node IPs: ${IPS[*]}" | sed 's/ /,/g'

SGL_WS_PATH="/sglang_disagg"  # Path inside Docker where DI_REPO_DIR is mounted
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

NNODES=$NUM_NODES

echo "MASTER_NODE is ${MASTER_NODE}"
echo "NODE0_ADDR is ${NODE0_ADDR}"
echo "NNODES is ${NNODES}"
echo "REPO Directory is ${DI_REPO_DIR}"
echo "USER_NAME is ${USER_NAME}"

# =============================================================================
# RDMA QoS Validation
# =============================================================================
# nicctl is the AMD AINIC control utility. It must be present on the host to
# configure RDMA traffic class (TC) and service level (SL) for MORI RDMA.
if ! command -v nicctl >/dev/null 2>&1; then
    echo "Error: nicctl command not found. Please ensure nicctl is installed and available." >&2
    exit 1
fi

# Export all variables needed by sglang_disagg_server.sh inside Docker
export DI_REPO_DIR=$DI_REPO_DIR
export SGL_WS_PATH=$SGL_WS_PATH
export NNODES=$NNODES
export NODE0_ADDR=$NODE0_ADDR
export MODEL_PATH=$MODEL_PATH
export MODEL_DIR=$MODEL_DIR
export xP=$xP
export yD=$yD
export MODEL_NAME=$MODEL_NAME
export USER_NAME=$USER_NAME
export IPADDRS="$(echo "${IPS[*]}" | sed 's/ /,/g')"
export PREFILL_TP_SIZE=$PREFILL_TP_SIZE
export PREFILL_ENABLE_EP=$PREFILL_ENABLE_EP
export PREFILL_ENABLE_DP=$PREFILL_ENABLE_DP
export DECODE_TP_SIZE=$DECODE_TP_SIZE
export DECODE_ENABLE_EP=$DECODE_ENABLE_EP
export DECODE_ENABLE_DP=$DECODE_ENABLE_DP
export DECODE_MTP_SIZE=$DECODE_MTP_SIZE
export BENCH_INPUT_LEN=$BENCH_INPUT_LEN
export BENCH_OUTPUT_LEN=$BENCH_OUTPUT_LEN
export BENCH_RANDOM_RANGE_RATIO=$BENCH_RANDOM_RANGE_RATIO
export BENCH_NUM_PROMPTS_MULTIPLIER=$BENCH_NUM_PROMPTS_MULTIPLIER
export BENCH_MAX_CONCURRENCY=$BENCH_MAX_CONCURRENCY

# Sanitize username for use in Docker container names (no special characters)
SANITIZED_USER=$(echo "$USER_NAME" | tr -c 'a-zA-Z0-9_.-' '_')
export DOCKER_CONT_NAME="container_sbatch_${SANITIZED_USER}_${MODEL_NAME}_${SLURM_JOB_ID}"
export RUN_FILE_FULL="$SGL_WS_PATH/${RUN_FILE}"


SELECTED_NODELIST_SRUN=$(echo "$SELECTED_NODES" | paste -sd,)


# =============================================================================
# Cleanup Handler
# =============================================================================
# On INT/TERM/HUP signals, remove any leftover log directories.
cleanup() {
  echo "[${SLURM_JOB_ID}] termination received on $(hostname); cleaning stale logs folder..."
  sudo rm -rf ${SLURM_SUBMIT_DIR}/logs 2>/dev/null || true
  echo "[${SLURM_JOB_ID}] cleanup done."
}

trap cleanup INT TERM HUP


# =============================================================================
# NFS Cache Refresh
# =============================================================================
# Flush stale NFS file handles on all nodes before launching Docker to avoid
# "stale file handle" errors when reading scripts from the NFS-mounted repo.
echo "Refreshing NFS caches on all nodes..."
srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c '
    sync
    # Force re-stat of the mounted directory to refresh NFS handles
    ls -la '"$DI_REPO_DIR"' > /dev/null 2>&1
    stat '"$DI_REPO_DIR"'/scripts/sglang_disagg_server.sh > /dev/null 2>&1
    cat '"$DI_REPO_DIR"'/scripts/sglang_disagg_server.sh > /dev/null 2>&1
    # Drop caches if we have permission (optional, requires root)
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
    echo "NFS cache refreshed on $(hostname)"
'

# =============================================================================
# Main Docker Launch (one container per node, via srun)
# =============================================================================
# Each node gets a Docker container with:
#   - Full GPU access (ROCm /dev/dri, /dev/kfd)
#   - InfiniBand devices (uverbs0-7) for MORI RDMA KV-cache transfers
#   - Host networking and IPC namespace for low-latency communication
#   - Bind mounts: model weights, SSH keys, nicctl, NFS, SGLang repo (optional)
#   - All env vars forwarded so sglang_disagg_server.sh can configure itself
# NODE_RANK is set from SLURM_PROCID so each node knows its role.
srun \
  --nodelist="$SELECTED_NODELIST_SRUN" \
  --kill-on-bad-exit=1 \
  --signal=TERM@30 \
  --unbuffered \
  bash -lc "
set -euo pipefail


echo \"Rank \$SLURM_PROCID on \$(hostname)\"

# Remove any leftover containers from previous runs (idempotent)
sudo docker ps -aq --filter \"name=^container_sbatch_\" | xargs -r sudo docker rm -f || true
if [[ $DRY_RUN -eq 1 ]]; then
    echo \"Dry run, do not stop other dockers\"
else
    sudo docker ps -aq | xargs -r sudo docker stop || true
fi

exec sudo docker run --rm \
    --init \
    --stop-timeout 10 \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --device=/dev/infiniband/rdma_cm \
    --device=/dev/infiniband/uverbs0 \
    --device=/dev/infiniband/uverbs1 \
    --device=/dev/infiniband/uverbs2 \
    --device=/dev/infiniband/uverbs3 \
    --device=/dev/infiniband/uverbs4 \
    --device=/dev/infiniband/uverbs5 \
    --device=/dev/infiniband/uverbs6 \
    --device=/dev/infiniband/uverbs7 \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v ${MODEL_DIR}:/models \
    -v \$HOME/.ssh:/root/.ssh \
    -v $(which nicctl):/usr/sbin/nicctl \
    -v /mnt/nfs/:/mnt/nfs/ -v /opt/amd:/opt/amd \
    ${SGLANG_MAP} \
    --shm-size 128G \
    -v /tmp:/run_logs \
    -v ${DI_REPO_DIR}:${SGL_WS_PATH} \
    -e SLURM_JOB_ID=\$SLURM_JOB_ID \
    -e SLURM_JOB_NODELIST=\$SLURM_JOB_NODELIST \
    -e NNODES=\$NNODES \
    -e NODE_RANK=\$SLURM_PROCID \
    -e NODE0_ADDR=\$NODE0_ADDR \
    -e MODEL_DIR=/models \
    -e LOAD_DUMMY=\$LOAD_DUMMY \
    -e SGL_WS_PATH=${SGL_WS_PATH} \
    -e xP=\$xP \
    -e yD=\$yD \
    -e DRY_RUN=\$DRY_RUN \
    -e MODEL_NAME=\$MODEL_NAME \
    -e IPADDRS=\$IPADDRS \
    -e PREFILL_TP_SIZE=\$PREFILL_TP_SIZE \
    -e PREFILL_ENABLE_EP=\$PREFILL_ENABLE_EP \
    -e PREFILL_ENABLE_DP=\$PREFILL_ENABLE_DP \
    -e DECODE_TP_SIZE=\$DECODE_TP_SIZE \
    -e DECODE_ENABLE_EP=\$DECODE_ENABLE_EP \
    -e DECODE_ENABLE_DP=\$DECODE_ENABLE_DP \
    -e DECODE_MTP_SIZE=\$DECODE_MTP_SIZE \
    -e BENCH_INPUT_LEN=\$BENCH_INPUT_LEN \
    -e BENCH_OUTPUT_LEN=\$BENCH_OUTPUT_LEN \
    -e BENCH_RANDOM_RANGE_RATIO=\$BENCH_RANDOM_RANGE_RATIO \
    -e BENCH_NUM_PROMPTS_MULTIPLIER=\$BENCH_NUM_PROMPTS_MULTIPLIER \
    -e BENCH_MAX_CONCURRENCY=\$BENCH_MAX_CONCURRENCY \
    --name \"$DOCKER_CONT_NAME\" \
    \"$DOCKER_IMAGE_NAME\" bash -lc '
        mkdir -p /run_logs/slurm_job-'\"\$SLURM_JOB_ID\"'
        '"$RUN_FILE_FULL"' 2>&1 | tee /run_logs/slurm_job-'\"\$SLURM_JOB_ID\"'/pd_sglang_bench_serving.sh_NODE'\"\$SLURM_PROCID\"'.log
    '

DOCKER_EXIT_CODE=\$?
if [[ \$DOCKER_EXIT_CODE -ne 0 ]]; then
  echo \"ERROR: docker exited rc=\$DOCKER_EXIT_CODE on \$(hostname)\"
  exit \$DOCKER_EXIT_CODE
fi
"

# Final cleanup: remove named containers from all nodes
srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c 'sudo docker rm -f $DOCKER_CONT_NAME 2>/dev/null || true'
