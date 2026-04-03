#!/bin/bash

# DeepSeek-V3.2-mxfp4 recipe.
# This is the default config used by `bash run.sh`.
# Use `CONFIG_FILE=./config.deepseek_v32.sh bash run.sh` to switch to another model.

# -----------------------------------------------------------------------------
# Model and image
# -----------------------------------------------------------------------------
# Supported MODEL_NAME values in this file: DeepSeek-V3.2-mxfp4
: "${MODEL_NAME:=DeepSeek-V3.2-mxfp4}"
: "${MODEL_DIR:=/mnt/nfs/RAID/shared/huggingface/hub}"
: "${DOCKER_IMAGE_NAME:=rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260331}"
# LOAD_DUMMY=1 uses --load-format dummy to skip loading real model weights.
: "${LOAD_DUMMY:=1}"

# -----------------------------------------------------------------------------
# Topology
# -----------------------------------------------------------------------------
: "${xP:=1}"
: "${yD:=1}"
: "${GPUS_PER_NODE:=8}"

: "${PREFILL_TP_SIZE:=8}"
: "${PREFILL_ENABLE_EP:=true}"
: "${PREFILL_ENABLE_DP:=false}"

: "${DECODE_TP_SIZE:=8}"
: "${DECODE_ENABLE_EP:=true}"
: "${DECODE_ENABLE_DP:=false}"
: "${DECODE_MTP_SIZE:=0}"

# -----------------------------------------------------------------------------
# Benchmark
# Only slowdown mode is supported in this simplified recipe.
# -----------------------------------------------------------------------------
: "${BENCH_MODE:=slowdown}"
: "${BENCH_POISSON_USE_SLOWDOWN_PHASE_CONTROL:=false}"
: "${BENCH_INPUT_LEN:=6144}"
: "${BENCH_OUTPUT_LEN:=1024}"
: "${BENCH_MAX_CONCURRENCY:=8}"
: "${BENCH_AUTO_CLAMP:=true}"
: "${BENCH_STARTUP_WAIT_SECONDS:=60}"
: "${BENCH_FORWARD_SLEEP_TIME:=90}"
: "${SLOWDOWN_DURATION:=60}"
: "${PREFILL_IDLE_TIMEOUT:=1200}"
: "${PREFILL_IDLE_REQUEST_TIMEOUT:=30}"
: "${PREFILL_IDLE_POLL_INTERVAL:=5}"

# Dataset used by the slowdown benchmark. It will be downloaded automatically
# when DATASET_FILE does not exist.
: "${DATASET_DIR:=/mnt/nfs/minchsun/dataset}"
: "${DATASET_FILE:=${DATASET_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json}"
: "${DATASET_URL:=https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json}"

# -----------------------------------------------------------------------------
# Runtime and logging
# -----------------------------------------------------------------------------
: "${DRY_RUN:=0}"
: "${WORKSPACE_SGLANG_DIR:=}"
: "${HEADNODE_PORT:=20000}"
: "${ROUTE_PROBE_TARGET:=10.2.224.0}"

# -----------------------------------------------------------------------------
# Cluster networking and MORI environment
# Leave *_OVERRIDE empty to auto-detect from the host node.
# -----------------------------------------------------------------------------
: "${IBDEVICES_OVERRIDE:=}"
: "${GLOO_SOCKET_IFNAME_OVERRIDE:=}"
: "${NCCL_SOCKET_IFNAME_OVERRIDE:=}"
: "${SGLANG_HOST_IP_OVERRIDE:=}"
: "${NCCL_IB_HCA_OVERRIDE:=}"
: "${MORI_RDMA_SL_OVERRIDE:=}"
: "${MORI_RDMA_TC_OVERRIDE:=}"

: "${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:=1200}"
: "${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:=1200}"
: "${MORI_SHMEM_MODE:=ISOLATION}"
: "${SGLANG_MORI_FP8_DISP:=True}"
: "${MORI_EP_LAUNCH_CONFIG_MODE:=AUTO}"
: "${MORI_MAX_DISPATCH_TOKENS_PREFILL:=16384}"
: "${MORI_MAX_DISPATCH_TOKENS_DECODE:=320}"
: "${MORI_APP_LOG_LEVEL:=INFO}"

# -----------------------------------------------------------------------------
# Docker customization
# Add custom KEY=VALUE pairs to CONTAINER_ENV_OVERRIDES when the container
# needs extra environment variables.
# Example:
#   CONTAINER_ENV_OVERRIDES=(
#       "PYTHONUNBUFFERED=1"
#       "MY_CUSTOM_ENV=1"
#   )
# -----------------------------------------------------------------------------
CONTAINER_ENV_OVERRIDES=(
    "PYTHONUNBUFFERED=1"
)

# Extra bind mounts in host_path:container_path form.
EXTRA_CONTAINER_MOUNTS=(
    "/mnt/nfs:/mnt/nfs"
    "/opt/amd:/opt/amd"
)

# Extra raw docker arguments appended to `docker run`.
EXTRA_DOCKER_ARGS=()

# -----------------------------------------------------------------------------
# Launch arguments
# Put flags here when you need to add or remove server/router parameters.
# -----------------------------------------------------------------------------
MODEL_BASE_ARGS=(
    --decode-log-interval 1
    --watchdog-timeout 3600
    --load-balance-method round_robin
    --attention-backend nsa
    --nsa-prefill-backend tilelang
    --nsa-decode-backend tilelang
    --disaggregation-transfer-backend mori
    --enable-dp-attention
    --enable-dp-lm-head
    --moe-dense-tp-size 1
    --eplb-algorithm deepseek
    --deepep-mode low_latency
    --disable-radix-cache
    --ep-dispatch-algorithm dynamic
)

MODEL_DP_ARGS=()

: "${PREFILL_MEM_FRACTION_STATIC:=0.8}"
: "${PREFILL_MAX_RUNNING_REQUESTS:=128}"
: "${PREFILL_CHUNKED_PREFILL_SIZE:=262144}"
PREFILL_CUDA_GRAPH_BS=( $(seq 1 128) )

PREFILL_ROLE_ARGS=(
    --mem-fraction-static "${PREFILL_MEM_FRACTION_STATIC}"
    --max-running-requests "${PREFILL_MAX_RUNNING_REQUESTS}"
    --chunked-prefill-size "${PREFILL_CHUNKED_PREFILL_SIZE}"
    --cuda-graph-bs "${PREFILL_CUDA_GRAPH_BS[@]}"
    --disable-radix-cache
)

: "${DECODE_MEM_FRACTION_STATIC:=0.85}"
: "${DECODE_MAX_RUNNING_REQUESTS:=256}"
: "${DECODE_CHUNKED_PREFILL_SIZE:=262144}"
DECODE_CUDA_GRAPH_BS=( 256 )

DECODE_ROLE_ARGS=(
    --mem-fraction-static "${DECODE_MEM_FRACTION_STATIC}"
    --max-running-requests "${DECODE_MAX_RUNNING_REQUESTS}"
    --chunked-prefill-size "${DECODE_CHUNKED_PREFILL_SIZE}"
    --cuda-graph-bs "${DECODE_CUDA_GRAPH_BS[@]}"
    --prefill-round-robin-balance
)

# Extra args appended to the prefill-side `python -m sglang.launch_server`.
# Example:
#   PREFILL_EXTRA_ARGS=(
#       --my_flag 1
#   )
PREFILL_EXTRA_ARGS=()

# Extra args appended to the decode-side `python -m sglang.launch_server`.
# Example:
#   DECODE_EXTRA_ARGS=(
#       --my_flag 1
#   )
DECODE_EXTRA_ARGS=()

# Extra args appended to the router command.
# Example:
#   ROUTER_EXTRA_ARGS=(
#       --policy random
#       --prefill-policy random
#       --decode-policy random
#       --my_router_flag 1
#   )
ROUTER_EXTRA_ARGS=(
    --policy random
    --prefill-policy random
    --decode-policy random
)
