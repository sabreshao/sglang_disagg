#!/bin/bash
# =============================================================================
# sglang_disagg_server.sh - Per-Node Server Launcher (runs inside Docker)
# =============================================================================
# This script runs inside the Docker container on EACH allocated node.
# It is called by run_xPyD_models.slurm via srun, and its role differs based
# on the node's rank:
#
#   NODE_RANK == 0:
#     - Starts the HEAD prefill server (TP rank 0 of prefill worker 0)
#     - Waits for ALL prefill and decode servers to be ready
#     - Starts the SGLang router (PD-disaggregation mode)
#     - Runs the benchmark (bench3.sh)
#     - Copies results to shared NFS and kills servers when done
#
#   0 < NODE_RANK < xP*PREFILL_NODES_PER_WORKER:
#     - Starts a non-head prefill server (additional TP ranks or workers)
#     - Waits for the router to come up, then idles until router closes
#
#   NODE_RANK >= xP*PREFILL_NODES_PER_WORKER:
#     - Starts a decode server
#     - Waits for the router to come up, then idles until router closes
#     - RANK-0 decode node also parses decode logs for TPOT/throughput metrics
#
# All synchronization across nodes uses socket_barrier.py (TCP polling).
# =============================================================================

# =============================================================================
# Environment Configuration
# (These are passed in from run_xPyD_models.slurm via Docker -e flags.
#  Defaults below are only for standalone testing.)
# =============================================================================

NODE0_ADDR="${NODE0_ADDR:-localhost}"    # IP address of node rank 0 (head node)
NODE_RANK="${NODE_RANK:-0}"             # This node's rank in the SLURM allocation
MODEL_DIR="${MODEL_DIR:-}"              # Base path where model weights are mounted (/models)
MODEL_NAME="${MODEL_NAME:-}"           # Model subdirectory name under MODEL_DIR

xP="${xP:-1}"  # Number of prefill worker groups
yD="${yD:-1}"  # Number of decode worker groups

IPADDRS="${IPADDRS:-localhost}"        # Comma-separated list of all node IPs
HEADNODE_PORT="${HEADNODE_PORT:-20000}" # Port used for multi-node TP dist-init

# Parallelism Configuration
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"       # Tensor Parallelism size for prefill
PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP:-true}" # Enable Expert Parallelism for prefill
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-true}" # Enable Data Parallelism for prefill
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"          # Tensor Parallelism size for decode
DECODE_ENABLE_EP="${DECODE_ENABLE_EP:-true}"   # Enable Expert Parallelism for decode
DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-true}"   # Enable Data Parallelism for decode
DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"        # Speculative decoding steps (0=disabled)

# Benchmark Configuration
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"               # Input sequence length (tokens)
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"             # Output sequence length (tokens)
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-1}" # Variance ratio for seq lengths
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"           # Request rate (inf = unlimited)
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}" # prompts = concurrency * this
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-512}"    # Max concurrent benchmark requests
LOAD_DUMMY="${LOAD_DUMMY:-"0"}"  # 1 = skip loading real weights (smoke test mode)

# Dry Run for debugging purpose (1 = print commands only, do not execute)
DRY_RUN="${DRY_RUN:-0}"


# =============================================================================
# Dependencies and Environment Setup
# =============================================================================
# set_env_vars.sh detects the node hostname and sets:
#   IBDEVICES         - comma-separated list of InfiniBand/RDMA devices
#   GLOO_SOCKET_IFNAME / NCCL_SOCKET_IFNAME - network interface for comms
#   SGLANG_HOST_IP    - this node's IP on the data-plane network
#   SGLANG_USE_AITER  - enable AITER attention backend
#   MORI_SHMEM_MODE   - shared memory isolation mode for MORI RDMA
#   MORI_MAX_DISPATCH_TOKENS_PREFILL/DECODE - max tokens dispatched per rank
#   MORI_RDMA_TC / MORI_RDMA_SL - RDMA traffic class and service level from QoS
source $SGL_WS_PATH/set_env_vars.sh

host_ip=$SGLANG_HOST_IP
host_name=$(hostname)

# Validate MORI_RDMA_TC is set and matches the expected value for this cluster.
# MORI_RDMA_TC=104 is for "mia1*" nodes; MORI_RDMA_TC=96 for GPU*/smci*/node* nodes.
if [[ -n "${MORI_RDMA_TC}" ]]; then
    echo "MORI_RDMA_TC is set to: $MORI_RDMA_TC"

    if [[ "$MORI_RDMA_TC" -eq 104 ]]; then
        if [[ "$host_name" != mia1* ]]; then
            echo "ERROR: MORI_RDMA_TC=104 should be applied on Node with prefix 'mia' but Host '$host_name' does not comply "
            exit 1
        fi
        echo "Host '$host_name' has been configured with MORI_RDMA_TC=104"
    elif [[ "$MORI_RDMA_TC" -eq 96 ]]; then
        if [[ "$host_name" == GPU* || "$host_name" == smci355-ccs-aus* || "$host_name" == node* ]]; then
            echo "MORI_RDMA_TC compliance check pass.. "
        else
            echo "ERROR: MORI_RDMA_TC=96 should be applied on Node with prefix 'GPU' or 'smci355-ccs-aus' but Host '$host_name' does not comply "
            exit 1
        fi
        echo "Host '$host_name' has been configured with MORI_RDMA_TC=96"
    else
        echo "ERROR: MORI_RDMA_TC=$MORI_RDMA_TC should be either 104 or 96. Please apply the recommended QoS/DSCP configs."
        exit 1
    fi
else
    echo "ERROR: MORI_RDMA_TC is not set. "
    exit 1
fi

# =============================================================================
# Update sglang repo
# =============================================================================
# Pull the correct SGLang branch inside the container before launching servers.
cd /sgl-workspace/sglang
git remote add new-repo https://github.com/alexsun07/sglang.git
git fetch new-repo amd_mori_0208
git reset --hard new-repo/amd_mori_0208


# =============================================================================
# Model-Specific Configuration Maps
# =============================================================================
# Each associative array maps a MODEL_NAME to a string of SGLang server flags.
# The final server command is assembled by build_server_config() by combining:
#   BASE_CONFIG + MTP_CONFIG (if MTP>0) + DP_CONFIG (if DP enabled) + SPECIFIC_CONFIG

# Common flags shared by both prefill and decode for each model
declare -A MODEL_BASE_CONFIGS=(
    ["DeepSeek-V3"]="--decode-log-interval 1 --watchdog-timeout 3600 --ep-dispatch-algorithm static --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --disaggregation-transfer-backend mori"
    ["DeepSeek-V3-0324"]="--decode-log-interval 1 --watchdog-timeout 3600 --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --disaggregation-transfer-backend mori"
    ["DeepSeek-R1"]="--decode-log-interval 1 --watchdog-timeout 3600 --ep-dispatch-algorithm static --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --disaggregation-transfer-backend mori"
    ["DeepSeek-R1-0528-MXFP4-Preview"]="--decode-log-interval 1 --watchdog-timeout 3600 --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter"
    ["Qwen3-235B"]="--decode-log-interval 1 --watchdog-timeout 3600 --ep-dispatch-algorithm static --load-balance-method round_robin --attention-backend aiter --disaggregation-transfer-backend mori"
)


# MTP (Multi-Token Prediction / speculative decoding) flags — only applied to
# the decode server when DECODE_MTP_SIZE > 0.
if [[ "$DECODE_MTP_SIZE" =~ ^[0-9]+$ ]] && [[ "$DECODE_MTP_SIZE" -gt 0 ]]; then
    declare -A MODEL_MTP_CONFIGS=(
        ["DeepSeek-V3"]="--speculative-algorithm NEXTN --speculative-num-steps ${DECODE_MTP_SIZE} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((DECODE_MTP_SIZE + 1))"
        ["DeepSeek-V3-0324"]="--speculative-algorithm NEXTN --speculative-num-steps ${DECODE_MTP_SIZE} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((DECODE_MTP_SIZE + 1))"
        ["DeepSeek-R1"]="--speculative-algorithm NEXTN --speculative-num-steps ${DECODE_MTP_SIZE} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((DECODE_MTP_SIZE + 1))"
        ["DeepSeek-R1-0528-MXFP4-Preview"]="--speculative-algorithm NEXTN --speculative-num-steps ${DECODE_MTP_SIZE} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((DECODE_MTP_SIZE + 1))"
	["Qwen3-235B"]="--speculative-algorithm NEXTN --speculative-num-steps ${DECODE_MTP_SIZE} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((DECODE_MTP_SIZE + 1))"
    )
fi

# DP-specific flags — applied when Data Parallelism is enabled.
# Enables MORI all-to-all backend and DP-aware attention/LM-head.
declare -A MODEL_DP_CONFIGS=(
    ["DeepSeek-V3"]="--moe-a2a-backend mori --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head"
    ["DeepSeek-V3-0324"]="--moe-a2a-backend mori --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head"
    ["DeepSeek-R1"]="--moe-a2a-backend mori --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head"
    ["DeepSeek-R1-0528-MXFP4-Preview"]="--moe-a2a-backend mori --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head"
    ["Qwen3-235B"]="--moe-a2a-backend mori --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head"
)


# Prefill-specific flags per model.
# When DP is enabled, cuda-graph batch sizes are kept small (1-3) because
# MORI dispatch tokens limit the effective batch size per rank. The
# chunked-prefill-size is also derived from MORI_MAX_DISPATCH_TOKENS_PREFILL.
if [[ "$PREFILL_ENABLE_DP" == "true" ]]; then
    prefill_cuda_graph_bs=($(seq 1 3))
    prefill_max_running_requests=128
    prefill_chunked_prefill_size=$((MORI_MAX_DISPATCH_TOKENS_PREFILL * PREFILL_TP_SIZE))
else
    prefill_cuda_graph_bs=($(seq 1 128))
    prefill_max_running_requests=128
    prefill_chunked_prefill_size=262144
fi

# Skip model loading if requested (smoke-test / DRY_RUN support)
if [[ "$LOAD_DUMMY" == "1" ]]; then
    LOAD_DUMMY_MODEL="--load-format dummy"
    echo "skip model loading"
else
    LOAD_DUMMY_MODEL=""
fi

declare -A MODEL_PREFILL_CONFIGS=(
    ["DeepSeek-V3"]="--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size} --cuda-graph-bs ${prefill_cuda_graph_bs[*]} --disable-radix-cache"
    ["DeepSeek-V3-0324"]="--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size} --cuda-graph-bs ${prefill_cuda_graph_bs[*]} --disable-radix-cache"
    ["DeepSeek-R1"]="--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size} --cuda-graph-bs ${prefill_cuda_graph_bs[*]} --disable-radix-cache"
    ["DeepSeek-R1-0528-MXFP4-Preview"]="--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size 16384  --cuda-graph-bs ${prefill_cuda_graph_bs[*]} --disable-radix-cache"
    ["Qwen3-235B"]="--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size 16384  --cuda-graph-bs ${prefill_cuda_graph_bs[*]} --disable-radix-cache"
)

# Decode-specific flags per model.
# When DP is enabled, cuda-graph batch sizes are larger (1-132) to support
# high concurrency. chunked-prefill-size for decode is derived from
# MORI_MAX_DISPATCH_TOKENS_DECODE.
# FIXME(billishyahao): chunked-prefill-size for decode nodes is a workaround;
# it will be eliminated in a future SGLang release.
if [[ "$DECODE_ENABLE_DP" == "true" ]]; then
    decode_cuda_graph_bs=($(seq 1 132))
    decode_max_running_requests=4096
    decode_chunked_prefill_size=$((MORI_MAX_DISPATCH_TOKENS_DECODE * DECODE_TP_SIZE))
else
    decode_cuda_graph_bs=($(seq 1 32))
    decode_max_running_requests=256
    decode_chunked_prefill_size=262144
fi

declare -A MODEL_DECODE_CONFIGS=(
    ["DeepSeek-V3"]="--mem-fraction-static 0.85 --max-running-requests ${decode_max_running_requests} --chunked-prefill-size ${decode_chunked_prefill_size} --cuda-graph-bs ${decode_cuda_graph_bs[*]} --prefill-round-robin-balance"
    ["DeepSeek-V3-0324"]="--mem-fraction-static 0.85 --max-running-requests ${decode_max_running_requests} --chunked-prefill-size ${decode_chunked_prefill_size} --cuda-graph-bs ${decode_cuda_graph_bs[*]} --prefill-round-robin-balance"
    ["DeepSeek-R1"]="--mem-fraction-static 0.85 --max-running-requests ${decode_max_running_requests} --chunked-prefill-size ${decode_chunked_prefill_size} --cuda-graph-bs ${decode_cuda_graph_bs[*]} --prefill-round-robin-balance"
    ["DeepSeek-R1-0528-MXFP4-Preview"]="--mem-fraction-static 0.85 --max-running-requests ${decode_max_running_requests} --chunked-prefill-size ${decode_chunked_prefill_size} --cuda-graph-bs ${decode_cuda_graph_bs[*]} --prefill-round-robin-balance"
    ["Qwen3-235B"]="--mem-fraction-static 0.85 --max-running-requests ${decode_max_running_requests} --chunked-prefill-size ${decode_chunked_prefill_size} --cuda-graph-bs ${decode_cuda_graph_bs[*]} --prefill-round-robin-balance"
)


# =============================================================================
# Cluster Topology Configuration
# =============================================================================
# Parse comma-separated IP list into an array for indexed access.
IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

# Calculate how many physical nodes each worker occupies.
# A worker with TP > 8 spans multiple nodes (e.g. TP=16 => 2 nodes per worker).
PREFILL_NODES_PER_WORKER=$(((PREFILL_TP_SIZE + 7) / 8))
DECODE_NODES_PER_WORKER=$(((DECODE_TP_SIZE + 7) / 8))

# NODE_OFFSET is the first node index that belongs to the decode workers.
NODE_OFFSET=$((PREFILL_NODES_PER_WORKER * xP))

# Build --prefill URL arguments for the router (one per prefill worker group).
PREFILL_HEADNODE_URLS=()
PREFILL_ARGS=""
for i in $(seq 0 $((xP - 1))); do
    prefill_idx=$((i * PREFILL_NODES_PER_WORKER))
    PREFILL_HEADNODE_URLS[$i]="${IP_ARRAY[$prefill_idx]}:${HEADNODE_PORT}"
    PREFILL_ARGS="$PREFILL_ARGS --prefill http://${IP_ARRAY[$prefill_idx]}:8000"
done

# Build --decode URL arguments for the router (one per decode worker group).
DECODE_HEADNODE_URLS=()
DECODE_ARGS=""
for i in $(seq 0 $((yD - 1))); do
    decode_idx=$((i * DECODE_NODES_PER_WORKER + NODE_OFFSET))
    DECODE_HEADNODE_URLS[$i]="${IP_ARRAY[$decode_idx]}:${HEADNODE_PORT}"
    DECODE_ARGS="$DECODE_ARGS --decode http://${IP_ARRAY[$decode_idx]}:8000"
done

echo "Prefill worker headnode list: ${PREFILL_HEADNODE_URLS[@]}"
echo "Decode  worker headnode list: ${DECODE_HEADNODE_URLS[@]}"

# =============================================================================
# Configuration Builder Functions
# =============================================================================

# build_server_config: Assembles the complete SGLang server flag string for
# either "prefill" or "decode" mode by combining parallelism args with
# model-specific base/MTP/DP/mode-specific config maps.
#
# Arguments:
#   $1 mode            - "prefill" or "decode"
#   $2 model_name      - key into MODEL_*_CONFIGS maps
#   $3 tp_size         - tensor parallelism size
#   $4 enable_ep       - "true"/"false"
#   $5 enable_dp       - "true"/"false"
#   $6 decode_mtp_size - MTP steps (0 = disabled)
build_server_config() {
    local mode="$1"
    local model_name="$2"
    local tp_size="$3"
    local enable_ep="$4"
    local enable_dp="$5"
    local decode_mtp_size="$6"

    # Calculate EP and DP sizes based on enable flags
    local ep_size=1
    local dp_size=1

    if [[ "$enable_ep" == "true" ]]; then
        ep_size=$tp_size
    fi

    if [[ "$enable_dp" == "true" ]]; then
        dp_size=$tp_size
    fi

    # Build parallelism arguments
    local parallel_args="--tp-size ${tp_size}"

    if [[ "$enable_ep" == "true" ]]; then
        parallel_args="$parallel_args --ep-size ${ep_size}"
    fi

    if [[ "$enable_dp" == "true" ]]; then
        parallel_args="$parallel_args --dp-size ${dp_size}"
    fi

    # Look up model-specific configuration strings
    local base_config=""
    local mtp_config=""
    local dp_config=""
    local specific_config=""

    if [[ -n "$model_name" ]]; then
        if [[ -n "${MODEL_BASE_CONFIGS[$model_name]}" ]]; then
            base_config="${MODEL_BASE_CONFIGS[$model_name]}"
        fi

        # MTP config is only used on the decode server
        if [ "$decode_mtp_size" -gt 0 ] && [[ -n "${MODEL_MTP_CONFIGS[$model_name]}" ]]; then
            mtp_config="${MODEL_MTP_CONFIGS[$model_name]}"
        fi

        if [[ "$enable_dp" == "true" ]] && [[ -n "${MODEL_DP_CONFIGS[$model_name]}" ]]; then
            dp_config="${MODEL_DP_CONFIGS[$model_name]}"
        fi

        if [[ "$mode" == "prefill" ]]; then
            if [[ -n "${MODEL_PREFILL_CONFIGS[$model_name]}" ]]; then
                specific_config="${MODEL_PREFILL_CONFIGS[$model_name]}"
            fi
        elif [[ "$mode" == "decode" ]]; then
            if [[ -n "${MODEL_DECODE_CONFIGS[$model_name]}" ]]; then
                specific_config="${MODEL_DECODE_CONFIGS[$model_name]}"
            fi
        fi
    fi

    # Combine: parallel args + base config + mtp config (decode only) + dp config + mode config
    local full_config="$parallel_args"
    if [[ -n "$base_config" ]]; then
        full_config="$full_config $base_config"
    fi
    if [[ -n "$mtp_config" ]] && [[ "$mode" == "decode" ]]; then
        full_config="$full_config $mtp_config"
    fi
    if [[ -n "$dp_config" ]]; then
        full_config="$full_config $dp_config"
    fi
    if [[ -n "$specific_config" ]]; then
        full_config="$full_config $specific_config"
    fi

    echo "$full_config"
}

# Build complete server configurations for prefill and decode
PREFILL_SERVER_CONFIG=$(build_server_config "prefill" "$MODEL_NAME" "$PREFILL_TP_SIZE" "$PREFILL_ENABLE_EP" "$PREFILL_ENABLE_DP" "$DECODE_MTP_SIZE")
DECODE_SERVER_CONFIG=$(build_server_config "decode" "$MODEL_NAME" "$DECODE_TP_SIZE" "$DECODE_ENABLE_EP" "$DECODE_ENABLE_DP" "$DECODE_MTP_SIZE")

if [[ -n "$MODEL_NAME" ]]; then
    echo "Using model-specific configuration for: $MODEL_NAME"
fi

# =============================================================================
# Container Synchronization
# =============================================================================
# All nodes wait at this barrier to ensure every Docker container has started
# before any node proceeds to launch a server. Timeout: 300 seconds.
echo "Waiting at the container creation barrier on $host_name"
python $SGL_WS_PATH/socket_barrier.py \
    --local-ip ${host_ip} \
    --local-port 5000 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 5000 \
    --wait-for-all-ports \
    --timeout 300


# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================

if [ "$NODE_RANK" -eq 0 ]; then
    # -------------------------------------------------------------------------
    # NODE RANK 0: Head node — prefill server + router + benchmark
    # -------------------------------------------------------------------------
    echo "NODE INFO ======================================="
    echo "================================================"
    echo "Node List : ${SLURM_JOB_NODELIST}"
    echo "Node IPs : ${IPADDRS}"
    echo "Model Name : ${MODEL_NAME:-'Not specified'}"
    echo "================================================"

    echo "CLUSTER INFO ===================================="
    echo "================================================"
    echo "${host_name}:${host_ip} is Proxy Node and Prefill Node"
    echo "Using prefill config: $PREFILL_MODEL_CONFIG"
    echo "Prefill parallelism: TP=${PREFILL_TP_SIZE}, EP enabled: ${PREFILL_ENABLE_EP}, DP enabled: ${PREFILL_ENABLE_DP}, MTP size=${DECODE_MTP_SIZE}"
    echo "Decode  parallelism: TP=${DECODE_TP_SIZE},  EP enabled: ${DECODE_ENABLE_EP},  DP enabled: ${DECODE_ENABLE_DP},  MTP size=${DECODE_MTP_SIZE}"
    echo "Prefill servers ($((PREFILL_TP_SIZE/8)) nodes): ${PREFILL_ARGS}"
    echo "Decode servers  ($((DECODE_TP_SIZE/8))  nodes): ${DECODE_ARGS}"
    echo "Prefill env: SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK: ${MORI_MAX_DISPATCH_TOKENS_PREFILL}"
    echo "Decode env: SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_DECODE}"
    echo "================================================"

    # Launch the head prefill server in the background.
    # SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK controls the maximum number
    # of tokens MORI dispatches per TP rank per step on the prefill side.
    PREFILL_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_PREFILL} python3 -m sglang.launch_server \
        --model-path $MODEL_DIR/$MODEL_NAME \
        ${LOAD_DUMMY_MODEL} \
        --disaggregation-mode prefill \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        ${PREFILL_SERVER_CONFIG}"

    # If a single prefill worker spans multiple nodes (TP > 8), add dist-init args.
    if [ "$PREFILL_NODES_PER_WORKER" -gt 1 ]; then
        PREFILL_CMD="$PREFILL_CMD --dist-init-addr ${PREFILL_HEADNODE_URLS[0]} --nnodes ${PREFILL_NODES_PER_WORKER} --node-rank 0"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        set -x
        eval "$PREFILL_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log >/dev/null &
        set +x
        prefill0_pid=$!
    fi


    echo "Waiting for all prefill and decode servers to be up . . ."

    # Wait for ALL node IPs to open port 8000 (server ready signal).
    # Timeout: 1800 seconds (30 minutes) to allow model loading time.
    BARRIER_CMD="python $SGL_WS_PATH/socket_barrier.py \
        --node-ips ${IPADDRS} \
        --node-ports 8000 \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi
    echo "Congratulations!!! All prefill and decode servers are up . . ."

    # Launch the SGLang PD-disaggregation router on port 30000.
    # The router load-balances requests across prefill and decode workers.
    ROUTER_CMD="python -m sglang_router.launch_router \
        --pd-disaggregation \
        --mini-lb \
        --port 30000 \
        --policy random \
        --prefill-policy random \
        --decode-policy random  \
        ${PREFILL_ARGS} \
        ${DECODE_ARGS}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $ROUTER_CMD"
    else
        set -x
        eval "$ROUTER_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null &
        proxy_pid=$!
        set +x

        # Wait for the router's /health endpoint to return 200 before starting benchmark.
        BARRIER_CMD="python $SGL_WS_PATH/socket_barrier.py \
        --node-ips ${NODE0_ADDR} \
        --node-ports 30000 \
        --wait-for-all-health \
        --health-endpoint /health \
        --timeout 1800"

        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "DRY RUN: $BARRIER_CMD"
        else
            eval "$BARRIER_CMD"
        fi

        echo "Router is ready for benchmarking"
    fi


    echo "Ready for benchmarking on ${host_name}:${host_ip}"

    echo "Benchmarking on ${host_name}:${host_ip}"
    cd /sglang_disagg

    # Export IS_MTP flag so bench3.sh knows whether speculative decoding is active
    if [ "$DECODE_MTP_SIZE" -gt 0 ]; then
        export IS_MTP=true
    else
        export IS_MTP=false
    fi
    # Head decode node IP used by bench3.sh for slow_down coordination
    export DECODE_HEAD_NODE=${IP_ARRAY[$NODE_OFFSET]}

    # Run benchmark: arguments are positional (see bench3.sh for details)
    # n_prefill n_decode prefill_gpus decode_gpus model_dir model_name log_path isl osl concurrency_list req_rate random_range_ratio num_prompts_multiplier
    BENCH_CMD="bash /sglang_disagg/bench3.sh ${xP} ${yD} $((PREFILL_TP_SIZE*xP)) $((DECODE_TP_SIZE*yD)) \
        $MODEL_DIR $MODEL_NAME /run_logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} \
        ${BENCH_OUTPUT_LEN} "${BENCH_MAX_CONCURRENCY}" ${BENCH_REQUEST_RATE} \
        ${BENCH_RANDOM_RANGE_RATIO} ${BENCH_NUM_PROMPTS_MULTIPLIER}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BENCH_CMD"
    else
        set -x
        eval "$BENCH_CMD"
        set +x
    fi

    # Ensure persistent log directory exists on the shared NFS filesystem
    if [ ! -d /sglang_disagg/logs ]; then
        mkdir -p /sglang_disagg/logs
        echo "Created directory: /sglang_disagg/logs"
    fi

    if [ ! -d /sglang_disagg/logs/slurm_job-${SLURM_JOB_ID} ]; then
        mkdir -p /sglang_disagg/logs/slurm_job-${SLURM_JOB_ID}
        echo "Created directory: /sglang_disagg/logs/slurm_job-${SLURM_JOB_ID}"
    fi

    # Copy logs from the ephemeral /tmp volume to persistent NFS storage
    if [[ "$DRY_RUN" -eq 0 ]]; then
        cp /run_logs/slurm_job-${SLURM_JOB_ID}/* /sglang_disagg/logs/slurm_job-${SLURM_JOB_ID}/
    fi

    echo "Killing the proxy server and prefill server"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        kill $proxy_pid
        kill $prefill0_pid
    fi

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -lt "$NODE_OFFSET" ]; then
    # -------------------------------------------------------------------------
    # NODE RANK 1 .. (xP*PREFILL_NODES_PER_WORKER - 1): Additional prefill nodes
    # These are either additional nodes for a multi-node prefill worker (TP>8)
    # or additional prefill workers (xP>1).
    # -------------------------------------------------------------------------
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using prefill config: $PREFILL_MODEL_CONFIG"
    echo "Prefill parallelism: TP=${PREFILL_TP_SIZE}, EP enabled: ${PREFILL_ENABLE_EP}, DP enabled: ${PREFILL_ENABLE_DP}"

    PREFILL_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_PREFILL} python3 -m sglang.launch_server \
        --model-path $MODEL_DIR/${MODEL_NAME} \
        ${LOAD_DUMMY_MODEL} \
        --disaggregation-mode prefill \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        ${PREFILL_SERVER_CONFIG}"

    # Compute intra-worker node rank for multi-node TP configurations
    if [ "$PREFILL_NODES_PER_WORKER" -gt 1 ]; then
        rank=$((NODE_RANK % PREFILL_NODES_PER_WORKER))
        prefill_idx=$((NODE_RANK / PREFILL_NODES_PER_WORKER))
        PREFILL_CMD="$PREFILL_CMD --dist-init-addr ${PREFILL_HEADNODE_URLS[0]} --nnodes ${PREFILL_NODES_PER_WORKER} --node-rank $rank"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        set -x
        eval "$PREFILL_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log >/dev/null &
        set +x
        prefill_pid=$!
    fi

    # Wait for the router on node 0 port 30000 to come up (benchmark running)
    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python $SGL_WS_PATH/socket_barrier.py \
        --node-ips ${NODE0_ADDR} \
        --node-ports 30000 \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi

    # Keep this node alive until the router closes (benchmark finished)
    echo "Waiting until proxy server closes..."
    WAIT_CMD="python $SGL_WS_PATH/socket_wait.py \
        --remote-ip ${NODE0_ADDR} \
        --remote-port 30000"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the rank $NODE_RANK prefill server"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        kill $prefill_pid
    fi

else
    # -------------------------------------------------------------------------
    # NODE RANK >= xP*PREFILL_NODES_PER_WORKER: Decode nodes
    # RANK is the decode-local rank (0-indexed within all decode nodes).
    # -------------------------------------------------------------------------
    RANK=$((NODE_RANK - xP * PREFILL_NODES_PER_WORKER))
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using decode config: $DECODE_MODEL_CONFIG"
    echo "Decode node rank: $RANK"
    echo "Decode parallelism: TP=${DECODE_TP_SIZE}, EP enabled: ${DECODE_ENABLE_EP}, DP enabled: ${DECODE_ENABLE_DP}"

    # SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK limits decode-side dispatch
    DECODE_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_DECODE} python3 -m sglang.launch_server \
        --model-path ${MODEL_DIR}/${MODEL_NAME} \
        ${LOAD_DUMMY_MODEL} \
        --disaggregation-mode decode \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        ${DECODE_SERVER_CONFIG}"

    # For multi-node decode workers (TP > 8), add dist-init coordination args
    if [ "$DECODE_NODES_PER_WORKER" -gt 1 ]; then
        rank=$((RANK % DECODE_NODES_PER_WORKER))
        decode_idx=$((RANK / DECODE_NODES_PER_WORKER))
        DECODE_CMD="$DECODE_CMD --dist-init-addr ${DECODE_HEADNODE_URLS[$decode_idx]} --nnodes ${DECODE_NODES_PER_WORKER} --node-rank $rank"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $DECODE_CMD"
    else
        set -x
        eval "$DECODE_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/decode_NODE${NODE_RANK}.log >/dev/null &

        set +x
        decode_pid=$!
    fi

    # Wait for the router on node 0 to come up
    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python $SGL_WS_PATH/socket_barrier.py \
        --node-ips ${NODE0_ADDR} \
        --node-ports 30000 \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi

    # Wait until the router closes (benchmark finished on node 0)
    echo "Waiting until proxy server closes..."
    WAIT_CMD="python $SGL_WS_PATH/socket_wait.py \
        --remote-ip ${NODE0_ADDR} \
        --remote-port 30000"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the rank $RANK decode server"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        kill $decode_pid
    fi

    # The head decode node (RANK=0) also copies logs and reports TPOT metrics
    if [ "$RANK" -eq 0 ]; then
        if [ ! -d /sglang_disagg/logs/slurm_job-${SLURM_JOB_ID} ]; then
            mkdir -p /sglang_disagg/logs/slurm_job-${SLURM_JOB_ID}
            echo "Created directory: /sglang_disagg/logs/slurm_job-${SLURM_JOB_ID}"
        fi

        # Copy decode logs to persistent NFS storage
        if [[ "$DRY_RUN" -eq 0 ]]; then
            cp /run_logs/slurm_job-${SLURM_JOB_ID}/* /sglang_disagg/logs/slurm_job-${SLURM_JOB_ID}/
        fi

        # Parse decode server log to extract TPOT and output throughput metrics
        result=$(python $SGL_WS_PATH/parse_decode_log.py "/run_logs/slurm_job-${SLURM_JOB_ID}/decode_NODE${NODE_RANK}.log" "$BENCH_OUTPUT_LEN" "$BENCH_MAX_CONCURRENCY")
        tpot=$(echo "$result" | sed -n '1p')
        output_throughput=$(echo "$result" | sed -n '2p')

        echo "Batch size = $BENCH_MAX_CONCURRENCY"
        echo "Input len  = $BENCH_INPUT_LEN"
        echo "Output len = $BENCH_OUTPUT_LEN"
        echo "TPOT       = $tpot ms"
        echo "Output throughput = $output_throughput tokens/s"
    fi

fi

echo "Script completed successfully"
exit 0
