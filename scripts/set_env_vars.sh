#!/bin/bash
# =============================================================================
# set_env_vars.sh - Node-Specific Environment Configuration
# =============================================================================
# Sourced by sglang_disagg_server.sh at the start of each per-node run.
# Detects the node hostname pattern to configure the correct network interfaces
# and RDMA devices for the local cluster environment.
#
# Variables set by this script:
#   IBDEVICES          - Comma-separated list of InfiniBand/RDMA device names
#                        passed to SGLang's --disaggregation-ib-device flag
#   GLOO_SOCKET_IFNAME - Network interface for PyTorch GLOO collective comms
#   NCCL_SOCKET_IFNAME - Network interface for NCCL collective comms
#   SGLANG_HOST_IP     - This node's IP address on the data-plane interface
#   NCCL_IB_HCA        - InfiniBand HCA list for NCCL (set equal to IBDEVICES)
#   SGLANG_USE_AITER   - Enable the AMD AITER attention kernel backend (=1)
#   SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT - Timeout for disagg bootstrap (s)
#   SGLANG_DISAGGREGATION_WAITING_TIMEOUT   - Timeout for disagg waiting (s)
#   MORI_SHMEM_HEAP_SIZE  - MORI shared memory heap size (e.g. 32G)
#   MORI_SHMEM_MODE       - MORI shared memory isolation mode (ISOLATION)
#   SGLANG_MORI_FP8_DISP  - Enable FP8 dispatch in MORI (True/False)
#   MORI_EP_LAUNCH_CONFIG_MODE - MORI expert parallelism launch config (AUTO)
#   MORI_MAX_DISPATCH_TOKENS_PREFILL - Max tokens dispatched per rank (prefill)
#   MORI_MAX_DISPATCH_TOKENS_DECODE  - Max tokens dispatched per rank (decode)
#   MORI_APP_LOG_LEVEL    - MORI application log verbosity (INFO)
#   MORI_RDMA_SL          - RDMA service level derived from QoS config
#   MORI_RDMA_TC          - RDMA traffic class derived from QoS DSCP value
# =============================================================================

# Detect hostname and configure per-cluster network settings.
# Each cluster uses different NIC naming conventions:
#   GPU* / smci355-ccs-aus* : AMD SuperMicro nodes — ionic_* RDMA, default GW NIC
#   node*                   : Generic nodes — ionic_* RDMA, fixed GW NIC
#   mia1*                   : Miami cluster — rdma* devices, default GW NIC
set -x
NODENAME=$(hostname)
if [[ $NODENAME == GPU* ]]; then
    export IBDEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
    export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}')
    export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}')
elif [[ $NODENAME == smci355-ccs-aus* ]]; then
    export IBDEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
    export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}')
    export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}')
elif [[ $NODENAME == node* ]]; then
    #export IBDEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
    export IBDEVICES=rocep9s0,rocep25s0,rocep105s0,rocep121s0,rocep137s0,rocep153s0,rocep233s0,rocep249s0
    export GLOO_SOCKET_IFNAME=enp193s0f1np1
    export NCCL_SOCKET_IFNAME=enp193s0f1np1
elif [[ $NODENAME == mia1* ]]; then
    export IBDEVICES=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
    export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
    export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
else
    echo "[Error] unable to fetch the hostname"
    exit 1
fi
# Derive the node's primary IP from the configured interface
export SGLANG_HOST_IP=$(ip -4 addr show dev ${GLOO_SOCKET_IFNAME} | awk '/inet / {print $2}' | cut -d/ -f1)
set +x


# NCCL InfiniBand HCA list — mirrors IBDEVICES for NCCL collective operations
export NCCL_IB_HCA=$IBDEVICES

#export AITER_ONLINE_TUNE=1
# Enable AMD AITER attention backend for optimized attention kernels
#export SGLANG_USE_AITER=1

# Disaggregation timeouts — allow up to 1200s for KV-cache bootstrap and transfer
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200

# MORI shared memory configuration
# MORI_SHMEM_HEAP_SIZE: total RDMA-registered shared memory pool size
# MORI_SHMEM_MODE=ISOLATION: each process gets its own isolated shmem region
# export MORI_SHMEM_HEAP_SIZE=32G
export MORI_SHMEM_MODE=ISOLATION

# Enable FP8 dispatch in MORI for reduced memory bandwidth during KV transfer
export SGLANG_MORI_FP8_DISP=True

# MORI expert parallelism launch configuration (AUTO = let MORI decide)
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO

# Maximum tokens MORI dispatches per TP rank per step.
# Prefill side allows larger chunks (16384) for throughput.
# Decode side uses smaller chunks (320) to keep batch latency low.
export MORI_MAX_DISPATCH_TOKENS_PREFILL=16384
export MORI_MAX_DISPATCH_TOKENS_DECODE=320

export MORI_APP_LOG_LEVEL=INFO

# =============================================================================
# RDMA QoS Configuration (from nicctl)
# =============================================================================
# Read the no-drop priority and its DSCP value from the NIC's QoS config,
# then compute MORI_RDMA_TC = 4 * DSCP (hardware traffic class encoding).
# MORI_RDMA_SL (service level) and MORI_RDMA_TC (traffic class) are used by
# MORI to send RDMA transfers with the correct priority/DSCP marking.
ND_PRIO=$(nicctl show qos  2>/dev/null | awk '/PFC no-drop priorities/ {print $NF; exit}')
ND_DSCP=$(nicctl show qos 2>/dev/null| awk -v p="$ND_PRIO" '
$1 == "DSCP" && $2 == ":" && $NF == p {
    print $3; exit
}')

TC=$(( 4 * $ND_DSCP ))

export MORI_RDMA_SL=$ND_PRIO
export MORI_RDMA_TC=$TC
