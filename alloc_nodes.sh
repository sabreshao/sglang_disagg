#!/bin/bash
# =============================================================================
# alloc_nodes.sh - Interactive Node Allocation Helper
# =============================================================================
# Convenience wrapper to request an interactive salloc allocation for a
# comma-separated list of specific nodes.
#
# Usage:
#   NODE_LIST=node01,node02,node03 bash alloc_nodes.sh
#
# The allocation is on the "pegasus" partition with 8 GPUs per node.
# =============================================================================

set -euo pipefail

array=(${NODE_LIST//,/ })
salloc -N ${#array[@]} --ntasks-per-node=1 --nodelist="${NODE_LIST}" --gres=gpu:8 -p pegasus -t 01:00:00
