#!/bin/bash
# =============================================================================
# alloc_2.sh - Interactive Node Allocation Helper
# =============================================================================
# Convenience wrapper to request an interactive salloc allocation for a
# comma-separated list of specific nodes. Used to reserve the nodes needed
# before running run_interactive_disagg.sh or run_interactive_disagg_qwen3.sh.
#
# Usage:
#   NODE_LIST=node01,node02,node03 bash alloc_2.sh
#
# The number of nodes is inferred automatically from NODE_LIST.
# The allocation is on the "pegasus" partition with 8 GPUs per node.
# Adjust the partition (-p) and time limit (-t) as needed for your cluster.
# =============================================================================

array=(${NODE_LIST//,/ })
salloc -N ${#array[@]} --ntasks-per-node=1 --nodelist=$NODE_LIST --gres=gpu:8 -p pegasus -t 00:15:00
