#!/bin/bash
# =============================================================================
# run_interactive_disagg.sh - Interactive Entry Point for DeepSeek-R1
# =============================================================================
# This script is the primary entry point for running a disaggregated inference
# job interactively (via salloc). It sets all environment variables and then
# delegates to run_xPyD_models.slurm, which launches Docker containers on each
# allocated node and starts the prefill/decode/router services inside them.
#
# Usage:
#   1. Allocate nodes with salloc:
#      salloc -N <NUM_NODES> --ntasks-per-node=1 --nodelist=<Nodes> \
#             --gres=gpu:8 -p <partition> -t 12:00:00
#   2. Run this script:
#      bash run_interactive_disagg.sh
#
# Logs are written to: log_<MODEL_NAME>_xP<xP>_yD<yD>.log
#   and also to:        /tmp/slurm_job-$SLURM_JOB_ID/
# =============================================================================

# -----------------------------------------------------------------------------
# Topology Configuration
# xP: Number of prefill worker groups (each may span multiple nodes if TP > 8)
# yD: Number of decode worker groups (each may span multiple nodes if TP > 16)
# NUM_NODES: Total nodes to use from the salloc allocation (xP + yD, or more
#            if a single worker spans multiple nodes for large TP sizes)
# -----------------------------------------------------------------------------
export xP=1
export yD=1
export NUM_NODES=2

# -----------------------------------------------------------------------------
# Model Configuration
# MODEL_NAME: Name of the model directory under MODEL_DIR. Must match one of
#             the supported models in run_xPyD_models.slurm.
#             Supported: DeepSeek-V3, DeepSeek-V3-0324, DeepSeek-R1,
#                        DeepSeek-R1-0528-MXFP4-Preview, Qwen3-235B
# MODEL_DIR:  Base path on a shared filesystem where model weights are stored.
#             The full model path is MODEL_DIR/MODEL_NAME.
# -----------------------------------------------------------------------------
export MODEL_NAME=DeepSeek-R1
export MODEL_DIR="/mnt/nfs/RAID/shared/huggingface/hub/"

# -----------------------------------------------------------------------------
# Parallelism Configuration (Prefill)
# PREFILL_TP_SIZE:   Tensor Parallelism degree for each prefill worker.
#                    Typically equals the number of GPUs per node (e.g. 8).
# PREFILL_ENABLE_EP: Enable Expert Parallelism for prefill. When true, EP size
#                    is set equal to PREFILL_TP_SIZE. (true/false)
# PREFILL_ENABLE_DP: Enable Data Parallelism for prefill. When true, DP size
#                    is set equal to PREFILL_TP_SIZE. (true/false)
# -----------------------------------------------------------------------------
export PREFILL_TP_SIZE=8
export PREFILL_ENABLE_EP=false
export PREFILL_ENABLE_DP=false

# -----------------------------------------------------------------------------
# Parallelism Configuration (Decode)
# DECODE_TP_SIZE:   Tensor Parallelism degree for each decode worker.
#                   Set to 16 when a single decode worker spans 2 nodes.
# DECODE_ENABLE_EP: Enable Expert Parallelism for decode. When true, EP size
#                   is set equal to DECODE_TP_SIZE. (true/false)
# DECODE_ENABLE_DP: Enable Data Parallelism for decode. When true, DP size
#                   is set equal to DECODE_TP_SIZE. (true/false)
# DECODE_MTP_SIZE:  Number of speculative decoding steps (Multi-Token
#                   Prediction). Set to 0 to disable MTP/speculative decoding.
# -----------------------------------------------------------------------------
export DECODE_TP_SIZE=8
export DECODE_ENABLE_EP=false
export DECODE_ENABLE_DP=false
export DECODE_MTP_SIZE=0

# -----------------------------------------------------------------------------
# Benchmark Configuration
# BENCH_INPUT_LEN:            Input sequence length (tokens) for benchmark.
# BENCH_OUTPUT_LEN:           Output sequence length (tokens) for benchmark.
# BENCH_RANDOM_RANGE_RATIO:   Variance ratio applied to input/output lengths.
#                             1 means lengths are fixed (no randomness).
# BENCH_NUM_PROMPTS_MULTIPLIER: Total prompts = BENCH_MAX_CONCURRENCY * this.
# BENCH_MAX_CONCURRENCY:      Maximum number of concurrent requests sent to
#                             the router during benchmark. Can be a single
#                             value or a descending list like "1024x512x128"
#                             (only the first value is active currently).
# -----------------------------------------------------------------------------
export BENCH_INPUT_LEN=3300
export BENCH_OUTPUT_LEN=400
export BENCH_RANDOM_RANGE_RATIO=1
export BENCH_NUM_PROMPTS_MULTIPLIER=10
export BENCH_MAX_CONCURRENCY=1536

# -----------------------------------------------------------------------------
# Misc Configuration
# LOAD_DUMMY:        Set to 1 to skip loading real model weights (use dummy
#                    weights). Useful for quick smoke tests. Default: 0.
# DRY_RUN:           Set to 1 to print all commands without executing them.
#                    Useful for validating configuration. Default: 0.
# DOCKER_IMAGE_NAME: ROCm Docker image containing SGLang, AITER, MoRI, and
#                    AINIC drivers for MI355X (GFX950) GPUs.
# -----------------------------------------------------------------------------
export LOAD_DUMMY=1
export DRY_RUN=0
# export DOCKER_IMAGE_NAME=rocm/sgl-dev:sglang-0.5.8-rocm700-mi35x-mori-0210

# Launch the main SLURM orchestration script and tee output to a log file.
bash scripts/run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log
