# SGLang Disaggregated Inference on AMD MI355X

Scripts for running **prefill-decode disaggregated inference** using SGLang on AMD MI355X (GFX950) clusters, with MORI RDMA for KV-cache transfer.

Supported models:
- DeepSeek-V3 / DeepSeek-V3-0324
- DeepSeek-R1 / DeepSeek-R1-0528-MXFP4-Preview
- Qwen3-235B

---

## Prerequisites

- A SLURM cluster with MI355X nodes (minimum 2 nodes: 1 prefill + 1 decode)
- A prebuilt ROCm Docker image with SGLang, AITER, MORI, and AINIC drivers, e.g.:
  `rocm/ali-private:sglang-0.5.8-rocm700-mi35x-mori-0210-qwen3-moe-0228`
- Model weights accessible from all nodes (typically via shared NFS)
- AMD AINIC (`nicctl`) installed on all nodes

---

## Quick Start

### 1. Configure AINIC on each bare-metal node

```bash
./enable_dcqcn.sh && ./qos.sh
```

### 2. Allocate nodes

```bash
# Using the helper (adjust partition and node names):
NODE_LIST=node01,node02,node03 bash alloc_nodes.sh

# Or allocate manually:
salloc -N 3 --ntasks-per-node=1 --nodelist=<Nodes> --gres=gpu:8 -p <partition> -t 12:00:00
```

### 3. Launch inference + benchmark

```bash
# DeepSeek-R1 (edit variables inside the script first):
bash run_interactive_disagg.sh

# Qwen3-235B:
bash run_interactive_disagg_qwen3.sh
```

Logs are written to:
- Console: `log_<MODEL_NAME>_xP<xP>_yD<yD>.log`
- Per-node: `/tmp/slurm_job-$SLURM_JOB_ID/`

---

## File Overview

**Root — user-facing entry points:**

| File | Description |
|------|-------------|
| `run_interactive_disagg.sh` | **Entry point** for DeepSeek-R1 interactive runs (salloc). Edit env vars here. |
| `run_interactive_disagg_qwen3.sh` | **Entry point** for Qwen3-235B interactive runs (salloc). Edit env vars here. |
| `run_submit_disagg.sh` | Entry point for non-interactive batch submission via sbatch. |
| `alloc_nodes.sh` | Helper to salloc a specific list of nodes by hostname. |

**`scripts/` — orchestration and server scripts:**

| File | Description |
|------|-------------|
| `scripts/run_xPyD_models.slurm` | Core SLURM orchestration: validates model, resolves node IPs, launches Docker on each node. |
| `scripts/sglang_disagg_server.sh` | Per-node script (runs inside Docker): starts prefill/decode servers, router, and benchmark based on node rank. |
| `scripts/submit_disagg.sh` | sbatch wrapper called by `run_submit_disagg.sh`. |
| `scripts/bench_throughput_with_slow_down.sh` | Throughput benchmark using `sglang.bench_one_batch_server` with slow_down coordination. |
| `scripts/bench_functional.sh` | Accuracy/functional benchmark: single chat completion + GSM8K. Run manually after servers are up. |
| `scripts/benchmark_lib.sh` | Shared benchmark utilities: `wait_for_server_ready`, `run_benchmark_serving`. |
| `scripts/set_env_vars.sh` | Sets RDMA devices, network interfaces, and MORI/SGLang env vars based on hostname. |
| `scripts/enable_dcqcn.sh` | Configures DCQCN congestion control on AMD AINIC devices. |
| `scripts/qos.sh` | Configures PFC and DSCP-priority QoS mappings on AINIC ports. |

**`utils/` — Python utilities and log parsers:**

| File | Description |
|------|-------------|
| `utils/socket_barrier.py` | Multi-node TCP barrier: waits for all nodes to open a port or pass a health check. |
| `utils/socket_wait.py` | Polls until a remote TCP port closes (used to detect when the router shuts down). |
| `utils/wait_for_prefill_idle.py` | Polls prefill server's `/v1/loads` until all DP ranks are idle (no pending requests). |
| `utils/benchmark_parser.py` | Parses benchmark log files into a table or CSV of throughput/latency metrics. |
| `utils/parse_decode_log.py` | Extracts TPOT (ms) and output throughput (tokens/s) from decode server logs. |

---

## Configuration Reference

### Topology

| Variable | Description |
|----------|-------------|
| `xP` | Number of prefill worker groups |
| `yD` | Number of decode worker groups |
| `NUM_NODES` | Total nodes to use from the salloc allocation (`xP * PREFILL_NODES_PER_WORKER + yD * DECODE_NODES_PER_WORKER`) |

When `PREFILL_TP_SIZE > 8`, a single prefill worker spans multiple nodes (`PREFILL_NODES_PER_WORKER = ceil(TP/8)`). Same for decode.

### Parallelism

| Variable | Description |
|----------|-------------|
| `PREFILL_TP_SIZE` | Tensor Parallelism size for prefill (usually = GPUs per node, e.g. 8) |
| `PREFILL_ENABLE_EP` | Enable Expert Parallelism for prefill (`true`/`false`). EP size = TP size when enabled. |
| `PREFILL_ENABLE_DP` | Enable Data Parallelism for prefill (`true`/`false`). DP size = TP size when enabled. |
| `DECODE_TP_SIZE` | Tensor Parallelism size for decode (set to 16 for 2-node decode workers) |
| `DECODE_ENABLE_EP` | Enable Expert Parallelism for decode (`true`/`false`) |
| `DECODE_ENABLE_DP` | Enable Data Parallelism for decode (`true`/`false`) |
| `DECODE_MTP_SIZE` | Multi-Token Prediction steps for speculative decoding (0 = disabled) |

### Benchmark

| Variable | Description |
|----------|-------------|
| `BENCH_INPUT_LEN` | Input sequence length in tokens |
| `BENCH_OUTPUT_LEN` | Output sequence length in tokens |
| `BENCH_RANDOM_RANGE_RATIO` | Variance ratio for sequence lengths (1 = fixed, 0.8 = ±80%) |
| `BENCH_NUM_PROMPTS_MULTIPLIER` | Total prompts = `BENCH_MAX_CONCURRENCY * multiplier` |
| `BENCH_MAX_CONCURRENCY` | Maximum concurrent requests. Can be a single value or a descending list like `"1024x512x128"` |

### Misc

| Variable | Description |
|----------|-------------|
| `MODEL_NAME` | Model directory name under `MODEL_DIR`. Must be one of the supported models. |
| `MODEL_DIR` | Base path to model weights on the shared filesystem |
| `DOCKER_IMAGE_NAME` | ROCm Docker image to use. Defaults to the Qwen3-capable image in `run_xPyD_models.slurm`. |
| `LOAD_DUMMY` | Set to `1` to skip loading real weights (use dummy weights for smoke tests) |
| `DRY_RUN` | Set to `1` to print all commands without executing them |
| `SGLANG_DIR` | Optional path to an external SGLang checkout to bind-mount into Docker |

---

## Specifying InfiniBand Devices

List available IB devices:

```bash
ibv_devinfo -l
```

Update `set_env_vars.sh` with the correct device names for your hostname pattern:

```bash
export IBDEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
```

---

## Log Files

After a run, logs are in `/tmp/slurm_job-$SLURM_JOB_ID/` and copied to `./logs/slurm_job-$SLURM_JOB_ID/`:

| File | Contents |
|------|----------|
| `pd_sglang_bench_serving.sh_NODE<N>.log` | Full output for node N |
| `prefill_NODE<N>.log` | Prefill server log for node N |
| `decode_NODE<N>.log` | Decode server log for node N |
| `proxy_NODE0.log` | Router log |

### Parse benchmark results

```bash
# Display results as a table
python3 utils/benchmark_parser.py /tmp/slurm_job-$SLURM_JOB_ID/pd_sglang_bench_serving.sh_NODE0.log

# Save to CSV
python3 utils/benchmark_parser.py /tmp/slurm_job-$SLURM_JOB_ID/pd_sglang_bench_serving.sh_NODE0.log --csv results.csv
```

---

## Check NIC Version

```bash
sudo nicctl show version host-software
sudo nicctl show version firmware
```

---

## Acknowledgements

This project is a helper repository for the AMD ROCm InferenceMAX recipe.
It builds on:
- [MAD](https://github.com/ROCm/MAD): AMD Model Automation and Dashboarding
- [InferenceMAX](https://github.com/InferenceMAX/InferenceMAX): Open-source inference benchmarking by SemiAnalysis
