# SGLang Disaggregated Inference on AMD MI355X

Scripts for running **prefill-decode disaggregated inference** using SGLang on AMD MI355X (GFX950) clusters, with MORI RDMA for KV-cache transfer.

Supported models:
- DeepSeek-V3 / DeepSeek-V3-0324
- DeepSeek-R1 / DeepSeek-R1-0528
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
cd scripts
./enable_dcqcn.sh && ./qos.sh
```

### 2. Allocate nodes

```bash
# Using the helper (adjust partition and node names):
NODE_LIST=node07,node08,node09 bash alloc_nodes.sh

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

The benchmark uses a **slow_down** technique to build up a full batch before decode begins:
the decode server is asked to hold all prefilled tokens in place until `BENCH_MAX_CONCURRENCY`
requests have accumulated. Once the batch is full, slow_down is released and all sequences
decode together at steady state.

Because of this, the throughput numbers reported by the SGLang benchmark tool itself are
**not valid** — they measure elapsed wall time from request submission, which includes the
slow_down hold period. The true decode performance must be extracted from the decode server
log instead.

The decode log records a stop slow_down timestamp when slow_down is released and a
finish timestamp when the last token is generated. The decode time is the interval between
these two events:

```
decode_time = t_finish - t_stop_slow_down
```

From that, the two key metrics are:

```
Output throughput (tok/s) = batch_size * output_len / decode_time
TPOT (ms)                 = decode_time * 1000 / output_len
```

Use `utils/parse_decode_log.py` to extract these automatically:

```bash
# e.g.
python3 utils/parse_decode_log.py decode_log.log $output_len $batch_size
```

---

## Check NIC Version

```bash
sudo nicctl show version host-software
sudo nicctl show version firmware
```
