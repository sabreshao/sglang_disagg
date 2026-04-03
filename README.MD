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
After reboot machine
```bash
cd scripts
./enable_dcqcn.sh && ./qos.sh
```

### 2. Allocate nodes

```bash
bash give_me_nodes.sh
squeue # make sure alloc is successful
```

### 3. Launch inference + benchmark

```bash
# Deepseek-V3.2 # edit ./config.deepseek_v32.sh if you want to change some configs
CONFIG_FILE=./config.deepseek_v32.sh bash run.sh

# Deepseek-V3.2-mxfp4
CONFIG_FILE=./config.deepseek_v32_mxfp4.sh bash run.sh

# DeepSeek-R1 
CONFIG_FILE=./config.deepseek_r1.sh bash run.sh

```

Logs are written to:
 - logs/slurm_job-{jobid}/date/
