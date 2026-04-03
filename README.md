# SGLang PD Simple

Simplified prefill/decode disaggregation launcher for AMD MI355X clusters.

This directory keeps the same high-level workflow as the original suite:

1. Run `bash give_me_nodes.sh`
2. Run `bash run_pd.sh`

The main differences are:

- Only one stable `sglang` source is used at runtime.
- `PYTHONPATH` is pinned to `/sgl-workspace/sglang/python`.
- Benchmark concurrency is validated/clamped before running.
- Invalid decode-log metrics fail loudly instead of printing negative values.

## Supported profiles

- `deepseek_v32_default`
- `deepseek_v32_recommended`
- `deepseek_r1_default`

Select a profile by exporting `PROFILE`, for example:

```bash
PROFILE=deepseek_v32_default bash run_pd.sh
```

## Important variables in `run_pd.sh`

- `MODEL_NAME`
- `MODEL_DIR`
- `DOCKER_IMAGE_NAME`
- `xP`, `yD`
- `PREFILL_*`
- `DECODE_*`
- `BENCH_*`
- `LOAD_DUMMY`
- `WORKSPACE_SGLANG_DIR`

## Benchmark modes

Two benchmark modes are supported:

- `BENCH_MODE=slowdown`
- `BENCH_MODE=poisson`

### `BENCH_MODE=slowdown`

- Keeps the current `slow_down` coordination flow.
- Uses `scripts/bench_throughput_with_slow_down.sh`.
- Final `TPOT` and `Output throughput` are parsed from decode logs.

### `BENCH_MODE=poisson`

- Uses `bench_serving/benchmark_serving.py`.
- Sends requests with a configurable arrival process.
- `BENCH_REQUEST_RATE=inf` means send all requests immediately.
- `BENCH_BURSTINESS=1.0` means Poisson arrival.
- Results are written to a JSON file instead of using decode-log parsing.

Relevant poisson variables:

- `BENCH_BACKEND`
- `BENCH_REQUEST_RATE`
- `BENCH_BURSTINESS`
- `BENCH_NUM_PROMPTS`
- `BENCH_NUM_WARMUPS`
- `BENCH_RESULT_DIR`
- `BENCH_RESULT_FILENAME`

## Source selection rule

At runtime the container follows one rule only:

- By default, use the image-bundled `/sgl-workspace/sglang`.
- If `WORKSPACE_SGLANG_DIR` is explicitly set, mount it as `/workspace_sglang` and relink `/sgl-workspace/sglang` to it.
- Then always export `PYTHONPATH=/sgl-workspace/sglang/python:...`

This avoids namespace-package and mixed-source ambiguity.

## Benchmark behavior

- Slowdown mode uses the `slow_down` method.
- Poisson mode uses `bench_serving/benchmark_serving.py`.
- If requested concurrency exceeds decode `max-running-requests`, it is clamped by default.
- Effective concurrency is printed in launch logs.
- Decode metrics are only parsed in slowdown mode and are rejected when timestamps are invalid.

### Example

Slowdown benchmark:

```bash
PROFILE=deepseek_v32_default BENCH_MODE=slowdown bash run_pd.sh
```

Poisson benchmark:

```bash
PROFILE=deepseek_v32_default \
BENCH_MODE=poisson \
BENCH_REQUEST_RATE=4 \
BENCH_BURSTINESS=1.0 \
BENCH_NUM_PROMPTS=2560 \
bash run_pd.sh
```

## Logs

Per-run logs are written to:

- top-level `log_<MODEL>_xP<xP>_yD<yD>.log`
- `logs/slurm_job-<jobid>/`

Each server launch also prints a `LAUNCH DETAIL` block with:

- final command
- key environment variables
- network / MORI settings
- effective benchmark concurrency
