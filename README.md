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

## Source selection rule

At runtime the container follows one rule only:

- By default, use the image-bundled `/sgl-workspace/sglang`.
- If `WORKSPACE_SGLANG_DIR` is explicitly set, mount it as `/workspace_sglang` and relink `/sgl-workspace/sglang` to it.
- Then always export `PYTHONPATH=/sgl-workspace/sglang/python:...`

This avoids namespace-package and mixed-source ambiguity.

## Benchmark behavior

- The benchmark still uses the `slow_down` method.
- If requested concurrency exceeds decode `max-running-requests`, it is clamped by default.
- Effective concurrency is printed in launch logs.
- Decode metrics are parsed from decode logs and rejected when timestamps are invalid.

## Logs

Per-run logs are written to:

- top-level `log_<MODEL>_xP<xP>_yD<yD>.log`
- `logs/slurm_job-<jobid>/`

Each server launch also prints a `LAUNCH DETAIL` block with:

- final command
- key environment variables
- network / MORI settings
- effective benchmark concurrency
