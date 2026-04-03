bash give_me_nodes.sh
sleep 5
LOAD_DUMMY=0 DEFAULT_DOCKER_IMAGE=rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260331  PROFILE=deepseek_v32_fp4_default  BENCH_MODE=poisson BENCH_REQUEST_RATE=inf BENCH_BURSTINESS=1.0 BENCH_NUM_PROMPTS=1 BENCH_NUM_WARMUPS=1 bash run_pd_tep8.sh
