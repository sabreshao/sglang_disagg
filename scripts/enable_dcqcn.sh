#!/bin/bash
set -euo pipefail

TOKEN_BUCKET_SIZE=800000
AI_RATE=160
ALPHA_UPDATE_INTERVAL=1
ALPHA_UPDATE_G=512
INITIAL_ALPHA_VALUE=64
RATE_INCREASE_BYTE_COUNT=431068
HAI_RATE=300
RATE_REDUCE_MONITOR_PERIOD=1
RATE_INCREASE_THRESHOLD=1
RATE_INCREASE_INTERVAL=1
CNP_DSCP=46

ROCE_DEVICES=$(ibv_devices | grep ionic_ | awk '{print $1}' | paste -sd " ")
for roce_dev in ${ROCE_DEVICES}; do
    sudo nicctl update dcqcn -r "${roce_dev}" -i 1 \
        --token-bucket-size "${TOKEN_BUCKET_SIZE}" \
        --ai-rate "${AI_RATE}" \
        --alpha-update-interval "${ALPHA_UPDATE_INTERVAL}" \
        --alpha-update-g "${ALPHA_UPDATE_G}" \
        --initial-alpha-value "${INITIAL_ALPHA_VALUE}" \
        --rate-increase-byte-count "${RATE_INCREASE_BYTE_COUNT}" \
        --hai-rate "${HAI_RATE}" \
        --rate-reduce-monitor-period "${RATE_REDUCE_MONITOR_PERIOD}" \
        --rate-increase-threshold "${RATE_INCREASE_THRESHOLD}" \
        --rate-increase-interval "${RATE_INCREASE_INTERVAL}" \
        --cnp-dscp "${CNP_DSCP}"
done
