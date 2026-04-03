#!/bin/bash

# Alias to the default DeepSeek-V3.2-mxfp4 recipe in config.sh.
# Use with:
#   CONFIG_FILE=./config.deepseek_v32_mxfp4.sh bash run.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/config.sh"
