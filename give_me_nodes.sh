#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cd "${SCRIPT_DIR}/scripts"
./enable_dcqcn.sh
./qos.sh

cd "${SCRIPT_DIR}"
NODE_LIST="${NODE_LIST:-node04,node08}" bash "${SCRIPT_DIR}/alloc_nodes.sh"
squeue
