#!/bin/bash
array=(${NODE_LIST//,/ })
salloc -N ${#array[@]} --ntasks-per-node=1 --nodelist=$NODE_LIST --gres=gpu:8 -p pegasus -t 02:00:00
