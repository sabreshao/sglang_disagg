#!/bin/bash
salloc -N 2 --ntasks-per-node=1 --nodelist=node28,node29 --gres=gpu:8 -p pegasus -t 01:00:00
