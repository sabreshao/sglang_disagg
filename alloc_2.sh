#!/bin/bash
salloc -N 2 --ntasks-per-node=1 --nodelist=node28,node31 --gres=gpu:8 -p pegasus -t 02:00:00
