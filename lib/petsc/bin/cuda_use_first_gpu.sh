#!/bin/bash

# Set the env var CUDA_VISIBLE_DEVICES following these example rules:
#  1) If CUDA_VISIBLE_DEVICES=1,2,3, pick the first GPU and set CUDA_VISIBLE_DEVICES=1
#  2) If CUDA_VISIBLE_DEVICES=(empty), keep it empty
#  3) If CUDA_VISIBLE_DEVICES in unset, set CUDA_VISIBLE_DEVICES=0
# Note if use ${CUDA_VISIBLE_DEVICES:-0}, in case 2) it would set CUDA_VISIBLE_DEVICES=0, which is not what we want.
export CUDA_VISIBLE_DEVICES=$(echo ${CUDA_VISIBLE_DEVICES-0} | cut -d ',' -f 1)
exec "$@"
