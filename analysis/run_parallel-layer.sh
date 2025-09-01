#!/usr/bin/env bash
BASE_DIR=/path/checkpoint-140
TARGET_DIR=/path/checkpoint-1600

LAYER_RANGES="0-9
10-19
20-29
30-39"

export BASE_DIR TARGET_DIR 
K_TOP=4096                          
K_TAIL=0

echo "$LAYER_RANGES" | \
parallel -j4 'CUDA_VISIBLE_DEVICES=$(({%}-1)) \
    python svd_recover_by_layer.py \
        --layers {} \
        --k_top '"$K_TOP"' \
        --k_tail '"$K_TAIL"
