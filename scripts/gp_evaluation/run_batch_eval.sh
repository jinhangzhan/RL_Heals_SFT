#!/usr/bin/env bash

LAYER_RANGES=(
#  "0 4"
#  "0 9"
#  "0 14"
#  "0 19"
  "0 24"
  "0 29"
  "0 34"
  "0 39"
)

GPU_NUM=4    

for idx in "${!LAYER_RANGES[@]}"; do
  read BEGIN END <<< "${LAYER_RANGES[$idx]}"
  GPU_ID=$(( idx % GPU_NUM ))

  echo ">>> Submit layers ${BEGIN}-${END}  on GPU ${GPU_ID}"
  nohup ./run_recover_layer_language_ood_eval.sh ${BEGIN} ${END} ${GPU_ID} \
        > recover_layer_${BEGIN}_${END}_ck1600.log 2>&1 &
done

echo "All jobs launched. Check recover_layer_*.log for progress."
