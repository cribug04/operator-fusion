#!/usr/bin/env bash
set -euo pipefail
device=${1:?用法：bash run_board.sh 设备下标}
cd "$(dirname "$0")"
results=$(mktemp -d "$PWD/results.XXXXXX")
printf '结果目录 %s\n' "$results"
dort device -i "$device" | tee "$results/device.log"
for input in cases/*.bin; do
    name=${input##*/}
    dort calbin run -i "$device" -f cal -m topk \
      --in-tensor in_raw --out-tensor out_raw --raw-in "$input" \
      --raw-out "$results/$name" 2>&1 | tee "$results/$name.log"
    python3 verify.py check "$input" "$results/$name" | tee -a "$results/check.log"
done
