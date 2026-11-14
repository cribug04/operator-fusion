#!/usr/bin/env bash
set -euo pipefail

device=${1:?用法：bash run_board.sh 设备下标}
cd "$(dirname "$0")"
results=$(mktemp -d "$PWD/results.XXXXXX")
printf '验收结果目录 %s\n' "$results"

dort device -i "$device" | tee "$results/device.log"
for variant in fused split; do
    dort calbin run -i "$device" -f "$variant/cal" -m "$variant" \
        --in-tensor in_raw --out-tensor out_raw \
        --raw-in input.bin --raw-out "$results/$variant.bin" \
        2>&1 | tee "$results/$variant.log"
done
python3 verify.py check input.bin "$results/fused.bin" "$results/split.bin" \
    2>&1 | tee "$results/correctness.log"
