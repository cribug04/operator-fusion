#!/usr/bin/env bash
set -euo pipefail
device=${1:?用法：bash run_board.sh 设备下标}
cd "$(dirname "$0")"
results=$(mktemp -d "$PWD/results.XXXXXX")
printf '结果目录 %s\n' "$results"
dort device -i "$device" | tee "$results/device.log"
dort calbin run -i "$device" -f cal -m tiny --tokens 7 -s 1 \
  --in-tokens in_input_ids --in-positions in_position_ids \
  --out-logits out_outputs_0 --dump-out "$results" 2>&1 | tee "$results/run.log"
python3 verify.py "$results/prefill-0.bin" | tee "$results/check.log"
