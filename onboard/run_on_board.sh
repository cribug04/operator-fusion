#!/usr/bin/env bash
# 在**板子所在的 Linux 主机**上一键跑完 4 个用例并对比（也可以照抄里面的命令手动执行）。
# 用法：
#   DEVICE=0 CAL_ROOT=./onboard_cal CASE_DIR=./cases OUT=./board_out bash run_on_board.sh
# 前置：package_all.sh 已产出 calbin；本机有 dort、python3+numpy；in_x/out_y 是 decl 名。
set -uo pipefail

DEVICE=${DEVICE:-0}
CAL_ROOT=${CAL_ROOT:-$(cd "$(dirname "$0")" && pwd)/onboard_cal}
CASE_DIR=${CASE_DIR:-$(cd "$(dirname "$0")" && pwd)/cases}
OUT=${OUT:-$(cd "$(dirname "$0")" && pwd)/board_out}
DORT=${DORT:-dort}
PY=${PY:-python3}
COMPARE=${COMPARE:-$(cd "$(dirname "$0")" && pwd)/compare_bf16.py}

mkdir -p "$OUT/logs"
fail() { echo "FAIL: $*" >&2; exit 1; }

echo "################ 0) 板卡健康检查 ################"
$DORT device -i "$DEVICE" 2>&1 | tee "$OUT/logs/device.txt"
if ! grep -q "RT up" "$OUT/logs/device.txt"; then
  cat >&2 <<'EOF'
板卡状态不是 “OS up  RT up”。先按下面的流程恢复（⚠ 共享板请先问负责人，不要自行 reprobe）：
  dort device reprobe -i 0 --yes        # SOFT_RESET，只有这一步会重启板上 RT
  sleep 15
  sudo rmmod calculet_pci && sudo insmod ~/cal-pcie-driver/calculet_pci.ko
  # 等 RT up 后重跑本脚本
EOF
  fail "board not ready"
fi

run_one() {
  local name=$1 cal=$2 model=$3 raw_in=$4 raw_out=$5
  echo
  echo "===== [$name] dort calbin run ====="
  local r
  for r in 1 2 3; do
    if ! $DORT calbin run -i "$DEVICE" -f "$CAL_ROOT/$cal" -m "$model" \
         --raw-in "$CASE_DIR/$raw_in" --raw-out "$OUT/${name}_run$r.bin" \
         --in-tensor in_x --out-tensor out_y 2>&1 | tee "$OUT/logs/run-$name-$r.log"; then
      fail "$name 第 $r 次运行失败（看 $OUT/logs/run-$name-$r.log）"
    fi
  done
  echo "--- 三次输出 sha256（应相同）---"
  sha256sum "$OUT/${name}_run"*.bin | tee "$OUT/logs/sha-$name.txt"
  local n; n=$(grep -c . "$OUT/logs/sha-$name.txt")
  [ "$n" -ge 3 ] && { local u; u=$(awk '{print $1}' "$OUT/logs/sha-$name.txt" | sort -u | wc -l); [ "$u" -eq 1 ] || echo "WARN: $name 三次输出不一致！"; }
}

echo
echo "################ 1) resize（32×32×16 → 64×64×16） ################"
run_one resize       resize       resize_pkg       resize_small/input.bin out_resize.bin
echo "--- 与 CPU 参考对比 ---"
$PY "$COMPARE" "$OUT/resize_run1.bin" "$CASE_DIR/resize_small/expected.bin" --label resize | tee "$OUT/logs/cmp-resize.txt"

echo
echo "################ 2) resize 通道分块（32×32×256 → 64×64×256） ################"
run_one resize_big   resize_big   resize_big_pkg   resize_big/input.bin out_resize_big.bin
echo "--- 与 CPU 参考对比 ---"
$PY "$COMPARE" "$OUT/resize_big_run1.bin" "$CASE_DIR/resize_big/expected.bin" --label resize_big | tee "$OUT/logs/cmp-resize-big.txt"

echo
echo "################ 3) conv stride-2 FAST 原型 vs 同形状慢路 ################"
run_one conv_fast    conv_s2      conv_s2_pkg      conv_s2/input.bin out_conv_fast.bin
run_one conv_ref     conv_s2_ref  conv_s2_ref_pkg  conv_s2/input.bin out_conv_ref.bin
echo "--- 严格对比：fast vs ref（裁掉 2 像素边框）---"
$PY "$COMPARE" "$OUT/conv_fast_run1.bin" "$OUT/conv_ref_run1.bin" \
  --actual-shape 84,84,64 --crop 2:82,2:82 --expected-shape 84,84,64 --expected-crop 2:82,2:82 \
  --label conv_fast_vs_ref | tee "$OUT/logs/cmp-conv-strict.txt"
echo "--- 宽松对比：两者分别 vs CPU 参考 ---"
$PY "$COMPARE" "$OUT/conv_fast_run1.bin" "$CASE_DIR/conv_s2/expected.bin" \
  --actual-shape 84,84,64 --crop 2:82,2:82 --expected-shape 80,80,64 --label conv_fast_vs_cpu | tee "$OUT/logs/cmp-conv-fast-cpu.txt"
$PY "$COMPARE" "$OUT/conv_ref_run1.bin" "$CASE_DIR/conv_s2/expected.bin" \
  --actual-shape 84,84,64 --crop 2:82,2:82 --expected-shape 80,80,64 --label conv_ref_vs_cpu | tee "$OUT/logs/cmp-conv-ref-cpu.txt"

echo
echo "################ 汇总 ################"
echo "所有原始输出与日志在: $OUT"
grep -h "VERDICT" "$OUT"/logs/cmp-*.txt || true
