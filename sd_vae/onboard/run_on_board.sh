#!/bin/bash
# SD1.5 VAE 解码器（全 NPU）上板跑图：5 段 calbin 串起来，把 latent 解成 512×512 图。
#
# 用法：
#   ./run_on_board.sh <工作目录> [设备号]
# 目录里需要有：
#   cal_a/ cal_b1/ cal_b2a/ cal_b2b/ cal_c/   —— docc calbin from-calvx 产物（含 param_blk0.bin）
#   in_x.bin                                  —— host.py input 生成（latent ⇒ stage A 的 in_x）
#   ref_image.npy                             —— torch 参考图（可选，用于数值对比）
#   host.py                                   —— 主机侧工具
# 产物：out_a.bin … out_c.bin、image.npy、image.png、compare.json、timing.txt
set -uo pipefail
D=${1:?usage: run_on_board.sh <dir> [device]}
DEV=${2:-0}
cd "$D" || exit 1

tstart=$(date +%s.%N)
: > timing.txt

run_stage () {  # $1=caldir  $2=submodel  $3=in_file  $4=out_file
  local t0 t1 rc
  t0=$(date +%s.%N)
  dort calbin run -i "$DEV" -f "$1" -m "$2" \
    --raw-in "$3" --raw-out "$4" \
    --in-tensor in_x --out-tensor out_y
  rc=$?
  t1=$(date +%s.%N)
  echo "$2 rc=$rc $(awk "BEGIN{printf \"%.1f\", $t1-$t0}")s" | tee -a timing.txt
  [ "$rc" -eq 0 ] || { echo "!! stage $2 failed"; exit 1; }
}

run_stage cal_a   pkg_a   in_x.bin    out_a.bin
run_stage cal_b1  pkg_b1  out_a.bin   out_b1.bin
run_stage cal_b2a pkg_b2a out_b1.bin  out_b2a.bin
run_stage cal_b2b pkg_b2b out_b2a.bin out_b2b.bin
run_stage cal_c   pkg_c   out_b2b.bin out_c.bin
tend=$(date +%s.%N)
echo "total $(awk "BEGIN{printf \"%.1f\", $tend-$tstart}")s" | tee -a timing.txt

# 解出图像并对比
python3 host.py image out_c.bin image.npy image.png || exit 1
if [ -f ref_image.npy ]; then
  python3 host.py compare image.npy ref_image.npy compare.json
fi
echo "=== done: $(pwd)/image.png ==="
cat timing.txt
