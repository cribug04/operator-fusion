#!/usr/bin/env bash
# 在板子所在主机上：Resize 单核 vs 双核，同形状、同输入，各跑 N 次，记录 dort 整条命令的墙钟时间并核对数值。
# 用法：DEVICE=0 CAL_ROOT=./onboard_cal CASE_DIR=./cases OUT=./perf_out N=5 bash run_resize_perf.sh
# 前置：package_all.sh 已产出 resize_big/resize_big2/resize_huge/resize_huge2；
#      cases/resize_huge 由 `python3 gen_resize_huge_case.py` 现场生成（约 2–3 分钟）。
# 注意：这里的时间包含 dort 进程启动、模型部署、输入输出搬运，不是 kernel 净时间；
#      只用于同形状单核/双核的相对比较，报告时必须注明。
set -euo pipefail
DEVICE=${DEVICE:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
CAL_ROOT=${CAL_ROOT:-$HERE/onboard_cal}
CASE_DIR=${CASE_DIR:-$HERE/cases}
OUT=${OUT:-$HERE/perf_out}
N=${N:-5}
DORT=${DORT:-dort}
PY=${PY:-python3}
COMPARE=${COMPARE:-$HERE/compare_bf16.py}

fail() { echo "FAIL: $*" >&2; exit 1; }
[[ ! -e "$OUT" ]] || fail "OUT 已存在：$OUT（换个目录，避免混旧结果）"
mkdir -p "$OUT/logs"
[[ -f "$CASE_DIR/resize_huge/input.bin" ]] || fail "缺 $CASE_DIR/resize_huge/input.bin，先跑 python3 $HERE/gen_resize_huge_case.py"
command -v "$DORT" >/dev/null || fail "找不到 dort"

$DORT device -i "$DEVICE" 2>&1 | tee "$OUT/logs/device.txt"
grep -q "OS up.*RT up" "$OUT/logs/device.txt" || fail "板卡不是 OS up / RT up"

TIMING="$OUT/timing.tsv"
echo -e "name\trun\tseconds\tsha256" > "$TIMING"

run_timed() { # name cal model case expected_bytes
  local name=$1 cal=$2 model=$3 case=$4 bytes=$5 r t0 t1 dt
  echo; echo "===== [$name] ×$N ====="
  for r in $(seq 1 "$N"); do
    t0=$(date +%s.%N)
    $DORT calbin run -i "$DEVICE" -f "$CAL_ROOT/$cal" -m "$model" \
      --raw-in "$CASE_DIR/$case/input.bin" --raw-out "$OUT/${name}_run$r.bin" \
      --in-tensor in_x --out-tensor out_y > "$OUT/logs/run-$name-$r.log" 2>&1 || fail "$name 第 $r 次失败，见 $OUT/logs/run-$name-$r.log"
    t1=$(date +%s.%N)
    dt=$(echo "$t1 - $t0" | bc)
    [[ $(wc -c < "$OUT/${name}_run$r.bin") -eq "$bytes" ]] || fail "$name 输出字节数不对"
    local s; s=$(sha256sum "$OUT/${name}_run$r.bin" | cut -c1-16)
    echo -e "$name\t$r\t$dt\t$s" | tee -a "$TIMING"
    [[ $r -eq 1 ]] || cmp -s "$OUT/${name}_run1.bin" "$OUT/${name}_run$r.bin" || fail "$name 第 $r 次输出与第 1 次不同"
  done
  echo "--- 与 CPU 期望对比 ---"
  $PY "$COMPARE" "$OUT/${name}_run1.bin" "$CASE_DIR/$case/expected.bin" --mode exact --label "$name" | tee "$OUT/logs/cmp-$name.txt"
}

# 顺序：单核→双核；每个都跑 N 次，首尾差异可看 timing.tsv
run_timed resize_big   resize_big   resize_big_pkg   resize_big  2097152
run_timed resize_big2  resize_big2  resize_big2_pkg  resize_big  2097152
run_timed resize_huge  resize_huge  resize_huge_pkg  resize_huge 33554432
run_timed resize_huge2 resize_huge2 resize_huge2_pkg resize_huge 33554432

echo; echo "===== 单核 vs 双核输出必须逐字节相同 ====="
cmp "$OUT/resize_big_run1.bin"  "$OUT/resize_big2_run1.bin"  && echo "big:  single == dual"
cmp "$OUT/resize_huge_run1.bin" "$OUT/resize_huge2_run1.bin" && echo "huge: single == dual"

echo; echo "===== 汇总（中位数，秒；含 dort 进程/部署/IO 开销）====="
$PY - "$TIMING" <<'PYEOF'
import sys, statistics, collections
rows = [l.rstrip('\n').split('\t') for l in open(sys.argv[1])][1:]
d = collections.defaultdict(list)
for name, run, sec, sha in rows: d[name].append(float(sec))
for name in ['resize_big', 'resize_big2', 'resize_huge', 'resize_huge2']:
    if name in d: print(f"{name:14s} n={len(d[name])} median={statistics.median(d[name]):.3f} min={min(d[name]):.3f} max={max(d[name]):.3f}")
for a, b in [('resize_big', 'resize_big2'), ('resize_huge', 'resize_huge2')]:
    if a in d and b in d:
        print(f"{a} / {b} 中位数比 = {statistics.median(d[a])/statistics.median(d[b]):.2f}（含固定开销，不是 kernel 加速比）")
PYEOF
echo "全部原始输出与日志在 $OUT；timing.tsv 可直接附进文档。"
