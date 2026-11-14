#!/bin/bash
# 打包 SD1.5 VAE 解码器（全 NPU）上板包。
#
# 用法（在 WSL 里，从仓库任意位置均可跑）：
#   bash sd_vae/onboard/package_all.sh [OUT_DIR] [TAR_PATH]
# 默认：OUT_DIR=/root/sd_vae/onboard_pkg，TAR_PATH=/root/sd_vae/sd_vae_onboard_pkg.tar.gz
#
# 环境变量：
#   CAL_DIR  固定版 calbin 根目录（含 cal_{a,b1,b2a,b2b,c}），默认 /root/sd_vae
#   PYTHON   带 numpy 的 python，默认优先 /root/sd_vae/venv/bin/python，其次 python3
#
# 输入全部来自本仓库（repo-relative）：
#   sd_vae/gen/host.py、sd_vae/onboard/run_on_board.sh、
#   docs/SD_VAE_REAL_DEVICE_VERIFICATION.md、
#   sd_vae/evidence/{latent.bin,ref_image.npy,reference_image.png}
# 产物：上头包目录 + 确定性 tar.gz（同名 sha256 打印在最后）。
set -euo pipefail

W=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$W/../.." && pwd)
CAL_DIR=${CAL_DIR:-/root/sd_vae}
OUT=${1:-/root/sd_vae/onboard_pkg}
TAR_PATH=${2:-/root/sd_vae/sd_vae_onboard_pkg.tar.gz}

# ---- python（生成 in_x.bin 需要 numpy）----
if [ -n "${PYTHON:-}" ]; then
  PY=$PYTHON
elif [ -x /root/sd_vae/venv/bin/python ]; then
  PY=/root/sd_vae/venv/bin/python
else
  PY=python3
fi
"$PY" -c 'import numpy' 2>/dev/null || {
  echo "!! $PY 不能 import numpy（用 PYTHON=/path/to/python 指定）" >&2; exit 1; }

# ---- 仓库内输入 ----
SRC_HOST="$REPO/sd_vae/gen/host.py"
SRC_RUN="$REPO/sd_vae/onboard/run_on_board.sh"
SRC_MANUAL="$REPO/docs/SD_VAE_REAL_DEVICE_VERIFICATION.md"
EVID="$REPO/sd_vae/evidence"
for f in "$SRC_HOST" "$SRC_RUN" "$SRC_MANUAL" \
         "$EVID/latent.bin" "$EVID/ref_image.npy" "$EVID/reference_image.png"; do
  [ -f "$f" ] || { echo "!! 缺少仓库输入文件: $f" >&2; exit 1; }
done

# ---- 组装包目录 ----
rm -rf "$OUT"
mkdir -p "$OUT"
for s in a b1 b2a b2b c; do
  [ -d "$CAL_DIR/cal_$s" ] || { echo "!! 缺少 calbin 目录: $CAL_DIR/cal_$s" >&2; exit 1; }
  cp -r "$CAL_DIR/cal_$s" "$OUT/cal_$s"
done
cp "$SRC_HOST" "$OUT/host.py"
cp "$SRC_RUN" "$OUT/run_on_board.sh"
cp "$SRC_MANUAL" "$OUT/SD_VAE_REAL_DEVICE_VERIFICATION.md"
cp "$EVID/latent.bin" "$OUT/latent.bin"
cp "$EVID/ref_image.npy" "$OUT/ref_image.npy"
cp "$EVID/reference_image.png" "$OUT/ref_image.png"
chmod +x "$OUT/run_on_board.sh"

# ---- 固定版 calbin 自检（2026-11-14 20:59 修复版；防止误用 pre-fix 包）----
check_size () {  # $1=相对路径  $2=预期字节数
  local got
  got=$(wc -c < "$OUT/$1")
  [ "$got" -eq "$2" ] || {
    echo "!! $1 大小 $got != 预期 $2：calbin 不是固定修复版（检查 CAL_DIR）" >&2; exit 1; }
}
check_size cal_a/pkg_a/ccu0_chip0.elf   1414672
check_size cal_a/pkg_a/ccu1_chip0.elf   1411672
check_size cal_b1/pkg_b1/ccu0_chip0.elf  968536
check_size cal_b1/pkg_b1/ccu1_chip0.elf  966160
check_size cal_b2a/pkg_b2a/ccu0_chip0.elf  398884
check_size cal_b2a/pkg_b2a/ccu1_chip0.elf  397144
check_size cal_b2b/pkg_b2b/ccu0_chip0.elf   31160
check_size cal_b2b/pkg_b2b/ccu1_chip0.elf   30700
check_size cal_c/pkg_c/ccu0_chip0.elf   1528008
check_size cal_c/pkg_c/ccu1_chip0.elf   1525764
check_size cal_a/param_blk0.bin  21483520
check_size cal_b1/param_blk0.bin 33452032
check_size cal_b2a/param_blk0.bin 28729344
check_size cal_b2b/param_blk0.bin 4722688
check_size cal_c/param_blk0.bin  12292096

# ---- 生成段 A 输入（必须 163,840 B）----
( cd "$OUT" && "$PY" host.py input latent.bin in_x.bin )
IN_BYTES=$(wc -c < "$OUT/in_x.bin")
[ "$IN_BYTES" -eq 163840 ] || {
  echo "!! in_x.bin = $IN_BYTES B != 163,840 B" >&2; exit 1; }

# ---- MANIFEST.txt：排序的“相对路径 字节数”，不含自身与 SHA256SUMS.txt ----
( cd "$OUT" && find . -type f ! -name MANIFEST.txt ! -name SHA256SUMS.txt \
    -printf '%P %s\n' | LC_ALL=C sort ) > "$OUT/MANIFEST.txt"

# ---- SHA256SUMS.txt：包内所有文件（不含自身与 MANIFEST.txt），相对包根路径 ----
( cd "$OUT" && find . -type f ! -name SHA256SUMS.txt ! -name MANIFEST.txt \
    -printf '%P\0' | LC_ALL=C sort -z | xargs -0 sha256sum ) > "$OUT/SHA256SUMS.txt"

# ---- 确定性 tar.gz（--mtime 固定 + 排序 + numeric-owner；gzip 走 stdin 故无 mtime/文件名头）----
mkdir -p "$(dirname "$TAR_PATH")"
rm -f "$TAR_PATH"
tar --sort=name --owner=0 --group=0 --numeric-owner \
    --mtime='2026-11-14 00:00:00Z' -czf "$TAR_PATH" -C "$OUT" .

TAR_BYTES=$(wc -c < "$TAR_PATH")
TAR_SHA=$(sha256sum "$TAR_PATH" | cut -d' ' -f1)
PKG_FILES=$(cd "$OUT" && find . -type f | wc -l)
PKG_BYTES=$(cd "$OUT" && find . -type f -printf '%s\n' | awk '{s+=$1} END{print s}')
echo "=== package: $OUT ==="
echo "files: $PKG_FILES  bytes: $PKG_BYTES"
echo "tar: $TAR_PATH"
echo "tar bytes: $TAR_BYTES"
echo "tar sha256: $TAR_SHA"
echo "=== self check: cd $OUT && sha256sum -c SHA256SUMS.txt ==="
( cd "$OUT" && sha256sum -c SHA256SUMS.txt ) | tail -3
echo "OK"
