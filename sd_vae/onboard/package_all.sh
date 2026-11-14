#!/bin/bash
# 打包 SD1.5 VAE 解码器（全 NPU）上板包。
#
# 用法（在 WSL 里）：bash package_all.sh [输出目录]
# 需要：/root/sd_vae/{cal_a,cal_b1,cal_b2a,cal_b2b,cal_c}（docc calbin from-calvx 产物）
#       /root/sd_vae/ref/latent.bin（或工作区 ref/latent.bin）
set -uo pipefail
W=$(cd "$(dirname "$0")" && pwd)
GEN="$W/../gen"
REF="$W/../ref"
ROOT=$(cd "$W/../../../.." && pwd)   # 工作区根（bohack/）
OUT=${1:-/root/sd_vae/onboard_pkg}
rm -rf "$OUT"; mkdir -p "$OUT"

for s in a b1 b2a b2b c; do
  cp -r "/root/sd_vae/cal_$s" "$OUT/cal_$s"
done

python3 "$GEN/host.py" input "$REF/latent.bin" "$OUT/in_x.bin"
cp "$GEN/host.py" "$OUT/host.py"
cp "$W/run_on_board.sh" "$OUT/run_on_board.sh"
chmod +x "$OUT/run_on_board.sh"
[ -f "$ROOT/SD_VAE_REAL_DEVICE_VERIFICATION.md" ] && cp "$ROOT/SD_VAE_REAL_DEVICE_VERIFICATION.md" "$OUT/"
[ -f "$REF/reference_image.npy" ] && cp "$REF/reference_image.npy" "$OUT/ref_image.npy"
[ -f "$REF/reference_image.png" ] && cp "$REF/reference_image.png" "$OUT/ref_image.png"
[ -f "$REF/latent.bin" ] && cp "$REF/latent.bin" "$OUT/latent.bin"

# 清单 + 校验
( cd "$OUT" && find . -type f -printf '%P %s\n' | sort ) > "$OUT/MANIFEST.txt"
( cd "$OUT" && find . -maxdepth 1 -type f ! -name SHA256SUMS.txt ! -name MANIFEST.txt -printf '%f\0' \
    | xargs -0 sha256sum ) > "$OUT/SHA256SUMS.txt"
echo "=== package: $OUT ==="
du -sh "$OUT"
echo "--- top files ---"
ls -l "$OUT" | head -20
echo "--- sha256sum -c ---"
( cd "$OUT" && sha256sum -c SHA256SUMS.txt ) | tail -5
