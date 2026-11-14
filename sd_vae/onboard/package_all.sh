#!/bin/bash
# 打包 SD1.5 VAE 解码器（全 NPU）上板包。
#
# 用法（在 WSL 里）：./package_all.sh [输出目录]
# 需要：/root/sd_vae/{cal_a,cal_b1,cal_b2a,cal_b2b,cal_c}（docc calbin from-calvx 产物）
#       /root/sd_vae/ref/latent.bin、../(工作区) _recon/work/sd_vae/ref/reference_image.npy
set -uo pipefail
W=$(cd "$(dirname "$0")" && pwd)
GEN="$W/../gen"
REF="$W/../ref"
OUT=${1:-/root/sd_vae/onboard_pkg}
rm -rf "$OUT"; mkdir -p "$OUT"

for s in a b1 b2a b2b c; do
  cp -r "/root/sd_vae/cal_$s" "$OUT/cal_$s"
done

python3 "$GEN/host.py" input /root/sd_vae/ref/latent.bin "$OUT/in_x.bin"
cp "$W/host.py" "$OUT/host.py"
cp "$W/run_on_board.sh" "$OUT/run_on_board.sh"
chmod +x "$OUT/run_on_board.sh"
if [ -f "$REF/reference_image.npy" ]; then
  cp "$REF/reference_image.npy" "$OUT/ref_image.npy"
fi
cp "$REF/latent.bin" "$OUT/latent.bin" 2>/dev/null || true

# 清单
( cd "$OUT" && find . -type f -printf '%P %s\n' | sort ) > "$OUT/MANIFEST.txt"
( cd "$OUT" && sha256sum $(find . -maxdepth 1 -type f -printf '%f ') ) > "$OUT/SHA256SUMS.txt"
echo "=== package: $OUT ==="
du -sh "$OUT"
cat "$OUT/MANIFEST.txt"
