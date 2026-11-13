#!/bin/bash
# Rebuild + package + doctor for the resize_bilinear2 probe example (run inside WSL).
# Usage: bash examples/resize_probe/verify.sh   (cwd = calBLAS 仓库根，工具链 channel=calculet)
set -uo pipefail
export PATH=/root/.cargo/bin:$PATH
export RUSTUP_HOME=${RUSTUP_HOME:-/root/.rustup}
export CARGO_HOME=${CARGO_HOME:-/root/.cargo}
cd "$(dirname "$0")/../.."

echo "== 1) device pass build =="
cargo build --manifest-path examples/resize_probe/Cargo.toml --lib --features step \
  --target dopamine-unknown-none-cvx
CVX=examples/resize_probe/target/dopamine-unknown-none-cvx/npu/resize_probe.cvx
CVXO=examples/resize_probe/target/dopamine-unknown-none-cvx/npu/resize_probe.cvxo
ls -la "$CVX" "$CVXO"

echo "== 2) docc calbin from-calvx =="
rm -rf /tmp/resize_pkg /tmp/resize_cal
mkdir -p /tmp/resize_pkg
cp "$CVX" /tmp/resize_pkg/step.cvx
: > /tmp/resize_pkg/param_blk0.bin
docc calbin from-calvx /tmp/resize_pkg -o /tmp/resize_cal

echo "== 3) docc calbin doctor（期望：拒收 0 条，exit 0） =="
docc calbin doctor /tmp/resize_cal
echo "doctor exit=$?"

echo "== 4) CCU ELF =="
ls -la /tmp/resize_cal/resize_pkg/*.elf
docc elf ccu view /tmp/resize_cal/resize_pkg/ccu0_chip0.elf | head -12

echo "== 5) 真机（有板时，主机侧） =="
cat <<'EOF'
dort calbin run -i 0 -f /tmp/resize_cal -m resize_pkg \
  --raw-in input_32x32x16_bf16.bin --raw-out out.bin \
  --in-tensor in_x --out-tensor out_y
# 然后把 out.bin 与 reference.py 生成的 expected_64x64x16_bf16.bin 逐元素比较
EOF
