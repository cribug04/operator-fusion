#!/bin/bash
# Rebuild + package + doctor for the channel-tiled resize probe (C=256), run inside WSL from calBLAS root.
set -uo pipefail
export PATH=/root/.cargo/bin:$PATH
export RUSTUP_HOME=${RUSTUP_HOME:-/root/.rustup}
export CARGO_HOME=${CARGO_HOME:-/root/.cargo}
cd "$(dirname "$0")/../.."

cargo build --manifest-path examples/resize_probe_big/Cargo.toml --lib --features step \
  --target dopamine-unknown-none-cvx
CVX=examples/resize_probe_big/target/dopamine-unknown-none-cvx/npu/resize_probe_big.cvx
rm -rf /tmp/resize_big_pkg /tmp/resize_big_cal
mkdir -p /tmp/resize_big_pkg
cp "$CVX" /tmp/resize_big_pkg/step.cvx
: > /tmp/resize_big_pkg/param_blk0.bin
docc calbin from-calvx /tmp/resize_big_pkg -o /tmp/resize_big_cal
docc calbin doctor /tmp/resize_big_cal
echo "doctor exit=$?"
docc elf ccu view /tmp/resize_big_cal/resize_big_pkg/ccu0_chip0.elf | head -12
