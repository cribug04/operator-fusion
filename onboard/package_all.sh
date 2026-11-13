#!/usr/bin/env bash
# 在 Linux 开发机（已装 calculet 工具链）上：编译 4 个 example + docc 打包 calbin + doctor。
# 用法：
#   CALBLAS=~/calBLAS CASE_DIR=./cases OUT=./onboard_cal bash package_all.sh
# 前置：已应用代码补丁（calBLAS/examples/resize_probe{,_big}、conv2d_s2_probe、conv2d_s2_ref 存在）
#      conv 的 param_blk0.bin 由 cases/conv_s2/param_blk0.bin 提供（gen_conv_s2_case.py 生成）。
set -euo pipefail

CALBLAS=${CALBLAS:-$HOME/calBLAS}
CASE_DIR=${CASE_DIR:-$(cd "$(dirname "$0")" && pwd)/cases}
OUT=${OUT:-$(cd "$(dirname "$0")" && pwd)/onboard_cal}
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_HOME=${CARGO_HOME:-$HOME/.cargo}
export RUSTUP_HOME=${RUSTUP_HOME:-$HOME/.rustup}

fail() { echo "ERROR: $*" >&2; exit 1; }
[ -d "$CALBLAS" ] || fail "找不到 CALBLAS=$CALBLAS（指向 calBLAS 仓库根目录）"
command -v cargo >/dev/null || fail "cargo 不在 PATH"
command -v docc  >/dev/null || fail "docc 不在 PATH（工具链没装好？）"
[ -f "$CASE_DIR/conv_s2/param_blk0.bin" ] || fail "缺少 $CASE_DIR/conv_s2/param_blk0.bin（先跑 python3 gen_conv_s2_case.py）"

cd "$CALBLAS" || fail "cd $CALBLAS 失败"
mkdir -p "$OUT/pkg" "$OUT/logs"
build_and_pkg() {
  local example=$1 crate=$2 model=$3 param=$4
  echo "===== [$model] cargo build ($example) ====="
  cargo build --manifest-path "$CALBLAS/examples/$example/Cargo.toml" --lib --features step \
      --target dopamine-unknown-none-cvx
  local cvx="$CALBLAS/examples/$example/target/dopamine-unknown-none-cvx/npu/$crate.cvx"
  [ -f "$cvx" ] || fail "没有 $cvx"
  local pkg="$OUT/pkg/${model}_pkg"
  rm -rf "$pkg" "$OUT/$model"
  mkdir -p "$pkg"
  cp "$cvx" "$pkg/step.cvx"
  if [ "$param" = "empty" ]; then : > "$pkg/param_blk0.bin"; else cp "$param" "$pkg/param_blk0.bin"; fi
  echo "===== [$model] docc calbin from-calvx ====="
  docc calbin from-calvx "$pkg" -o "$OUT/$model" 2>&1 | tee "$OUT/logs/from-calvx-$model.log"
  echo "===== [$model] docc calbin doctor ====="
  docc calbin doctor "$OUT/$model" 2>&1 | tee "$OUT/logs/doctor-$model.log"
  grep -q "拒收 0 条" "$OUT/logs/doctor-$model.log" || fail "$model doctor 有拒收，见 $OUT/logs/doctor-$model.log"
}

build_and_pkg resize_probe     resize_probe     resize        empty
build_and_pkg resize_probe_big resize_probe_big resize_big    empty
build_and_pkg conv2d_s2_probe  conv2d_s2_probe  conv_s2       "$CASE_DIR/conv_s2/param_blk0.bin"
build_and_pkg conv2d_s2_ref    conv2d_s2_ref    conv_s2_ref   "$CASE_DIR/conv_s2/param_blk0.bin"

echo
echo "################ 打包完成 ################"
echo "calbin 目录："
for n in resize resize_big conv_s2 conv_s2_ref; do echo "  $OUT/$n"; done
echo
echo "真机上依次运行："
cat <<EOF
dort device -i 0 | grep "Dev  :"    # 需要 OS up  RT up

dort calbin run -i 0 -f $OUT/resize      -m resize_pkg      --raw-in $CASE_DIR/resize_small/input.bin --raw-out out_resize.bin      --in-tensor in_x --out-tensor out_y
dort calbin run -i 0 -f $OUT/resize_big  -m resize_big_pkg  --raw-in $CASE_DIR/resize_big/input.bin   --raw-out out_resize_big.bin  --in-tensor in_x --out-tensor out_y
dort calbin run -i 0 -f $OUT/conv_s2     -m conv_s2_pkg     --raw-in $CASE_DIR/conv_s2/input.bin     --raw-out out_conv_fast.bin   --in-tensor in_x --out-tensor out_y
dort calbin run -i 0 -f $OUT/conv_s2_ref -m conv_s2_ref_pkg --raw-in $CASE_DIR/conv_s2/input.bin     --raw-out out_conv_ref.bin    --in-tensor in_x --out-tensor out_y

# 对比（在能跑 python3+numpy 的机器上）：
python3 compare_bf16.py out_resize.bin     $CASE_DIR/resize_small/expected.bin --label resize
python3 compare_bf16.py out_resize_big.bin $CASE_DIR/resize_big/expected.bin   --label resize_big
python3 compare_bf16.py out_conv_fast.bin  $CASE_DIR/conv_s2/expected.bin --actual-shape 84,84,64 --crop 2:82,2:82 --expected-shape 80,80,64 --mode conv-cpu --label conv_fast_vs_cpu
python3 compare_bf16.py out_conv_ref.bin   $CASE_DIR/conv_s2/expected.bin --actual-shape 84,84,64 --crop 2:82,2:82 --expected-shape 80,80,64 --mode conv-cpu --label conv_ref_vs_cpu
python3 compare_bf16.py out_conv_fast.bin  out_conv_ref.bin --actual-shape 84,84,64 --crop 2:82,2:82 --expected-shape 84,84,64 --expected-crop 2:82,2:82 --mode exact --label conv_fast_vs_ref
EOF
