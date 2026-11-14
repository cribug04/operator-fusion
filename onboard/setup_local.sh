#!/usr/bin/env bash
# 在 Jax 的 OrbStack calc Linux 中运行。只做本地构建，不运行或重置 NPU。
set -euo pipefail

KIT=$(cd "$(dirname "$0")/.." && pwd)
CALBLAS=${CALBLAS:-$HOME/bohack/calBLAS}
export PATH="$HOME/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=calculet
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || fail '请在 OrbStack 的 calc（Linux x86_64）中运行。'
for tool in git cargo rustc docc; do
  command -v "$tool" >/dev/null || fail "找不到 $tool，请检查 calculet 工具链。"
done
rustc --version
CALBLAS=$(cd "$CALBLAS" && pwd) || fail '找不到 calBLAS；默认路径为 ~/bohack/calBLAS。'
for repo in calDNN npu; do
  [[ -f "$CALBLAS/../$repo/Cargo.toml" ]] || fail "缺少同级仓库 $repo。"
done
[[ -f "$KIT/onboard/cases/conv_s2/param_blk0.bin" ]] || fail '验证数据不完整。'

PATCH="$KIT/patches/onboard_kit.patch"
if git -C "$CALBLAS" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  printf '补丁已经应用，继续构建。\n'
elif git -C "$CALBLAS" apply --check "$PATCH"; then
  git -C "$CALBLAS" apply "$PATCH"
else
  fail '补丁与现有代码不兼容；已停止，不会覆盖文件。请保留上面的错误信息。'
fi

# 每次生成独立目录，保留之前的构建和日志。
mkdir -p "$HOME/bohack/operator-fusion-builds"
OUT=$(mktemp -d "$HOME/bohack/operator-fusion-builds/run.XXXXXX")
printf '本次构建目录：%s\n' "$OUT"
CALBLAS="$CALBLAS" CASE_DIR="$KIT/onboard/cases" OUT="$OUT" \
  bash "$KIT/onboard/package_all.sh" 2>&1 | tee "$OUT/build-all.log"
printf '\n本地编译、打包及 doctor 检查通过。输出：%s\n' "$OUT"
printf '尚未进行 NPU 真机运行或数值、性能验证。\n'
