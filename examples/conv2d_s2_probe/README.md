# conv2d_s2_fast（实验原型）：stride-2 3×3 卷积的 FAST 路

**状态：实验原型（未进共享 `conv2d!` 模板、未接任何模型、数值/性能未验证）**。
目的：验证 `conv2d!` 只对 `S == 1` 开 FAST 路（`vision.rs:122`）的问题能否用“偶/奇像素分块”解决——YOLO26n 有 6 个 stride-2 卷积只能走慢路。

## 思路

对输出行 `oy`，3 个输入行（padded 行 `oy*2+ky+1`）各拆成 **奇像素 / 偶像素两份 VB cache**（从 DDR 按像素行距 `2*CI` 直接绑视图加载；odd 起点 padded 列 `B-1`、even 起点 `B`）。
于是 9 个 tap 的 lhs 都是某份 cache 里的**连续 16 行**：

| tap kx | lhs 来源（padded 列 = 2·ox + kx + 1） |
| --- | --- |
| 0 | `odd[ky]` 行 `mb*16`（odd 从 padded 列 B-1 起） |
| 1 | `even[ky]` 行 `mb*16`（even 从 padded 列 B 起） |
| 2 | `odd[ky]` 行 `mb*16+1` |

这样 stride-2 也能享受“输入行只从 DDR 读一次、9 个 tap 从 VB 取 lhs”的快路。

## 本例

YOLO26n 的 `conv3s2_80x80_64to64` 同形状：**80×80×64 → 40×40×64，k=3、s=2、silu**，输入 padded NHWC `[164][164][64]`，
权重 `[9][64][64]` bf16（BN 已折入），bias f32 `[64]`（放在 `@w` 偏移 73728），双核（`cores = 2`，按输出行分）。

## 构建/打包（WSL）

```bash
cd ~/calBLAS
cargo build --manifest-path examples/conv2d_s2_probe/Cargo.toml --lib --features step \
  --target dopamine-unknown-none-cvx
# => conv2d_s2_probe.cvx (14,496 B) / .cvxo (2,608 B)

# docc：calbin 需要参数块（本原型用全 0 占位；长度必须等于 @w 声明）
rm -rf /tmp/conv_s2_pkg /tmp/conv_s2_cal && mkdir -p /tmp/conv_s2_pkg
cp examples/conv2d_s2_probe/target/dopamine-unknown-none-cvx/npu/conv2d_s2_probe.cvx /tmp/conv_s2_pkg/step.cvx
truncate -s 73984 /tmp/conv_s2_pkg/param_blk0.bin
docc calbin from-calvx /tmp/conv_s2_pkg -o /tmp/conv_s2_cal   # exit 0
docc calbin doctor /tmp/conv_s2_cal                           # 拒收 0 条
```

## 本机验证结果（2026-11-13，仅平台侧）

| 项 | 结果 |
| --- | --- |
| 设备遍编译 | ✅ `.cvx` 14,496 B / `.cvxo` 2,608 B |
| `docc calbin from-calvx` | ✅ exit 0；权重块 73,984 B；ccu0/ccu1 ELF 各 **22,168 B**（双核各编一份） |
| `docc calbin doctor` | ✅ **拒收 0 条**（2 条 ping/pong 警告） |
| 数值正确性 | ❌ **未验证**（无板卡、无 `cvx` 模拟器；TU 累加次序与 numpy 不可能逐位） |
| 性能 | ❌ **未验证**（需真机对比慢路） |

## 后续

1. **上板前先做 CPU 标定**：用 numpy 生成“同布局输入 + 权重 + bias”，按 TU 无法复刻的累加次序给容差口径；
2. 上板与慢路版本（同形状 `conv2d!`）比：① 输出 max diff/余弦；② 单算子耗时；
3. 通过后把分支合并进 `conv2d!`（`S == 2` 时走这套 cache），并接入 YOLO26n 的 6 个 conv3s2；
4. 推广：`dwconv3!`/`maxpool5!` 的 stride-2 变体同样可用这个“偶/奇分块”技巧。
