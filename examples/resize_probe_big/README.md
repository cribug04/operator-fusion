# resize_bilinear2_tiled：通道分块版双线性上采样（大 C 支持）

本 example 验证 **`resize_bilinear2_tiled!`**：语义与 `resize_bilinear2!` 完全相同（NHWC bf16、×2、`align_corners=False`、边缘 clamp、f32 中间计算 + 最后一次 RNE 舍 bf16），
区别是把 C 切成 `NC = C/CC` 块顺序处理，使中间 cache 只占 `W × CC`，从而支持原版放不下的 **大通道数**。

- 本例：**32×32×256 bf16 → 64×64×256 bf16**，`CC = 64`（4 块）
- 原版上限 `C <= 128`（f32 中间 cache 一行 ≤ 512 B）；分块版约束：`CC % 32 == 0`、`C % CC == 0`、`W*CC*56 B <= 384 KiB`

## 构建 / 打包（WSL）

```bash
cd ~/calBLAS
cargo build --manifest-path examples/resize_probe_big/Cargo.toml --lib --features step \
  --target dopamine-unknown-none-cvx
# => .../npu/resize_probe_big.cvx (40,369 B) / .cvxo (7,118 B)

rm -rf /tmp/resize_big_pkg /tmp/resize_big_cal && mkdir -p /tmp/resize_big_pkg
cp examples/resize_probe_big/target/dopamine-unknown-none-cvx/npu/resize_probe_big.cvx /tmp/resize_big_pkg/step.cvx
: > /tmp/resize_big_pkg/param_blk0.bin
docc calbin from-calvx /tmp/resize_big_pkg -o /tmp/resize_big_cal   # exit 0
docc calbin doctor /tmp/resize_big_cal                               # 拒收 0 条
```

## 本机验证结果（2026-11-13，WSL2 + calculet 工具链）

| 项 | 结果 |
| --- | --- |
| 设备遍编译 | ✅ `resize_probe_big.cvx` 40,369 B / `.cvxo` 7,118 B（C=256，4 块） |
| `docc calbin from-calvx` | ✅ exit 0，ccu0 86,568 B / ccu1 6,872 B |
| `docc calbin doctor` | ✅ **拒收 0 条**（2 条 ping/pong 警告） |
| CCU 反汇编 | ✅ 15,997 条指令全部可识别（0 unknown，每字节覆盖）；`load_size 81612` = 各节之和 |
| 构建确定性 | ✅ 两次 calbin 逐字节相同 |
| CPU 参考 | ✅ 65,536×4 个输出与独立 float64 参照 **bf16 舍入 100% 一致**（≤0.996 ULP） |
| 真机数值 | ❌ 未验证（本机无板卡/无 `cvx` 模拟器） |

数据：`input_32x32x256_bf16.bin`（524,288 B）+ `expected_64x64x256_bf16.bin`（2,097,152 B），
真机命令同原版：`dort calbin run -i 0 -f /tmp/resize_big_cal -m resize_big_pkg --raw-in … --raw-out … --in-tensor in_x --out-tensor out_y`。

## 说明

- 通道分块不改数值（逐通道独立），只是把工作集切开；**同一套 CPU 参考/误差口径**。
- 生成代码里 `for cc in 0..NC` 是常数轮数循环（与 `conv2d!` 的 `for nb in 0..CO/NC` 同构），内层仍是单层运行时行循环。
- 代码由 `verify.sh` 一键复现。
