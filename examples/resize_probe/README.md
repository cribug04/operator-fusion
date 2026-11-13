# resize_bilinear2：Calculet NPU 双线性上采样 ×2 算子（探针 example）

本 example 用一个最小整步程序（`npu::decls!` + `#[npu::program]`）只跑一个 kernel：
**`resize_bilinear2`（NHWC bf16，[H][W][C] → [2H][2W][C]，align_corners=False）**。

## 文件

| 文件 | 说明 |
| --- | --- |
| `../../src/kernels/vision.rs` | 算子模板 `resize_bilinear2!`（追加在 `upsample2!` 之后） |
| `Cargo.toml` / `src/lib.rs` | 独立 example crate（自带 `[workspace]`，不影响 calBLAS 主工作区） |
| `src/kernels.rs` | 尺寸特化：32×32×16 ⇒ 64×64×16 |
| `src/step.rs` | 整步程序：`in_x`/`out_y` 两个 decl + `#[npu::program]` 调 kernel |
| `verify.sh` | WSL 内一键：设备遍编译 → `docc calbin from-calvx` → `docc calbin doctor` → 反汇编/查看 ELF |

## 编译 / 打包（WSL，工具链已安装）

```bash
cd ~/calBLAS
cargo build --manifest-path examples/resize_probe/Cargo.toml --lib --features step \
  --target dopamine-unknown-none-cvx
# => examples/resize_probe/target/dopamine-unknown-none-cvx/npu/resize_probe.cvx / .cvxo

rm -rf /tmp/resize_pkg /tmp/resize_cal && mkdir -p /tmp/resize_pkg
cp examples/resize_probe/target/dopamine-unknown-none-cvx/npu/resize_probe.cvx /tmp/resize_pkg/step.cvx
: > /tmp/resize_pkg/param_blk0.bin
docc calbin from-calvx /tmp/resize_pkg -o /tmp/resize_cal
docc calbin doctor /tmp/resize_cal        # 期望：拒收 0 条
```

## 已验证（本机 WSL2，2026-11-13）

- ✅ 设备遍编译：`cargo build ... --target dopamine-unknown-none-cvx` 成功（rustc 1.98.1-calculet）
- ✅ `docc calbin from-calvx` 成功：产出 2 个 CCU ELF（ccu0 19,560 B / ccu1 6,872 B）+ CPU 命令流
- ✅ `docc calbin doctor`：**拒收 0 条，exit 0**（2 条 ping/pong 地址警告，见下）
- ✅ 确定性：同一输入两次 `docc` 构建，calbin 逐字节相同（`docc calbin diff` 报「一致」）
- ✅ CCU ELF 反汇编：2849 条指令全部可识别（0 unknown，每个字节都被覆盖）；`docc elf ccu view` 显示 `load_size = 14608 = .text + .nops + .rodata` 各节之和
- ✅ CPU 参考：按 kernel 运算次序的 f32 结果与独立 float64 数学参照在全部 65,536 个输出上 bf16 舍入 100% 一致

## ❌ 本机无法验证（缺少条件）

- **没有 `cvx` 模拟器**（calvxus 不在赛题三个仓库，工具链也不带），也没有 NPU 真机（`dort overview` 报「未发现 Calculet 设备」）
  ⇒ **数值/真机行为尚未验证**；`input_32x32x16_bf16.bin` / `expected_64x64x16_bf16.bin` 已生成，有板即可跑：

```bash
dort calbin run -i 0 -f /tmp/resize_cal -m resize_pkg \
  --raw-in input_32x32x16_bf16.bin --raw-out out.bin \
  --in-tensor in_x --out-tensor out_y
# 比较 out.bin 与 expected_64x64x16_bf16.bin（都是 bf16 原始位型，64×64×16）
```

## 已知限制 / 后续

1. **单核**（`cores = 1`）：为绕开 docc 的「for 体内带结果 if 不支持」限制，采用单核 + 三段展开；后续可做 cores=2 版本（按行分核，需避开 clamp 条件分支）。
2. `docc` 的两条 ping/pong 地址警告：`mem.decl` 已标 `double_buffer`，但 docc 对「程序不切换 ping/pong」的场景仍把两份分到同一地址（仅警告，doctor 通过）；不影响单次推理。
3. `docc elf kabi` 对该 kernel 报 `calt-bind` 错误（docc 自己的诊断也说「多半是 docc 自己的缺陷」）；`docc calbin from-calvx` 路线不受影响。
4. 目前只支持 **×2、NHWC、C ≤ 128**；任意尺寸 / 任意倍率 / NCHW / 进 calDNN 图（Engine 注册）是下一步。
5. 性能未测（无板）；单核 + 每行重复计算 3 行，优化空间明确（行流水复用可减少 2/3 的行内插值计算）。
