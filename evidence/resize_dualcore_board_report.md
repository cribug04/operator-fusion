# Resize 双核（`resize_bilinear2_tiled2`）真机上板报告 · 2026-11-14

> 结论：**双核版与单核版输出逐字节相同**（1 048 576 与 16 777 216 元素，均与 CPU 期望 EXACT）；
> **同形状墙钟中位数 `huge` 快 10%（52 → 47 ms）**，`big` 无可测收益（固定开销主导）。

## 0. 环境与来源

| 项 | 值 |
|---|---|
| 板机 | `calculet-MS-A2`，x86_64，`/dev/calculet0`（PCI `0000:07:00.0`，ID `20f5:c001`） |
| 驱动 / 固件 | Driver **0.9.3**；DRT **`74832e51`** · `0x03010101`（rt 3.1 / ABI 1.1，线协议 rev 2） |
| 链路 / 芯粒 | PCIe **16.0 GT/s x4**；`Dev: OS up  RT up  chips: 2` |
| 工具链 | rustc **1.98.1-calculet** · docc **0.4.1** · dort **0.4.1** |
| 程序来源 | 分支 `jax/resize-dualcore` @ `2ddef94`（代码提交）；本地以 LF 树 + 该分支 `patches/onboard_kit.patch` 与 `patches/caldnn_resize.patch` 重编 |
| 构建一致性 | 四个 `.cvx` 的 sha256 前缀与该分支 `docs/JUDGE_GUIDE.md §C` 记录**全部 MATCH**：big `5a22650a…` · big2 `e13fba79…` · huge `fc7f8dc0…` · huge2 `172a4ee9…` |
| 板锁 | 缺 `/opt/data/locks` ⇒ 全部带 `--board-lock external`；CalRT band 锁亦缺 ⇒ `启动流带：关`（性能降级，不影响数值） |

## 1. 数值（判据 `--mode exact`，逐位）

| 用例 | 元素数 | 逐位相同 | max\|Δ\| | 输出 sha256（前 16） |
|---|---|---|---|---|
| `resize_big`（单核） | 1 048 576 | **100%** | 0 | `6fb683176318acd3` |
| `resize_big2`（双核） | 1 048 576 | **100%** | 0 | `6fb683176318acd3` |
| `resize_huge`（单核） | 16 777 216 | **100%** | 0 | `5c1d4c42450b96ce` |
| `resize_huge2`（双核） | 16 777 216 | **100%** | 0 | `5c1d4c42450b96ce` |

- `cmp` 结论：**`big: single == dual`**、**`huge: single == dual`** ⇒ 两核按通道块各半，数值上完全透明。
- 交叉验证：`resize_big` 输出哈希与第一版 resize 上板包一致；`resize_huge` 输出哈希 = 用例 `expected.bin` 哈希 `5c1d4c42450b96ce276fc03bd6ca9735ce59d95d32a6eefcf844bc6f3bb46762`。
- 用例数据：`resize_huge` 由 `onboard/gen_resize_huge_case.py` 生成（输入 8 388 608 B sha256 `32ad08ef…`、期望 33 554 432 B sha256 `5c1d4c42…`），期望值经三路独立算法交叉验证 PASS（16 777 216 元素）。

## 2. 性能（墙钟，`dort calbin run` 整条命令，N=5）

| 用例 | 中位数 | min | max | 中位数比（单核/双核） |
|---|---|---|---|---|
| `resize_big` | 0.006 s | 0.006 | 0.006 | **1.04**（噪声级，无收益） |
| `resize_big2` | 0.006 s | 0.006 | 0.006 | |
| `resize_huge` | 0.052 s | 0.048 | 0.053 | **1.10**（双核快约 10%） |
| `resize_huge2` | 0.047 s | 0.046 | 0.047 | |

**口径说明（必须随数字一起报）**：计时包含 `dort` 进程启动、模型部署、输入输出搬运（`huge` 输入 8 MiB + 输出 32 MiB），
**不是 kernel 净时间**，只能作为同形状单核/双核的墙钟比值。

**粗估（模型，非实测）**：按 `T = 固定开销 + kernel` 两点拟合 —— `huge` 段 `K ≈ 10 ms → 5 ms`（约 2 倍），固定开销 `C ≈ 42 ms`；
`big` 段 kernel 时间远小于固定开销，故无可测收益。⇒ 要给出 kernel 净加速比需要 `debug.stamp` 插桩。

## 3. 复跑

```bash
tar -xzf resize_dualcore_onboard.tar.gz -C ~ && cd ~/rz2_pkg
for m in resize_big resize_big2 resize_huge resize_huge2; do : > $m/param_blk0.bin; done   # 无权重程序，0 字节 param
CAL_ROOT=$PWD N=5 bash run_resize_perf.sh        # 脚本默认 CAL_ROOT=$HERE/onboard_cal，本包需显式指定
```

## 4. 结论与边界

- **功能/数值**：双核 resize 在真机上与单核逐字节一致，可直接替换（约束：`C/CC` 为偶数、`CC` 为 ≥32 的 32 倍数、`W*CC*56 ≤ 384 KiB`）。
- **性能**：大图（8 MiB 输入级）墙钟 **1.10×**；小图无收益。kernel 净加速比未测。
- 未覆盖：`W*CC` 触到 VB 上限的形状、超过 2 核的分块、`stamp` 级逐 kernel 计时。
