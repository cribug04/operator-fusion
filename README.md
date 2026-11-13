# operator-fusion — Calculet NPU 算子（Resize / stride-2 卷积原型）与勘察、真机验证工具包

本仓库是 **Calculet NPU CalVX 黑客松**工作的快照：从零实现/验证的 CV 算子、优化原型、仓库勘察报告，以及一套**可直接在真机上执行**的验证工具包。

> 一句话：**Bilinear Resize ×2（含通道分块版）已在 calculet 工具链上完成"编译→CalVX→docc calbin→doctor"全链路平台验证；stride-2 FAST 卷积原型已编译/打包通过；真机数值验证所需的 calbin、数据、脚本、判定标准与排错表全部就绪。**

---

## 1. 内容总览

| 目录 | 内容 |
| --- | --- |
| [`docs/CALCULET_RECON.md`](docs/CALCULET_RECON.md) | 仓库勘察报告：环境/工具链实测、算子覆盖、docc/dort 真实用法、真机测试证据、缺失算子与选题建议、48h 分工 |
| [`docs/RESIZE_OPERATOR.md`](docs/RESIZE_OPERATOR.md) | **Resize 算子交付说明**：语义、代码位置、构建/打包/验证数据、真机步骤、已知限制 |
| [`docs/OPTIMIZATION_CANDIDATES.md`](docs/OPTIMIZATION_CANDIDATES.md) | **还能优化什么**：按 ROI 排序的算子/框架优化清单 + 本轮落地的两项 |
| [`docs/CALDNN_INTEGRATION.md`](docs/CALDNN_INTEGRATION.md) | **Resize 接入 calDNN 图 / 引擎表**：`ResizeDesc` + `Op::Resize` + `Graph::resize` + calBLAS `Resize` 引擎 + 探针引擎表；主机侧编排/bind 演示与 5 项测试；未发射 |
| [`docs/REAL_DEVICE_VERIFICATION.md`](docs/REAL_DEVICE_VERIFICATION.md) | **真机验证操作手册**（逐条命令、判定标准、排错表、记录模板） |
| [`kernels/`](kernels/) | 3 个新宏的源码（与补丁内容一致，便于审阅） |
| [`examples/`](examples/) | 4 个可直接编译的探针程序（独立 workspace，不污染 calBLAS 主工作区） |
| [`patches/onboard_kit.patch`](patches/onboard_kit.patch) | 应用到 calBLAS 的完整补丁（`vision.rs` 3 宏 + `engines.rs` Resize 引擎 + 4 个 example 与引擎表、`resize-plan` 主机演示） |
| [`patches/caldnn_resize.patch`](patches/caldnn_resize.patch) | 应用到 calDNN 的补丁：`Interp` / `ResizeDesc` / `Op::Resize` / `Graph::resize` |
| [`onboard/`](onboard/) | 真机验证工具包：打包、上板、bf16 对比、数据生成；`cases/` 含已生成的输入/期望数据 |
| [`evidence/`](evidence/) | 平台侧验证证据：`.cvx/.cvxo`、calbin（含 CCU ELF）、doctor 日志、反汇编摘要、确定性 sha256 |

---

## 2. 三个算子/原型

| 名称 | 功能 | 状态 |
| --- | --- | --- |
| `resize_bilinear2` | **双线性上采样 ×2**（NHWC bf16，`align_corners=False`，边缘 clamp；f32 中间计算 + RNE 舍 bf16）；`[H][W][C] → [2H][2W][C]` | ✅ 编译 / docc calbin / doctor / 确定性 / 反汇编 / CPU 参考全过；✅ 已注册进 calDNN 图与引擎表（主机侧编排通过，未发射）；❌ 真机数值待板 |
| `resize_bilinear2_tiled` | 同上，**通道分块版**：按 `CC` 把 C 切块，支持大 C（实测 C=256；原版上限 C≤128） | ✅ 同上（C=256 全链路通过） |
| `conv2d_s2_fast`（实验） | **3×3、stride=2 卷积 + bias + SiLU** 的 FAST 路原型：偶/奇像素各一份 VB cache，让 tap 的 lhs 都成为连续 16 行；用于补齐 `conv2d!` 只对 `S==1` 开 FAST 的缺口 | ✅ 编译 / docc calbin / doctor / 反汇编；❌ 数值与性能待板；**未合入共享 `conv2d!`**（不影响现有模型），配同形状慢路对照 [`examples/conv2d_s2_ref`](examples/conv2d_s2_ref) |

平台侧实测数字（本机 WSL2 + calculet 工具链 `rustc 1.98.1` / `docc 0.4.1` / `dort 0.4.1`）：

| 用例 | `.cvx` | calbin ELF (ccu0/ccu1) | ELF load_size | 反汇编 | doctor |
| --- | --- | --- | --- | --- | --- |
| resize（32×32×16→64×64×16） | 35,160 B | 19,560 / 6,872 B | 14,608 | 2,849 条，0 unknown | 拒收 0 条 |
| resize_tiled（32×32×256→64×64×256） | 40,369 B | 86,568 / 6,872 B | 81,612 | 15,997 条，0 unknown | 拒收 0 条 |
| conv_s2_fast（80×80×64→40×40×64） | 14,496 B | 22,168 / 22,168 B | 17,216 | 2,699 条，0 unknown | 拒收 0 条 |
| conv_s2_ref（同形状慢路） | 15,241 B | 16,152 / 16,152 B | 11,204 | 1,897 条，0 unknown | 拒收 0 条 |

> 两次 `docc calbin from-calvx` 构建的 calbin **逐字节相同**；resize 两个用例的 CPU 参考（numpy 复刻运算次序）与独立 float64 数学参照在全部输出上 **bf16 舍入 100% 一致**（≤0.996 ULP）。

---

## 3. 快速使用

### 3.1 应用到 calBLAS（三仓库同级）

```bash
for r in calBLAS calDNN npu; do git clone http://115.159.82.89/git/calculet/$r.git; done
cd calDNN && git apply /path/to/patches/caldnn_resize.patch && cd ..
# 期望：M src/graph.rs src/lib.rs（Interp / ResizeDesc / Op::Resize / Graph::resize）
cd calBLAS && git apply /path/to/patches/onboard_kit.patch
# 期望：M src/kernels/vision.rs src/engines.rs；新增 examples/resize_probe{,_big}/、examples/conv2d_s2_{probe,ref}/
# 图接入演示（主机侧，不上板）：cd examples/resize_probe && cargo run -p resize-plan && cargo test -p resize-plan
```

### 3.2 编译 + docc 打包（Linux + calculet 工具链）

```bash
cd onboard
CALBLAS=$HOME/calBLAS CASE_DIR=$PWD/cases OUT=$PWD/onboard_cal bash package_all.sh
# 产出 onboard_cal/{resize,resize_big,conv_s2,conv_s2_ref}（calbin 目录）
```

### 3.3 真机验证

按 [`docs/REAL_DEVICE_VERIFICATION.md`](docs/REAL_DEVICE_VERIFICATION.md) 执行（健康检查 → `dort calbin run` → 拉回输出 → `compare_bf16.py` 对比 → 判定/记录）。一键脚本：

```bash
DEVICE=0 CAL_ROOT=$PWD/onboard_cal CASE_DIR=$PWD/cases OUT=$PWD/board_out bash run_on_board.sh
```

**判定标准（摘要）**：
- resize 两个用例 vs CPU 参考：逐位相同（允许极少数 1 ULP，因 f32 中间合同为 `rnu`、numpy 为 RNE）；
- conv_fast vs 同硬件慢路 `conv2d_s2_ref`：**必须逐位相同**（同 tap/累加次序，仅 lhs 来源不同）——这是严格判据；
- conv vs CPU 参考：宽松（cos ≥ 0.9999、≤4 ULP；TU 分块累加 + SiLU 近似决定不可能逐位）。

---

## 4. 开发环境（本机验证时）

- Windows 11 + **WSL2**（WSL 3.0.5，Ubuntu 24.04.3，kernel 6.18.58.1-microsoft-standard-WSL2）
- rustup 1.29.3 + **calculet 工具链**：`rustc 1.98.1`、`cargo 1.98.1`、`docc 0.4.1`、`dort 0.4.1`
- 三仓库 commit：calBLAS `32d2bb8`，calDNN `1e06c41`，npu `172aad2`（2026-11-13 下发版）

> 官方要求 **Linux x86_64**；Windows 原生装不上（发行包无 Windows host，实测报错），所以走 WSL2。

---

## 5. 已知限制 / 下一步

1. **真机数值未验证**：本机无板卡、无 `cvx` 模拟器；数据/calbin/脚本/判定标准已备好（`onboard/`）。
2. `resize`：单核、固定 ×2、NHWC、`C≤128`（tiled 版按 `CC` 放宽到约 `W*CC*56B<=384KiB`）；行流水优化已分析但**因无板无法数值验证而暂缓**。
3. `conv2d_s2_fast`：实验原型，未合入共享 `conv2d!`；需先做同硬件慢路对拍 + 性能 A/B，再考虑合并/接入 YOLO26n。
4. **图/引擎接入未做**：CV 算子目前无法通过 calDNN 引擎表注册、被图调用（路线 A 要求），需跨仓库改 `calDNN`（Desc + Op + Graph 方法 + Engine），估计 1.5–2 天。
5. 更完整的优化清单（LLM/框架方向）见 `docs/OPTIMIZATION_CANDIDATES.md`。

---

## 6. 说明

- 本仓库为 2026-11-13 黑客松工作快照；所有"已验证"数字均来自本机真实执行（日志与产物在 `evidence/`），**未编造 API 或结果**；未验证项在文档中显式标注。
- 代码采用与 calBLAS 仓库一致的风格；所有原语调用均来自仓库真实 API（`npu/src/lib.rs`、calBLAS 模板），未猜测接口。
- 真机验证后请把 `board_out/`（输出 + 日志）与 `docs/REAL_DEVICE_VERIFICATION.md` §8 记录模板一起归档/提 PR。

## 本地复现与验收更新

Jax 的环境与部署记录见 [JAX_LOCAL_SETUP.md](docs/JAX_LOCAL_SETUP.md)。测试工具修复、40 项回归测试（含 19 项 Resize CPU 边界测试）及两组 Resize CPU 数据复核结果见 [LOCAL_VALIDATION.md](docs/LOCAL_VALIDATION.md)。Resize 接入 calDNN 图与引擎表的改动、演示输出和测试见 [CALDNN_INTEGRATION.md](docs/CALDNN_INTEGRATION.md)。这些本地结果不等于 NPU 真机验证。
