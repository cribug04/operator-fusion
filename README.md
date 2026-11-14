# operator-fusion — Calculet NPU：双线性 Resize 算子（路线 A）与 SD1.5 VAE 解码器移植（路线 B）

BoHack 2026 · Calculet NPU CalVX 算子与端到端模型移植赛题 · 团队「你说的都队」（吴羿辰、吴文宇、赵宇涵、邓思涵）· 2026-11-13 14:00 开赛，本仓库首个提交为 `5fee97b`（11-13 22:55）。提交记录支持以下开发时间线，不能单独证明开赛前无已有代码。

**仓库**：[GitHub](https://github.com/cribug04/operator-fusion)（主）· [Gitee](https://gitee.com/nssdd/operator-fusion)（镜像，同一 master）

## 0. 评委请先看这里

| 成果 | 真机结果 | 平台侧 / 本地 | 入口 |
| --- | --- | --- | --- |
| **路线 A · Bilinear Resize ×2 算子**（NHWC bf16，普通版 C≤128 + 通道分块版 C=256） | ✅ 两个尺寸输出与 CPU 参考**逐位相同**，1,114,112 个元素 0 差异，三次重跑同哈希 ※ | 设备端编译 → CalVX → docc calbin → doctor 拒收 0；43 项 CPU 测试（13 种尺寸三路独立算法逐位一致）；已注册进 calDNN 图与引擎表，Plan / bind 通过 | [RESIZE_OPERATOR.md](docs/RESIZE_OPERATOR.md) · [LOCAL_VALIDATION.md](docs/LOCAL_VALIDATION.md) · [CALDNN_INTEGRATION.md](docs/CALDNN_INTEGRATION.md) |
| **路线 B · SD1.5 VAE 解码器全 NPU 移植**（latent [4,64,64] → 512×512，5 段 CalVX 程序，35 conv + 30 GN + 1 注意力） | ✅ 5/5 段跑通出图；与 diffusers fp32 参考 **PSNR 52.58 dB / max\|Δ\| 0.0536**（判据 ≥40 dB / ≤0.08，通过）；整链 34.6 s（含部署开销，板端流带未开启，非正式性能数据）※ | 5 段编译 / docc 拒收 0 / 构建逐字节确定；本地黄金仿真 52.83 dB；审查发现并修复生成器两处错误后上板 | [SD_VAE_ON_NPU.md](docs/SD_VAE_ON_NPU.md) · [SD_VAE_REAL_DEVICE_VERIFICATION.md](docs/SD_VAE_REAL_DEVICE_VERIFICATION.md) · [REVIEW_SD_VAE_2026-11-14.md](docs/REVIEW_SD_VAE_2026-11-14.md) |
| Resize **双核版** + 单核/双核同形状计时程序 | ❌ 未上板，无性能数据 | 编译 / docc / doctor 拒收 0；CPU 分配检查通过 | [RESIZE_DUALCORE.md](docs/RESIZE_DUALCORE.md) |
| stride-2 卷积 FAST 原型 `conv2d_s2_fast` | ❌ 未上板 | 编译 / docc / doctor 拒收 0 | §2 |

※ 真机数字来源：2026-11-14 晚队友在 calculet-MS-A2（2 芯粒，dort 0.4.1）上的运行报告，运行的是审查修复版程序（提交 `4cea49b`）。板上原始输出、`timing.txt`、`compare.json` 正在入库，入库后此处换成文件链接。**在那之前，这一列是转述，不是仓库内可核验的证据。**

**10 分钟内无板子能复现什么**：见 [JUDGE_GUIDE.md](docs/JUDGE_GUIDE.md)（calDNN 图演示、43 项测试、CPU 参考复核；有 calculet 工具链时 7 个程序的编译与 docc doctor）。

**42 小时时间线**（Git 提交记录，北京时间）：

| 时间 | 谁 | 事 |
| --- | --- | --- |
| 11-13 14:00 | — | 开赛 |
| 11-13 22:55 | cribug04 | 初始提交：Resize 两版、stride-2 卷积原型、勘察报告、上板工具包 |
| 11-13 23:47 – 11-14 00:29 | Jax（092135） | 验收工具修复与回归测试、19 项 Resize CPU 边界测试、Resize 接入 calDNN 图与引擎表 |
| 11-14 14:11 – 14:43 | nssdd | SD1.5 VAE 解码器 5 段移植、生成器 / 打包器 / 仿真器、真机手册 |
| 11-14 15:37 / 17:02 | Jax | VAE 代码审查（两条阻断级发现）、生成器修复并重新生成 5 段 |
| 11-14 19:17 – 19:33 | nssdd / Jax | 参考图入库；两个分支合并进 master |
| 11-14 20:16 | Jax | Resize 双核版与计时程序 |
| 11-14 晚 | 队友 | 真机运行 Resize 两例与 VAE 5 段（见 ※） |

---

## 1. 内容总览

| 目录 | 内容 |
| --- | --- |
| [`docs/SD_VAE_ON_NPU.md`](docs/SD_VAE_ON_NPU.md) | **SD1.5 VAE 解码器全 NPU 移植**：5 段整步程序、层表、布局/自定义 kernel、权重重排、平台与仿真证据、上板步骤与判据 |
| [`docs/SD_VAE_REAL_DEVICE_VERIFICATION.md`](docs/SD_VAE_REAL_DEVICE_VERIFICATION.md) | **SD VAE 真机验证手册**：逐条命令、每段字节数、判据、排错表、记录模板 |
| [`sd_vae/SMOKE.md`](sd_vae/SMOKE.md) | 模板/语言实测笔记：8 个模板确切写法、`docc` 两个 bug 的完整记录与绕法、`attention!` 头维上限实证 |
| [`sd_vae/`](sd_vae/) | SD VAE 的生成器/打包器/仿真器、5 段 kernel 源码、上板包脚本与证据 |
| [`docs/CALCULET_RECON.md`](docs/CALCULET_RECON.md) | 仓库勘察报告：环境/工具链实测、算子覆盖、docc/dort 真实用法、真机测试证据、缺失算子与选题建议、48h 分工 |
| [`docs/RESIZE_OPERATOR.md`](docs/RESIZE_OPERATOR.md) | **Resize 算子交付说明**：语义、代码位置、构建/打包/验证数据、真机步骤、已知限制 |
| [`docs/OPTIMIZATION_CANDIDATES.md`](docs/OPTIMIZATION_CANDIDATES.md) | **还能优化什么**：按 ROI 排序的算子/框架优化清单 + 本轮落地的两项 |
| [`docs/CALDNN_INTEGRATION.md`](docs/CALDNN_INTEGRATION.md) | **Resize 接入 calDNN 图 / 引擎表**：`ResizeDesc` + `Op::Resize` + `Graph::resize` + calBLAS `Resize` 引擎 + 探针引擎表；主机侧编排/bind 演示与 5 项测试；未发射 |
| [`docs/REAL_DEVICE_VERIFICATION.md`](docs/REAL_DEVICE_VERIFICATION.md) | **真机验证操作手册**（逐条命令、判定标准、排错表、记录模板） |
| [`kernels/`](kernels/) | 4 个新宏的源码（与补丁内容一致，便于审阅） |
| [`examples/`](examples/) | 7 个可直接编译的探针程序（独立 workspace，不污染 calBLAS 主工作区） |
| [`patches/onboard_kit.patch`](patches/onboard_kit.patch) | 应用到 calBLAS 的完整补丁（`vision.rs` 4 宏 + `engines.rs` Resize 引擎 + 7 个 example 与引擎表、`resize-plan` 主机演示） |
| [`patches/caldnn_resize.patch`](patches/caldnn_resize.patch) | 应用到 calDNN 的补丁：`Interp` / `ResizeDesc` / `Op::Resize` / `Graph::resize` |
| [`onboard/`](onboard/) | 真机验证工具包：打包、上板、bf16 对比、数据生成；`cases/` 含已生成的输入/期望数据 |
| [`evidence/`](evidence/) | 平台侧验证证据：`.cvx/.cvxo`、calbin（含 CCU ELF）、doctor 日志、反汇编摘要、确定性 sha256 |

---

## 2. 三个算子/原型

| 名称 | 功能 | 状态 |
| --- | --- | --- |
| `resize_bilinear2` | **双线性上采样 ×2**（NHWC bf16，`align_corners=False`，边缘 clamp；f32 中间计算 + RNE 舍 bf16）；`[H][W][C] → [2H][2W][C]` | ✅ 编译 / docc calbin / doctor / 确定性 / 反汇编 / CPU 参考全过；✅ 已注册进 calDNN 图与引擎表（主机侧编排通过，未发射）；✅ **真机逐位相同**（见 §0 ※） |
| `resize_bilinear2_tiled` | 同上，**通道分块版**：按 `CC` 把 C 切块，支持大 C（实测 C=256；原版上限 C≤128） | ✅ 同上（C=256 全链路通过）；✅ 真机逐位相同（见 §0 ※） |
| `resize_bilinear2_tiled2` | 分块版的**双核版**：两核各算一半通道块，数值与分块版逐位相同；配套 `resize_probe_big2` / `resize_probe_huge{,2}` 三个程序做单核 vs 双核同形状计时 | ✅ 编译 / docc / doctor 拒收 0；✅ CPU 分配检查；❌ **真机未跑，无性能数据**，见 [`docs/RESIZE_DUALCORE.md`](docs/RESIZE_DUALCORE.md) |
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

## 2.5 Stable Diffusion 1.5 · VAE 解码器全 NPU 移植（2026-11-14）

把 SD1.5 的 **VAE decoder**（latent `[4,64,64]` → 512×512 RGB 图像）完整做成 5 段 CalVX 整步程序（`a→b1→b2a→b2b→c`），一段一个 calbin、板上串行喂 latent 出图。详见 [`docs/SD_VAE_ON_NPU.md`](docs/SD_VAE_ON_NPU.md)。

| 段 | 输入 → 输出 | `.cvx` B | 参数块 B | ccu0/ccu1 ELF B | doctor |
| --- | --- | --- | --- | --- | --- |
| a | latent → 64²×512（conv_in + mid resnet×2 + 单头注意力） | 243,366 | 21,483,520 | 1,414,672 / 1,411,672 | 拒收 0 |
| b1 | 64²×512 → 128²×512（up_blocks.0 + nearest×2+conv） | 156,915 | 33,452,032 | 968,536 / 966,160 | 拒收 0 |
| b2a | 128²×512 → 128²×512（up_blocks.1 的 3 个 resnet） | 68,253 | 28,729,344 | 398,884 / 397,144 | 拒收 0 |
| b2b | 128²×512 → 256²×512（nearest×2 + conv） | 22,461 | 4,722,688 | 31,160 / 30,700 | 拒收 0 |
| c | 256²×512 → 512×512 图（up_blocks.2/3 + norm + conv_out） | 270,253 | 12,292,096 | 1,528,008 / 1,525,764 | 拒收 0 |
| 合计 | — | **761,248** | **100,679,680** | **4,341,260 / 4,331,440** | — |

**本地数值**：用 torch 按 `step.rs` 的算子顺序与我们的布局**仿真同一份程序**，与 diffusers fp32 黄金参考（确定性 decode）比较：最终图 **PSNR 52.83 dB**、max\|Δ\| 0.0473、mean\|Δ\| 0.00336；四个中间段边界 26.3–43.9 dB（张量数值范围 ±233，bf16 存储误差）。**真机：PSNR 52.58 dB / max\|Δ\| 0.0536，通过判据**（见 §0 ※）。2026-11-14 晚审查发现并修复生成器两处错误（单核调用两核分块模板、NCHW 暂存 0 字节），上表为修复后重新编译打包的数字，详见 [`docs/REVIEW_SD_VAE_2026-11-14.md`](docs/REVIEW_SD_VAE_2026-11-14.md)。

两个工程发现（写进文档）：
1. **docc 0.4.1 的 riscvc 21 位分支限制**：内核对 `for part in 0..NH` 的常量上界会被整段展开，`group_norm` 在 128²×512 时单函数 `.text ≈ 1.12 MB` > ±1,048,575 B ⇒ 链接失败（实测 hw=4096/8192/12288/16384 → .text 282K/560K/831K/1,116K）。把 NH 做成普通标量实参无效（内联+常量传播），只能从 SRAM 读；我们改为自写 `gn_stats`/`gn_apply` 两个 kernel（运行期循环上界），同时把 GN 参数从 313 MB 缩到 96 MB。
2. **docc calt-fate 的 GOB 累加串次序**：注意力里「分数链 + 4 个 PV 累加器链」不能同循环在飞；拆成「先算 p 存 VB、再单独累加 PV」即通过（与 conv2d FAST 四累加器同构）。

---

## 3. 快速使用

### 3.1 应用到 calBLAS（三仓库同级）

```bash
for r in calBLAS calDNN npu; do git clone http://115.159.82.89/git/calculet/$r.git; done
cd calDNN && git apply /path/to/patches/caldnn_resize.patch && cd ..
# 期望：M src/graph.rs src/lib.rs（Interp / ResizeDesc / Op::Resize / Graph::resize）
cd calBLAS && git apply /path/to/patches/onboard_kit.patch
# 期望：M src/kernels/vision.rs src/engines.rs；新增 examples/resize_probe{,_big,_big2,_huge,_huge2}/、examples/conv2d_s2_{probe,ref}/
# 图接入演示（主机侧，不上板）：cd examples/resize_probe && cargo run -p resize-plan && cargo test -p resize-plan
```

### 3.2 编译 + docc 打包（Linux + calculet 工具链）

```bash
cd onboard
CALBLAS=$HOME/calBLAS CASE_DIR=$PWD/cases OUT=$PWD/onboard_cal bash package_all.sh
# 产出 onboard_cal/{resize,resize_big,conv_s2,conv_s2_ref,resize_big2,resize_huge,resize_huge2}（calbin 目录）
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

1. **真机已验证**：Resize 两例、VAE 5 段（§0）。**未上板**：Resize 双核版、`conv2d_s2_fast`。真机原始产物尚未入库。
2. `resize`：固定 ×2、NHWC、`C≤128`（tiled 版按 `CC` 放宽到约 `W*CC*56B<=384KiB`）；已上板的两个包是单核版；双核版已编译待板（[`docs/RESIZE_DUALCORE.md`](docs/RESIZE_DUALCORE.md)）；行流水优化未做。
3. `conv2d_s2_fast`：实验原型，未合入共享 `conv2d!`；需先做同硬件慢路对拍 + 性能 A/B，再考虑合并/接入 YOLO26n。
4. **图/引擎接入**：Resize 已注册进 calDNN 图与 calBLAS 引擎表（[CALDNN_INTEGRATION.md](docs/CALDNN_INTEGRATION.md)），主机侧 Plan / bind 通过；calDNN 的图发射路径官方尚未打通，所以上板仍走整程序 calbin 路线。
5. 更完整的优化清单（LLM/框架方向）见 `docs/OPTIMIZATION_CANDIDATES.md`。

---

## 6. 说明

- 本仓库为 2026-11-13 黑客松工作快照；平台侧 / 本地数字均来自真实执行（日志与产物在 `evidence/`），真机数字来源见 §0 ※；未验证项在文档中显式标注。
- 代码采用与 calBLAS 仓库一致的风格；所有原语调用均来自仓库真实 API（`npu/src/lib.rs`、calBLAS 模板），未猜测接口。
- 真机验证后请把 `board_out/`（输出 + 日志）与 `docs/REAL_DEVICE_VERIFICATION.md` §8 记录模板一起归档/提 PR。

## 本地复现与验收更新

Jax 的环境与部署记录见 [JAX_LOCAL_SETUP.md](docs/JAX_LOCAL_SETUP.md)。测试工具修复、43 项回归测试（含 19 项 Resize CPU 边界测试、3 项双核分配测试）及两组 Resize CPU 数据复核结果见 [LOCAL_VALIDATION.md](docs/LOCAL_VALIDATION.md)。Resize 接入 calDNN 图与引擎表的改动、演示输出和测试见 [CALDNN_INTEGRATION.md](docs/CALDNN_INTEGRATION.md)。这些本地结果不等于 NPU 真机验证。
