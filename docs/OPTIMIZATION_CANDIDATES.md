# 可继续优化的算子清单（证据版）

**日期**：2026-11-13 21:25
**方法**：全部结论来自仓库源码/文档的 `文件:行` 证据 + 本机 WSL2 工具链实测；区分「无需真机即可验证（编译/docc/静态）」与「必须真机才能验证（数值/性能）」。**没有编造数字**。
**当前基线**：yolo26（CV）与 qwen3-8b（LLM）的设备遍在现工具链上都能编过（本机实测：`cargo build --manifest-path examples/yolo26/Cargo.toml --lib --target dopamine-unknown-none-cvx` = exit 0；qwen3-8b 同）。

---

## 0. 结论速览（按 ROI 排序）

> **真机验证**：按 [REAL_DEVICE_VERIFICATION.md](REAL_DEVICE_VERIFICATION.md) 操作（开发机打包 → 板卡主机 dort → 对比判定）。

> **更新（2026-11-13 21:45）**：本轮已落地 ① 的**通道分块版**（`resize_bilinear2_tiled!`，C=256 全链路验证通过）与 ② 的**可编译原型**（`conv2d_s2_fast!`，双核 docc calbin + doctor 通过；数值/性能待真机）。行流水因“无板无法数值验证、且需三重展开”暂缓。详见文末 §8。


| # | 优化项 | 类别 | 现状证据 | 预期收益 | 工作量 | 需要真机？ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **resize_bilinear2：大尺寸分块（W/C 分块）+ 行流水复用** | 新算子 | 当前 `C<=128`、单核、每行重复算 3 行插值（`vision.rs:748,761`；生成代码 3 load/行对） | 可处理真实特征图（如 64×64×320）；行内计算约 **-67%**、DDR 读约 -2/3 | 中 | 编译/打包可无板；**性能/数值需板** |
| 2 | **conv2d! FAST 路扩到 stride=2** | 现有 CV 模板 | FAST 条件写死 `S == 1`（`vision.rs:122`）；YOLO26n 有 **6 个 conv3s2**（`examples/yolo26/src/kernels.rs:5,10,15,24,54,56`） | 下采样卷积从“每块每抽头走 DDR⇒L1B”切到“整行进 VB”，是 YOLO/UNet 里 stride-2 层的直接提速 | 低-中 | 性能需板；编译可无板 |
| 3 | **group_norm! 实例化 + CPU 参考 + 上板验证** | 现有 CV 模板 | 模板已写好（`vision.rs:518-600`，注释按 SD UNet 设计）、`cores=2` 内置；但全仓库 **0 实例、0 主机参考**（grep 确认） | SD/UNet 解码器的归一化落地；把“死代码”变成可验证算子 | 低 | 数值需板；编译可无板 |
| 4 | conv2d! 形状扩展：k=5/7、dilation、stride 4、act=gelu、groups | 现有 CV 模板 | 当前只 k∈{1,3}、s∈{1,2}、act∈{silu,none}、CI/CO 16 倍数（`vision.rs:93,97-119`） | SAM/ViT/CLIP 的 patch embed（k=s=14/16）与 5×5/7×7 骨干 | 中 | 需板（性能/数值） |
| 5 | **CV 算子进 calDNN 图/引擎表** | 框架 | `calDNN::Op` 封闭枚举无 CV 变体、`Graph::op` 私有、vision 模板不 emit `desc!`；yolo26 干脆不依赖 caldnn | 路线 A 硬要求（“被图调用”）；SD/SAM 端到端前提 | 大 | 编译/Plan 可无板 |
| 6 | upsample2! 通用化（任意倍率/尺寸）并并入 resize 模板族 | 现有 CV 模板 | 只支持固定 ×2、带边框 NHWC、C≤256（`vision.rs:390-425`） | 预处理/letterbox/任意尺寸上采样；替换主机 CPU 兜底 | 低-中 | 编译可无板 |
| 7 | LLM 线性层 kernel 侧优化（系数铺开 / 次序 / lm_head 块形） | LLM 算子 | 已有大量性能记录：fp8 q_proj 2.1→0.7 ms、24 GB/s（`examples/qwen3_8b/deploy/DEPLOY.md:155-164`）；两核合计 ~40 GB/s（`:141`）；文档明说剩下大头是 docc 的 rhs 下降（`:169-171`） | LLM decode 的主要耗时（线性层约 42%+22%+16%+14%+3%） | 大 | **必须板**（这是数据/带宽问题） |
| 8 | 融合引擎（matmul + pointwise 尾处理） | 框架/图 | README 明确列为“还没写”（`calBLAS/README.md:114`） | 少一次中间张量写读；对 attention/FFN 链有效 | 大 | 性能需板 |
| 9 | 编码器多 token 注意力 / prefill 其余算子 / M-RoPE | LLM/CV 混合 | README “还没写”（`README.md:110-112`） | ViT/VLM/SD 文本编码器的端到端必须；也是 prefill 提速基础 | 大 | 需板 |
| 10 | dwconv3!/maxpool5! 形状扩展（s=2、avgpool） | 现有 CV 模板 | 固定 3×3 s1 / 5×5 s1（`vision.rs:285-288,352-388`） | 更多骨干结构可用 | 中 | 需板 |

> **一句话建议**：如果继续围绕“48h 能交付、真机可跑”的目标，优先 **1 → 2 → 3**；如果要冲路线 B（端到端模型），**5 + 9** 是绕不开的大工程。

---

## 1. 我们自己的算子：resize_bilinear2 还能怎么优化

代码：`calculet-repos/calBLAS/src/kernels/vision.rs`（追加的宏）+ `examples/resize_probe/`。

### 1.1 大尺寸支持（W/C 分块）——最实用的下一步
- 现状：`[W, C]` 整行进 VB，f32 中间 cache 一行 `C*4 <= 512 B` ⇒ **C ≤ 128**（`vision.rs:748,761`）；H/W 是编译期常量。
- 问题：真实 SD/SAM 特征图远大于此（如 latent 64×64×320、SAM 64×64×1280），直接套不上。
- 做法：照 `group_norm!` 的 `Cols::norm_cols/norm_parts`（`vision.rs:32-49`，按 ≤256 KB 分段）对 **C 分块**；对 **W 分块**（每次处理一个 `W_tile`，边界列单独处理），行方向可流式处理。
- 验证：无需真机——可编译 + `docc calbin from-calvx` + `doctor` 验证结构；数值/性能需板。

### 1.2 cores = 2（双核）
- 现状：`#[npu::kernel(numerics="ks1")]`，单核（为此绕开了 docc 的“for 体内带结果 if”限制）。
- 收益：理论 ~2×（像素级并行）。
- 难点：分核后第 0 行/最后一行归属、行 clamp 的边界行需要“常量化的核号分支”（`if core_rank() == 0 { … }` 是语句 if，docc 支持；带结果的 if 不支持）。
- 做法：核心区间按 `halves!(H)` 分；边界两行固定由 0 号/最后一号核处理（语句 if），中间行循环边界用常量/线性表达式。

### 1.3 行流水复用（行内插值 -67%）
- 现状：每个输入行 `iy` 都重新算 `iy-1, iy, iy+1` 三行的行内插值（he/ho），相邻迭代重复 2/3；生成 CalVX 里每行对是 **3 次 load、20 mul、10 add、4 store、7 cvt**。
- 做法：保留 `prev/curr/next` 三组 he/ho cache，循环里只算新的 `next`（每次 1 次 load + 1/3 行内计算），轮换 cache。
- 验证：编译可无板；**收益必须上板测**（DDR 与 VU 谁先到瓶颈要看实际）。

### 1.4 布局与语义扩展
- 支持 vision 现有“NHWC + 四周 2 像素边框”布局（可直接接 YOLO/UNet 特征图缓冲，省一次主机重排）；`BORDER=2` 已在 `vision.rs:22`。
- 任意倍率（SX/SY）或任意 `H_out×W_out`：坐标/权重需要运行时算，注意 docc 的标量控制流限制（参考我们踩过的 for-if 坑）。
- NCHW 版本：可用现成 `nchw_to_nhwc!`/`nhwc_to_nchw!`（但这两个也零实例，第一次用要一起验证）。

---

## 2. 现有 CV 模板：按“缺形状 / 没验证 / 性能”分类

### 2.1 conv2d!（全仓库用最多：72 个实例）
- **缺 FAST 路覆盖**：`FAST = S==1 && KC==1 && (K*WI+16)*CI*2 <= 384KB`（`vision.rs:122`）。YOLO26n 的 6 个 stride-2 卷积全部走慢路；把 `S==2` 纳入快路（每行读 `K` 行的 stride 视图）是低风险高收益改动。
- **缺形状**：k 只有 1/3（`@taps 1/3` 宏分支），s 文档只写 1/2，act 只有 silu/none，通道要求 16 的倍数（`vision.rs:97-119`）。SAM/CLIP 的 patch embed 是 k=s=14/16，ViT 骨干常见 5×5/7×7、stride 4、dilation。
- **已有性能记录**（不是猜的）：`vision.rs:18-19` 记录 npu3 2026-11-09 sd15：64² 320→320 10.3⇒2.4 ms、32² 320→640 5.4⇒1.5 ms（快路 vs 原路），说明模板已被优化过一轮；进一步空间在 FAST 覆盖范围与 TAIL/块交错策略。
- **接入**：主机侧目前还要 im2col（YOLO 的首层，`examples/yolo26/deploy/ref.py:17-32`）；把首层/任意 k 卷积搬到 NPU 可减少 CPU 兜底。

### 2.2 group_norm!（模板有、0 实例）
- 代码：`vision.rs:518-600`；`cores=2` 内置（:526）、两遍均值/方差、按 ≤256 KB 分段（:535）、注释明确按 SD UNet 设计（:522 “UNet 里最大的组 10 × 4096”）。
- 价值：SD UNet 每个 block 都要 GroupNorm；这是 CV 路线落地最缺的一块。
- 下一步：实例化一个（如 32×32、C=128、G=32）+ numpy 参考 + `docc` 打包验证；有板后对逐元素误差/耗时。
- 风险：两遍法 + `rsqrt_approx`（KS1 近似，≤1 ULP）⇒ 与 torch 不会逐位相同，需容差口径（和 `CALCULET_RECON.md` §8.1 的近似表一致）。

### 2.3 upsample2! / dwconv3! / maxpool5! / spatial_attn!
- `upsample2!`（`vision.rs:390-425`）：固定 ×2、带边框、bf16、C≤256、不进图；只被 yolo26 使用两次。通用化见 §1.4。
- `dwconv3!`（`:285-350`）：3×3 s1、C≤128；`maxpool5!`（`:352-388`）：5×5 s1、C≤256。扩展 s=2/avgpool 成本低。
- `spatial_attn!`（`:654-741`）：YOLO PSA 专用（每头 kd/hd、整图当 token、补到 128 倍数）；泛化到 SD 的 attention 需要另做（SD 有现成 `attention!` 模板，见 README DiT 行）。
- 零实例的辅助模板：`nchw_to_nhwc!`( :434)、`nhwc_to_nchw!`( :479)、`transpose2d!`( :605)——写好了但没人用 ⇒ 第一次接入时要一起验证。

### 2.4 用量数据（决定优化优先级）
全 examples 统计（`grep -rho "calblas::<模板>!" examples/*/src/*.rs`）：
`linear! 107`、`conv2d! 72`、`binary! 43`、`rms_norm! 32`、`rope! 24`、`all_gather! 22`、`copy_i32! 19`、`embedding! 12`、`scatter_kv! 11`、`flash_decode! 9`、`dwconv3! 8`、`moe_* 7/6…`
⇒ **conv2d/linear 的改动影响面最大**。

---

## 3. 框架级优化（不改算子数学，但决定“能不能用/快不快”）

| 项 | 证据 | 收益 | 依赖 |
| --- | --- | --- | --- |
| CV 算子进 calDNN 图/引擎 | `calDNN/src/graph.rs:405-451` 封闭 `Op`；`Graph::op` 私有；vision 模板无 `desc!`；yolo26 不依赖 caldnn | 路线 A 的注册/被图调用；端到端前提 | 跨仓库改动；可无板验证编译/Plan |
| 融合引擎（matmul+pointwise） | `calBLAS/README.md:114` “还没写” | 少一次中间张量写读 | 需板测收益 |
| docc 限制清理 | `README.md:102-108` 五类绕法；`STATUS.md:43-56` 列出 25 个 W4 kernel 因 ui4 视图编不出 ELF（旧口径，**需在当前 CalVX 工具链上复核**） | 去掉垫片/展开/对齐绕法，代码更简、更易优化 | docc 侧；可无板验证 |
| 逐算子计时口径 | `README.md:95-100`（profbuf `0x091c0000`、`dort mem read`）、`AGENTS.md:128-132`（DDR ~84 GB/s 下界） | 优化前后可比；避免“拿单芯粒比厂商两芯粒” | **必须真机** |

---

## 4. LLM 方向（如果队伍想转 LLM / 保底拿分）

- **线性层是绝对大头**：qwen3-8b 单步耗时分布 gate/up 42%、down 22%、lm_head 16%、q/o 14%、k/v 3%（`examples/qwen3_8b/deploy/DEPLOY.md:134`）。
- 已有优化历史（直接可参考，不要重复踩）：q_proj 2.1→0.7 ms、W8A8 n 256 一块 24 GB/s（`:155-164`）；整模型两芯粒 52.1 ms/token（`:198-204`）。
- 文档点名的剩余瓶颈：**docc 的 rhs 下降路径**（DDR⇒L1B⇒VB(fp8⇒bf16)⇒L1B⇒8 次 hwptn⇒GIB1 全串行，`:169-171`）以及“每个 kb 把系数按列铺开（gate/up 每 kb 约 290 条 VU）”——后者是 kernel 侧可动的点。
- 其他：融合引擎（`README.md:114`）、prefill 多 token（`:112`）、动态 KV（`:113`）。

---

## 5. ⚠️ 看似能优化、其实被约束禁掉的（别白做）

| 想法 | 为什么不行（证据） |
| --- | --- |
| **单遍 online softmax 代替两遍 flash** | `attention.rs:154` 明确 ⛔：“在线 softmax：舍入次序不同，上下文过 16 后整模型与基线分叉，docc 在 10.109 上见过”；`README.md:39` 也是“两遍（⛔ 在线 softmax）” |
| 把 `Notes::BF16_REDUCTION` 的 10 个算子改成 f32 累加 | 这些是**刻意对齐厂商**的数值口径（`engines.rs` 的 Notes），改了会破坏逐位对照；除非目标是新精度口径 |
| 把模拟器（`cvx run`）当优化验收 | `AGENTS.md:77-87`：KS1 的 `rsqrt`/`exp` 是查表近似，模拟器算精确值，**不能当真机参照** |
| 用脚本拼 MLIR 做整步 | `AGENTS.md:15-27` 明确 ⛔（绕过类型检查，已踩四种坑） |

---

## 6. 建议的下一步（Top 3）

1. **resize_bilinear2：大尺寸分块 + 行流水（+ 尝试双核）** —— 我们最熟、已有编译/打包/CPU 参考全套；无需板即可验证结构，真机到了直接测数值/性能。
2. **conv2d! FAST 路扩到 stride=2** —— 改一个常量条件 + 行视图，YOLO26n 有 6 个直接受益实例；风险低、影响面大（conv2d 是第二大用量模板）。
3. **group_norm! 实例化 + CPU 参考** —— 模板已按 UNet 写好但从未跑过；这是 SD 路线落地最缺的一块，工作量最小。

> 如果目标是“路线 A 提交”：先 2/3 做出可验证算子，再补 **5（图接入）**；
> 如果目标是“性能分数”：必须先把板子拿到，再按 §3 的计时口径做前后对比，优先线性层（LLM）或 conv2d s=2（CV）。

---

## 7. 证据索引（本次扫描用到的关键行）

- `calBLAS/README.md:102-114`（docc 绕法清单 + “还没写”三项）
- `calBLAS/AGENTS.md:128-132`（DDR 84 GB/s 下界、性能口径）
- `calBLAS/STATUS.md:43-56`（docc 编不出 ELF 的原因表；旧口径需复核）
- `calBLAS/src/kernels/vision.rs:17-19,32-49,93-122,285-288,352-365,390-425,434-516,518-600,605-652,654-741,748-761`
- `calBLAS/src/engines.rs:45-131`（Notes：10 个 `BF16_REDUCTION`、6 个 `TWO_PASS_SOFTMAX`、2 个 `CROSS_CHIP`）
- `calBLAS/examples/yolo26/src/kernels.rs:5,10,15,24,54,56`（6 个 stride-2 卷积）；`deploy/DEPLOY.md:7`（~40 ms 未优化）
- `calBLAS/examples/qwen3_8b/deploy/DEPLOY.md:134,141,155-171,198-204`（线性层耗时/优化历史）
- `calBLAS/src/kernels/attention.rs:154`（⛔ 在线 softmax）
- 本机实测：yolo26 / qwen3-8b 设备遍在现工具链编译通过（exit 0）。

---

## 8. 本轮落地结果（2026-11-13 21:45，✅ 本机实测）

### 8.1 ① resize 通道分块版：`resize_bilinear2_tiled!`

- 代码：`calculet-repos/calBLAS/src/kernels/vision.rs`（追加宏）+ `examples/resize_probe_big/`（32×32×256 → 64×64×256，`cc=64`，4 块）
- 解决：原版 f32 中间 cache 一行 ≤512 B ⇒ `C<=128`；分块版按 `CC` 切通道，`C=256` 已跑通，约束为
  `CC % 32 == 0`、`C % CC == 0`、`W*CC*56 B <= 384 KiB`
- 验证（WSL2 + calculet 工具链）：

| 项 | 结果 |
| --- | --- |
| 设备遍编译 | ✅ `resize_probe_big.cvx` 40,369 B / `.cvxo` 7,118 B |
| `docc calbin from-calvx` | ✅ exit 0（ccu0 86,568 B / ccu1 6,872 B） |
| `docc calbin doctor` | ✅ 拒收 0 条（2 ping/pong 警告） |
| CCU 反汇编 | ✅ 15,997 条：4,177 宽 / 10 custom / 11,810 标量，0 unknown；`load_size 81,612` = 各节之和 |
| 构建确定性 | ✅ 两次 calbin 逐字节相同（`docc calbin diff` 一致） |
| CPU 参考 | ✅ 32×32×256：与独立 float64 参照 **bf16 舍入 100% 一致**（max 1.953e-3，≤0.996 ULP） |
| 真机数值 | ❌ 未验证（无板/无模拟器） |

### 8.2 ② stride-2 FAST 卷积原型：`conv2d_s2_fast!`（实验，未进共享模板）

- 问题：`conv2d!` 的 FAST 路写死 `S == 1`（`vision.rs:122`），YOLO26n 的 6 个 conv3s2 只能走慢路。
- 做法：每个输出行把 3 个输入行各拆成“偶像素 / 奇像素”两份 VB cache（从 DDR 按像素行距 `2*CI` 直接加载）；
  tap `kx=0/1/2` 分别取 `even[ky]` 行 `mb*16` / `odd[ky]` 行 `mb*16` / `even[ky]` 行 `mb*16+1`——**都是连续 16 行**，stride-2 也能用 FAST 路。
- 代码：`vision.rs` 里追加的实验宏 + `examples/conv2d_s2_probe/`（YOLO 同形状 80×80×64→40×40×64，k3s2 silu，双核）
- 验证（仅平台侧；⚠ 原型曾有一处 padding 奇偶映射错误，已在写档前修正并重建）：

| 项 | 结果 |
| --- | --- |
| 设备遍编译 | ✅ `.cvx` 14,496 B / `.cvxo` 2,608 B |
| `docc calbin from-calvx` | ✅ exit 0；权重块 73,984 B；**ccu0/ccu1 各 22,168 B**（双核各编一份）；另有同形状慢路对照 `conv2d_s2_ref`（ELF 各 16,152 B） |
| `docc calbin doctor` | ✅ 拒收 0 条（2 ping/pong 警告） |
| CCU 反汇编 | ✅ 2,699 条：903 宽 / 10 custom / 1,786 标量，0 unknown；`load_size 17,216` = 各节之和 |
| 构建确定性 | ✅ 两次 calbin 逐字节相同 |
| 数值 / 性能 | ❌ **未验证**：无板、无 `cvx` 模拟器；TU 累加次序与 numpy 不可能逐位 ⇒ 需要真机容差口径 |

### 8.3 ③ 行流水（原计划的下一项）——暂缓，理由

- 现状：每个输入行重复计算 `iy-1/iy/iy+1` 三行的行内插值（生成代码每行对 3 load、20 mul、10 add）。
- 若做流水，需要三套 he/ho cache 轮换；要么每次迭代 2 次 `vec.copy` 轮换（收益可能被拷贝抵消），
  要么把循环体三重展开（代码 3×、边界/最后一行逻辑复杂）。
- **无板就无法数值验证**，结构性改动一旦有静默错误无法发现；因此本轮不改已交付算子，等真机到位后用
  `dort` 对逐元素 + 单算子耗时做 A/B 再决定是否落地（计划对齐 `AGENTS.md:77-87` 的“模拟器/真机”口径）。

### 8.4 产物路径

- 分块 resize：`evidence/resize_big/`（cvx/cvxo、calbin、反汇编、doctor、确定性、CPU 参考与 bin）
- s2 conv 原型：`evidence/conv_s2_fast/`、`evidence/conv_s2_ref/`（cvx/cvxo、calbin、反汇编、doctor、确定性）
- 统一补丁：`patches/onboard_kit.patch`（已重新生成，含后续改动）（763 行，17 个文件；sha256 `882fd866…`）
- 工作区仓库：`calculet-repos/calBLAS`（`vision.rs` 追加 3 个宏；`examples/resize_probe{,_big}`、`examples/conv2d_s2_probe/`）

