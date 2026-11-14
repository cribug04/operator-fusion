# Stable Diffusion 1.5 · VAE 解码器全 NPU 移植（比赛期间实现）

**日期**：2026-11-14
**目标**：把 SD1.5 的 VAE decoder（latent `[4,64,64]` → 512×512 RGB 图像）完整放到 Calculet NPU 上跑，作为赛题路线 B（端到端模型移植）的第一步。
**当前状态**：✅ 平台侧全部打通（2026-11-14 晚按审查修复两处生成器错误后重新通过，见 §5.1 注）（设备遍编译 / docc calbin / doctor 拒收 0 / 本地黄金仿真与 diffusers fp32 参考 PSNR ≈ 52.8 dB）；队友已报告真机通过；原始输出尚待入库，数字与证据边界见 [README](../README.md) §0。

---

## 1. 结论（可复现的事实）

| 项 | 结果 |
| --- | --- |
| 程序划分 | 5 段 CalVX 整步程序（`a → b1 → b2a → b2b → c`），一段一个 `.cvx` / 一个 calbin |
| 设备遍 | 5/5 `cargo build --lib --features step --target dopamine-unknown-none-cvx` exit 0 |
| docc | 5/5 `docc calbin from-calvx` exit 0；5/5 `docc calbin doctor` **拒收 0 条**（各 2 条 ping/pong 同址警告，单次推理无影响） |
| 权重 | 从 SD1.5 **fp32** safetensors 直接重排为 bf16 参数块，5 段共 **100.7 MB**；打包器自校验 138 个张量逐字节回读一致（bad=0） |
| 本地数值 | 用 torch 按 `step.rs` 的算子顺序 + 我们的布局**仿真同一份程序**，与 diffusers fp32 参考：整图 **PSNR 52.83 dB**、max\|Δ\| 0.0473、mean\|Δ\| 0.00336；图像统计与参考几乎一致（见 §5） |
| 真机数值 | ❌ 未验证（无板卡登录；上板包已备好，见 §7） |

> “平台侧全通”不等于“算对”：所以另做了一套**黄金仿真器**——它不重写数学，而是按我们生成的 `plan.json` 逐步执行（同样的 border/NHWC 布局、同样的 bf16 舍入点、同样的权重块），把“打包 + 布局 + 层序 + 注意力 + 归一化”全部验到和 torch 参考一致，剩下的不确定只剩“硬件 kernel 本身的行为”。

---

## 2. 交付物

> 路径说明：下表的“工作区”指比赛用工作目录（队友本机工作目录）；**在本 Gitee 仓库里，`_recon/work/sd_vae/xxx` 对应 `sd_vae/xxx`**（`gen/`、`kernels/`、`evidence/`、`onboard/`、`SMOKE.md`、`STAGE_EVIDENCE.md`），`docs/` 下是本文与真机手册。

| 路径（工作区） | 内容 |
| --- | --- |
| `calculet-repos/calBLAS/examples/sd_vae_{a,b1,b2a,b2b,c}/` | 5 个 example crate：`src/kernels.rs`（kernel 特化）、`src/step.rs`（整步程序）、Cargo.toml |
| `_recon/work/sd_vae/gen/gen.py` | **层表 → Rust 代码生成器**（段划分、缓冲分配、参数块布局、kernel 去重） |
| `_recon/work/sd_vae/gen/pack.py` | **权重打包器**（safetensors fp32 → bf16 参数块；含自校验） |
| `_recon/work/sd_vae/gen/emulate.py` | **黄金仿真器**（torch 执行我们的程序语义） |
| `_recon/work/sd_vae/gen/compare_stages.py` | 逐段对 torch dump 的数值比对 |
| `_recon/work/sd_vae/gen/host.py` | 主机侧：latent → `in_x.bin`；`out_y.bin` → 图像/PNG；图像比对 |
| `SD_VAE_REAL_DEVICE_VERIFICATION.md` | **真机验证手册**（逐条命令、字节数表、判据、排错表、记录模板） |
| `sd_vae/onboard/` | 上板包脚本与校验：`package_all.sh`、`run_on_board.sh`、`MANIFEST.txt`、`SHA256SUMS.txt`、`sd_vae_onboard_pkg.tar.gz.sha256`（tar 约 80 MB，解出约 108 MB） |
| `_recon/work/sd_vae/SMOKE.md` | 模板/语言实测笔记：8 个模板确切写法、`docc` 两个 bug 的完整记录与绕法、`attention!` 头维上限实证 |
| `_recon/work/sd_vae/ref/` | torch 黄金参考（sd-ref 产出）：`REFERENCE.md`、`dumps_index.json`、`layer_table.json`、`latent.bin`、`reference_image.{bin,npy,png}`（**仓库里：`sd_vae/evidence/ref_image.npy`（fp32 [1,3,512,512]，3,145,856 B，sha256 `b7cc6839…`）+ `latent.bin` + PNG**） |
| `_recon/work/sd_vae/emu_out/` | 仿真结果：`emulated_image.{npy,png}`、`compare.json`、`stage_compare.json` |

生成的 `.cvx` 与 calbin 不入库（可由源码 + 生成器一键重建，见 §9）；上板包在本地，需要时拷到板上。

---

## 3. 结构：VAE decoder 层表与 5 段划分

SD1.5 VAE（diffusers `AutoencoderKL`，config：`block_out_channels=[128,256,512,512]`、`layers_per_block=2`、`norm_num_groups=32`、`latent_channels=4`）解码器 = `conv_in` + 2 个 mid resnet（中间一个单头注意力）+ 4 个 up_block（各 3 个 resnet，前三个带 nearest×2+conv 上采样）+ `conv_norm_out` + `conv_out`。

**层表实测**（来自参考侧逐层 hook，`layer_table.json`）：decoder 共 **35 个 conv + 30 个 GroupNorm + 1 个注意力 + 3 个上采样**；`post_quant_conv`（4→4，1×1）已**数学等价地折进 `conv_in`**（`W_eff = W_in · W_pq`，`b_eff = b_in + W_in · b_pq`）。

| 段 | 输入 | 包含层 | 输出 | 参数 | @act |
| --- | --- | --- | --- | --- | --- |
| **a** | latent `[4,64,64]`（border bf16 `[68][68][16]`，147,968 B） | conv_in（4→512，k3，post_quant 已折入）、mid resnet0、**单头注意力**、mid resnet1 | `[68][68][512]` bf16（4,734,976 B） | 21.48 MB | 37.6 MiB |
| **b1** | 64²×512 border | up_blocks.0 的 3 个 resnet + nearest×2 + conv（→128²） | `[132][132][512]`（17,842,176 B） | 33.45 MB | 34.6 MiB |
| **b2a** | 128²×512 border | up_blocks.1 的 3 个 resnet | `[132][132][512]` | 28.73 MB | 67.1 MiB |
| **b2b** | 128²×512 border | nearest×2 + conv 512→512（→256²） | `[260][260][512]`（69,222,400 B） | 4.72 MB | 66.0 MiB |
| **c** | 256²×512 border | up_blocks.2（512→256，含 1×1 shortcut）、up_blocks.3（256→128）、conv_norm_out、conv_out（128→3，补到 16） | 图像 `[516][516][16]` bf16（8,536,064 B） | 12.29 MB | 748.2 MiB |

> 为什么分 5 段而不是一整份：**docc 0.4.1 的 riscvc 链接器 21 位分支限制**。整份程序的单个 CCU `.text` 超过 1 MiB 时链接报 `distance … 超出 21 位`。分段后单段 ELF 0.03–1.53 MB，全部可链接。另外分段也天然便于逐段对数值、逐段上板定位。

---

## 4. 实现要点

### 4.1 数据布局
- 特征图沿用 calBLAS 视觉模板的 **NHWC + 四周 2 像素边框**布局：`[H+4][W+4][P]`，`P` = 通道行距（我们取 = 实际通道数，都是 16 的倍数）。边框由程序开头一次性清零（`calblas::fill16k!`，运行期块数，不会展开）。
- 卷积用 `calblas::conv2d!`（k=1/3、s=1、BN 已折入、bias f32、act=none；resnet 的第二个 conv 直接带 `res=` 做残差）。
- GroupNorm 需要 NCHW：每个 norm 前 `nhwc_to_nchw!`，norm 后 `nchw_to_nhwc!`（模板自带，块处理，两核分）。NCHW 暂存在 `@act` 里做**共享 scratch**（顺序复用，按最大 GroupNorm 输入分配，c 段 128 MiB = 134,217,728 B）。
- 上采样：`calblas::upsample2!`（nearest ×2，border 布局）。模板 `c ≤ 256`，512 通道时**拆两次调用**（指针 +512 B、通道窗口 256）。
- 注意力：q/k/v/o 紧排 `[4096, 512]` bf16；`border → tight` 用自写 `k_crop`、反向用 `k_pad`；K 用 `transpose2d!` 转成 `[512,4096]` 供 mma 右乘。

### 4.2 三个手写 kernel（都在 example crate 里，不改 calBLAS 源码）

1. **`gn_stats` + `gn_apply`**（GroupNorm 的统计与应用，取代 `calblas::group_norm!`）
   - 原因：模板的 `for part in 0..NH`（NH 编译期常量）会被 docc 整段展开，单函数 `.text ≈ 545 B × R`，`R = (C/G)·HW/CC`；128²×512 时 R=2048、`.text ≈ 1.12 MB` > RISC-V 21 位分支 ±1,048,575 B ⇒ 链接失败（实测 hw=4096→282,484 B、8192→560,716、12288→831,812、16384→1,116,476 B）。把 NH 做成普通标量实参也无效（docc 内联 + 常量传播又折回常量）；只有从 SRAM 读才不被展开（sd-kernels 另有实证）。
   - 我们的做法：`gn_stats` 按**组**统计（一组 = CG 通道 × HW 像素，均值/方差全组共享），`gn_apply` 归一 + 仿射 + silu；段内循环上界全部走运行期标量 ⇒ 代码恒定、不再触发该 bug；同时把 group_norm 模板要求的「按行展开的 f32 [rows][8]」参数（673 MB 级）改成 **每通道 16 份 f32（[C][16]，一行 64 B）**，参数块从 313 MB 降到 **96 MB**。
2. **`sd_vae_attn`**（单头 flash 注意力：T=4096、D=512、无 mask、scale=1/√512）
   - 两遍：第一遍分数取行最大；第二遍先算 `p=exp(s−m)` **全存进 VB（`pall[512,128]` bf16 = 128 KB）**并累计行和，再单独一段用 4 个 D 块累加器按固定次序 `Σ p·V`。
   - 为什么这样拆：docc 的 `calt-fate` 拒绝「分数累加链 + 4 个输出累加器链同时在飞」（GOB FIFO 累加串次序），拆到不同循环即可；这与 `conv2d!` FAST 路四累加器同构。
   - **注意力有残差**（diffusers `Attention` 的 `residual_connection`）：`out = proj(attn) + x`，我们让 proj 的 1×1 conv 直接带 `res=x` 实现。
3. **`k_crop` / `k_pad`**（border ⇄ tight 的搬运，单核；一次搬「一行图像 × 256 通道」）

### 4.3 权重重排（packer）
- bf16 用 RNE（就近舍入到偶数）从 fp32 转；conv 权重布局 `[k·k][CI补16][CO补16]`（`flat[(ky·K+kx)·CI_pad·CO_pad + ci·CO_pad + co]`）；bias f32 `[CO_pad]`。
- `conv_in` 融合：`W_eff[co,ci,ky,kx] = Σ_j W_in[co,j,ky,kx]·W_pq[j,ci]`（fp64 计算后转 fp32），`b_eff = b_in + Σ W_in·b_pq`。
- attention 的 Linear 权重按 1×1 conv 打包（`w[0][ci][co] = W[co][ci]`），bias 原样。
- GroupNorm gamma/beta：`[C][16]`（16 份重复，供 kernel 行读 + reduce Max 取标量）。
- 打包器写完参数块后**从块里读回每个张量**与期望字节逐字节比对：138 个张量全一致（`bad=0`），参数块 sha256 见各段 `pack_report.json`。

### 4.4 程序约定
- `#[npu::program(chiplets = 1, cores = 2)]`；每个 `cores = 2` kernel 调用后 `sync_cores()`；`fill16k!` / `k_crop` / `k_pad` 这类不分块的单核 kernel 包在 `if core_rank() == 0 {}` 里再同步；`nhwc_to_nchw!` / `nchw_to_nhwc!` / `upsample2!` / `transpose2d!` 内部用 `halves!()` 按 `num_cores()` 分块（docc 把它折成程序的核数 2），**必须两核都调用**，与官方 yolo26 程序一致（2026-11-14 审查修正）。
- 所有 `@act` 偏移、参数块偏移由生成器统一分配（4 KiB 对齐），程序与打包器共用同一份 `params.json`/`plan.json`。
- **段间靠裸字节衔接**（`out_a.bin` 直接喂下一段 `in_x`），而 conv 只写内部区域 ⇒ 每段程序开头把 `out_y` 声明的边框也清零（`fill16k`），否则会把上一段内存残留当 padding 读进去。

---

## 5. 验证证据

### 5.1 平台侧（本机实测）

> **2026-11-14 晚修复版**：审查（`docs/REVIEW_SD_VAE_2026-11-14.md`）发现生成器两处错误并已修复：① `nhwc_to_nchw!` / `nchw_to_nhwc!` / `upsample2!` / `transpose2d!` 内部按 `num_cores()` 分块，原来只由 0 号核调用会漏掉一半数据块，现改为两核都调用；② NCHW 暂存区大小算成 0 字节，现按最大 GroupNorm 输入分配（c 段 128 MiB）。修复后 5 段重新编译、`docc calbin from-calvx`、`doctor` 全部通过（拒收 0，各 2 条 ping/pong 警告），数字见下表；修复前的数字保留在 `sd_vae/STAGE_EVIDENCE.md` §1。本地黄金仿真不建模地址与核分工，修复前后输出相同；**真机仍未验证**。

| 段 | `.cvx` B | `.cvxo` B | 参数块 B | @act B | ccu0 ELF B | ccu1 ELF B | doctor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a | 243,366 | 38,610 | 21,483,520 | 39,436,288 | 1,414,672 | 1,411,672 | 拒收 0 |
| b1 | 156,915 | 25,127 | 33,452,032 | 36,306,944 | 968,536 | 966,160 | 拒收 0 |
| b2a | 68,253 | 11,324 | 28,729,344 | 70,369,280 | 398,884 | 397,144 | 拒收 0 |
| b2b | 22,461 | 3,888 | 4,722,688 | 69,222,400 | 31,160 | 30,700 | 拒收 0 |
| c | 270,253 | 43,162 | 12,292,096 | 784,547,840 | 1,528,008 | 1,525,764 | 拒收 0 |
| 合计 | **761,248** | **122,111** | **100,679,680** | — | **4,341,260** | **4,331,440** | — |

（doctor 的 2 条警告都是 `in_x`/`out_y` 的 ping/pong 同址；单次推理不触发，重复喂输入时需要加 `double_buffer`。）

补充证据（以下三项在 **修复前** 的构建上测得，修复版未重跑；完整表见 `sd_vae/STAGE_EVIDENCE.md`）：
- **确定性**：删 `target` 重编两次，5 段 `.cvx` sha256 全部逐字节相同；`docc calbin from-calvx` 跑两次，5 段两组 CCU ELF sha256 逐字节相同。
- **反汇编**：5 段 × 2 核共 12 份 CCU ELF，`docc elf ccu disasm` 全部“走完了：每个字节都被一条认得出的指令覆盖”，**unknown 0 条**（合计 780,588 条 ccu0 / 496,845 条 ccu1）。
- **装载量**：`docc elf ccu view` 的 `load_size` 与“各节字节和”自校验一致（a 1,408,896 / b1 963,140 / b2a 393,656 / b2b 26,120 / c 1,521,536 B，ccu0）。

### 5.2 本地黄金仿真 vs diffusers fp32 参考

参考侧（sd-ref）：torch 2.15.0+cpu、diffusers 0.43.0，权重 fp32（sha256 `a2b5134f…`），latent = `default_rng(1234).standard_normal((1,4,64,64))/0.18215`（sha256 `79ed3410…`），确定性两次 decode 逐 bit 相同。
比对用的参考张量就是仓库里的 **`sd_vae/evidence/ref_image.npy`**（`[1,3,512,512]` f32，3,145,856 B；npy 文件 sha256 `b7cc6839afa2b5a6d0d16b73bc061216d5cc4fcab032e39f3a3f7e8442bce126`，其 data 段原始字节 sha256 `5dcfd42c0a36f4727ddaff2ef14c5b01c4cbcf6e36ed852acb9648858e03475b`，与 `REFERENCE.md` 记录的 `reference_image.bin` 一致；`reference_image.png` 只是预览）。板卡回传 `out_c.bin` 后用 `host.py compare image.npy ref_image.npy` 即可得到 PSNR。

| 边界 | module | 与参考 PSNR | max\|Δ\| | 参考范围 |
| --- | --- | --- | --- | --- |
| 段 a 输出 | `decoder.mid_block.resnets.1` | 43.91 dB | 0.212 | [-19.08, 18.86] |
| 段 b1 输出 | `decoder.up_blocks.0` | 35.62 dB | 0.531 | [-79.01, 64.03] |
| 段 b2a 输出 | `decoder.up_blocks.1.resnets.2` | 34.42 dB | 0.808 | [-97.68, 73.01] |
| 段 b2b 输出 | `decoder.up_blocks.1`（含 upsampler） | 26.27 dB | 1.285 | [-232.8, 214.6] |
| 段 c 输出（最终图） | `decoder.conv_out` | **52.83 dB** | **0.0473** | [-1.867, 1.653] |

最终图统计对比（参考 vs 仿真）：min -1.867134 / -1.8672、max 1.653239 / 1.6484、mean -0.089909 / -0.0898、std 0.738124 / 0.7379；**mean|Δ| = 0.00336**，PNG 肉眼看不出区别（`emu_out/emulated_image.png` vs `ref/reference_image.png`）。

中间层 PSNR 低是因为这些张量数值范围很大（±80/±233），误差来自 bf16 存储；到最终图和参考仍 52.8 dB，说明布局/打包/层序/注意力全部正确。

### 5.3 真机（待办）
见 §7；判据：5 段 rc=0；最终图 vs 参考 **PSNR ≥ 40 dB 且 max|Δ| ≤ 0.08**（仿真给的是 52.8 dB/0.047，留余量给硬件近似指令）；同输入重复跑两次数值一致（确定性）。

---

## 6. 复现（本机）

```bash
# 0) 前提：WSL 里 /root/{calBLAS,calDNN,npu} 与 calculet 工具链；SD1.5 fp32 VAE 权重在 /root/sd_vae/model/vae/
cd calculet-repos/calBLAS/examples
# 1) 生成 5 段源码（工作区 gen.py 也可直接跑）
for s in a b1 b2a b2b c; do
  python3 _recon/work/sd_vae/gen/gen.py /root/sd_vae/build/$s --stage $s
  cp /root/sd_vae/build/$s/{kernels.rs,step.rs} /root/calBLAS/examples/sd_vae_$s/src/
done
# 2) 设备遍
for s in a b1 b2a b2b c; do
  ( cd /root/calBLAS/examples/sd_vae_$s && cargo build --lib --features step --target dopamine-unknown-none-cvx )
done
# 3) 打包权重 + docc
for s in a b1 b2a b2b c; do
  python3 _recon/work/sd_vae/gen/pack.py /root/sd_vae/build/$s/params.json \
    /root/sd_vae/model/vae/diffusion_pytorch_model.safetensors /root/sd_vae/pack_$s
  # pkg_$s = step.cvx + param_blk0.bin ⇒ docc calbin from-calvx ⇒ doctor
done
# 4) 黄金仿真 + 逐段比对 + 最终图
python3 _recon/work/sd_vae/gen/emulate.py /root/sd_vae/build '/root/sd_vae/pack_{stage}' \
  _recon/work/sd_vae/ref/latent.bin _recon/work/sd_vae/ref/reference_image.bin /root/sd_vae/emu_out
python3 _recon/work/sd_vae/gen/compare_stages.py /root/sd_vae/emu_out /root/sd_vae/ref
# 5) 打上板包
bash sd_vae/onboard/package_all.sh   # ⇒ /root/sd_vae/onboard_pkg + /root/sd_vae/sd_vae_onboard_pkg.tar.gz
```

---

## 7. 真机验证（怎么操作）

### 7.1 包内容（`sd_vae_onboard_pkg.tar.gz`，约 80 MB，解出约 108 MB；tar sha256 = `cbfe5f7dd597561b52bb2ea07cd6587e347e468d153cfd90c734aedd7d1d4cda`，也可直接看 `sd_vae/onboard/sd_vae_onboard_pkg.tar.gz.sha256`）
```
cal_a/ cal_b1/ cal_b2a/ cal_b2b/ cal_c/   # docc calbin 目录（含 param_blk0.bin 与 pkg_*/ 下 CCU ELF）
in_x.bin        # stage a 的输入：latent ⇒ border bf16 [68][68][16] + 补零到 decl 163,840 B（sha256 cf3eaee6…）
latent.bin      # 原始 z，留档（sha256 79ed3410…）
ref_image.npy   # torch 参考图 [3,512,512] f32（sha256 b7cc6839…）
ref_image.png   # 参考图（人工看）
host.py         # input / image / stage / raw / compare 工具（sha256 0afe3eca…）
run_on_board.sh # 一键串 5 段 + 出图 + 对比 + 计时
SD_VAE_REAL_DEVICE_VERIFICATION.md   # 真机手册
MANIFEST.txt SHA256SUMS.txt
```

### 7.2 上板步骤
```bash
mkdir -p onboard_pkg && tar xzf sd_vae_onboard_pkg.tar.gz -C onboard_pkg && cd onboard_pkg
# 板子健康（AGENTS.md 流程）：OS up / RT up
dort device -i 0 | grep "Dev  :"
# 一条龙（脚本逐段串起来；每段 rc≠0 会停）
./run_on_board.sh . 0
# 中间段自检（可选）：python3 host.py stage out_a.bin 64 512  … 见真机手册 §5
```
脚本内部就是 5 条（`--in-tensor in_x --out-tensor out_y`，raw 读写):
```
dort calbin run -i 0 -f cal_a   -m pkg_a   --raw-in in_x.bin    --raw-out out_a.bin
dort calbin run -i 0 -f cal_b1  -m pkg_b1  --raw-in out_a.bin   --raw-out out_b1.bin
dort calbin run -i 0 -f cal_b2a -m pkg_b2a --raw-in out_b1.bin  --raw-out out_b2a.bin
dort calbin run -i 0 -f cal_b2b -m pkg_b2b --raw-in out_b2a.bin --raw-out out_b2b.bin
dort calbin run -i 0 -f cal_c   -m pkg_c   --raw-in out_b2b.bin --raw-out out_c.bin
python3 host.py image out_c.bin image.npy image.png
python3 host.py compare image.npy ref_image.npy compare.json
```
产物：`image.png`（512×512）、`image.npy`、`compare.json`、`timing.txt`（每段耗时）。

### 7.3 判据
| 项 | 通过标准 |
| --- | --- |
| 每段 | `rc=0`；板上 `Dev: OS up RT up` |
| 数值 | `compare.json`: **PSNR ≥ 40 dB 且 max_abs ≤ 0.08**（仿真基准 52.83 dB / 0.047） |
| 确定性 | 同输入连续跑两次，`out_c.bin` 逐字节相同（或至少图像 max\|Δ\|=0） |
| 图像 | `image.png` 与 `ref_image.png` 视觉一致（随机 latent 的抽象图） |

### 7.4 常见问题
- **`dort calbin run` 报 configure 失败/NoDevice**：板子 RT 掉了，按 AGENTS.md：`dort device reprobe -i 0 --yes` → 等 15 s → `sudo rmmod calculet_pci && sudo insmod ~/cal-pcie-driver/calculet_pci.ko` → 等 `RT up`。
- **某段挂死/越界**：先只跑那一段（用上一段的 `out_*.bin` 当输入），若单段可跑则问题在段间数据（我们段间都是同一 border 布局，理论上只拷字节）；再查 `timing.txt`。
- **数值偏大**：先确认喂的是 `in_x.bin`（不是 latent.bin）；`in_x.bin` 是 border 布局，不能直接拿 `latent.bin` 喂。
- **重复推理**：当前 decl 没开 `double_buffer`（doctor 的 ping/pong 警告指这个），一次进程跑一张图没问题；连续多张需要把 `in_x`/`out_y` 的 decl 加 `double_buffer` 重新 docc。

---

## 8. 还没做 / 风险

- **真机数值未验**（P0，等板卡访问）。平台侧 + 仿真侧都已过，风险主要在：硬件 `exp/rsqrt` 近似（KS1 标定 ≤1 ULP）、两核同步、以及大 buffer 在真实 DDR 的放置。
- **性能未知**：段 c 的 @act 650 MiB、最后一段 conv 在 512² 上很多；板上一张图预计在秒级，待实测（上板包会打 `timing.txt`）。
- **bf16 vs fp16**：原模型 fp16；我们用 bf16 存激活，仿真看最终 PSNR 52.8 dB，视觉无损；如果板上误差偏大，可把关键层改 f32（模板 conv 只吃 bf16，需要另写）。
- **还不是完整 SD**：CLIP 文本塔、UNet（20 步采样 + CFG）、调度器、ComfyUI 前端都没做——这是路线 B 的后续里程碑（VAE 是其中算子覆盖面最广的一块，已打样）。

---

## 9. 与路线的对应

- 赛题路线 B 要求「补齐目标模型缺失的算子并完成推理前端桥接，在 NPU 上跑通完整模型推理」。这份工作把 **SD1.5 VAE decode 这一段完整模型**跑通了设备遍/docc/本地数值，等于把 VAE 子模型移植完成；下一步是 UNet（复用同一套生成器/打包器/仿真器）与 CLIP/ComfyUI。
- 同时产出的两个通用件：**层表→rust 生成器**、**黄金仿真器**——后续接任何静态图模型都可复用。
