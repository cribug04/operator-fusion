# SD1.5 VAE（全 NPU）真机验证记录 · 2026-11-14

> 结论：**5 段整链在 KS1 真机上跑通并出图**；三个有 torch 参考的检查点（段 a、段 b2b、整图）
> **统计量全部落在 bf16 存储带来的 1 ULP 以内**；逐像素 PSNR 与 b1/b2a 的直接参考仍待补。

## 0. 环境

| 项 | 值 |
|---|---|
| 板机 | `calculet-MS-A2`，x86_64，`/dev/calculet0`（PCI `0000:07:00.0`） |
| 驱动 / 固件 | Driver **0.9.3**；DRT **`74832e51`** · `0x03010101`（rt 3.1 / ABI 1.1，线协议 rev 2） |
| 链路 / 芯粒 | PCIe **16.0 GT/s x4**；`Dev: OS up  RT up  chips: 2` |
| 工具链 | rustc **1.98.1-calculet** · docc **0.4.1** · dort **0.4.1** |
| 程序来源 | `jax/review-sd-vae`（修复版：`gn_stats` 种类修正 + layout/up/transpose 两核调用）；5 段源码与修复后 `gen.py` 产物**逐字节相同** |
| 权重 | SD1.5 **fp32** VAE `vae/diffusion_pytorch_model.safetensors`（**334 643 276 B**，248 张量）→ `pack.py`（仅 numpy）打成 5 份 bf16 参数块，**读回自校验 `bad=0`** |
| 板锁 | 本机无 `/opt/data/locks` ⇒ 用 `--board-lock external`（放弃进程间互斥，跑前确认 `dort processes` 无人占用）；CalRT 的 **band 锁**同样拿不到 ⇒ `启动流带：关`，属性能降级（CSR 写不合并），不影响数值 |

## 1. 五段的含义、形状与真机耗时

| 段 | 对应 decoder 子模块 | 输入 → 输出 | 参数块 | `@act` | kernel | 真机耗时 |
|---|---|---|---|---|---|---|
| **a** | `conv_in`(4→512，post_quant 已折入) + `mid_block.resnets.0` + 单头注意力 + `mid_block.resnets.1` | latent `[68][68][16]`(64²×16) → `[68][68][512]`(64²×512) | 21 483 520 B | 37.61 MiB | 14 | **0.9 s** |
| **b1** | `up_blocks.0`：3 resnet + nearest×2 + conv（→128²） | 64²×512 → `[132][132][512]`(128²×512) | 33 452 032 B | 34.62 MiB | 8 | 1.0 s |
| **b2a** | `up_blocks.1` 的 3 resnet（**不上采样**） | 128²×512 → 128²×512 | 28 729 344 B | 67.11 MiB | 6 | 3.9 s |
| **b2b** | `up_blocks.1.upsamplers.0`：nearest×2 + conv 512→512（→256²） | 128²×512 → `[260][260][512]`(256²×512) | 4 722 688 B | 66.02 MiB | 2 | 0.3 s |
| **c** | `up_blocks.2`(512→256) + `up_blocks.3`(256→128) + `conv_norm_out` + `conv_out`(128→3，补到 16) | 256²×512 → 图像 `[516][516][16]` | 12 292 096 B | **748.20 MiB** | 27 | **28.4 s** |
| 合计 | 35 conv + 30 GroupNorm + 1 注意力 + 3 上采样 | latent → 512×512 RGB | 100.68 MB | — | 57 | **34.6 s** |

> 上采样共 3 次，分别在 **b1（64²→128²）、b2b（128²→256²）、c（256²→512²）**；b2a 是 128² 上的第二个阶段
> （所以 5 段只对应 4 个分辨率）。输出都是**带 2 px 边框的 NHWC**（文件开头 8 个 bf16 为 0 是边框，不是全零输出）。
> 分段不是为了建模，而是 **CCU 代码 >1 MiB 时 riscvc 链接器 21 位分支够不着**（ELF 31 KB–1.53 MB）。

## 2. 数值：三个检查点与 torch 参考一致

参考来自 `sd_vae/evidence/stage_summary.json`（torch fp32 逐层 hook）。

| 检查点 | 板上（真机输出） | torch 参考 | 差值 / 折算 |
|---|---|---|---|
| **a** ⇔ `mid_block.resnets.1` | min −19.0 · max 18.875 · mean 0.037610 · std 1.596313 | −19.0790 · 18.8602 · 0.037577 · 1.5962 | min 差 **0.079 ≤ 1 ULP(0.125)**；max 差 0.0148 ≤1 ULP；mean 差 3.3e-5；std 差 1.1e-4 |
| **b2b** ⇔ `up_blocks.1.upsamplers.0` | min −232.0 · max 214.0 · mean −0.809742 · std 18.621119 | −232.7969 · 214.5545 · −0.809622 · 18.6178 | min 差 **0.797 ≤ 1 ULP(1.0)**；max 差 0.55 ≤1 ULP；mean 差 1.2e-4；std 差 3.3e-3 |
| **整图** ⇔ `conv_out` | min −1.8672 · max 1.6484 · mean −0.0898 · std 0.7376 | −1.8671 · 1.6532 · −0.089909 · 0.7381 | min 1e-4；max 4.8e-3；mean 1e-4；std 5e-4 |

### 2.1 整图 PSNR / max|Δ|（真机 `host.py compare`，与 torch fp32 参考）

```json
{ "max_abs": 0.0535825, "mean_abs": 0.00347418, "rmse": 0.00469921, "psnr": 52.5801, "shape": [3,512,512] }
```

| 指标 | **真机** | 仿真器（同口径） | 文档判据 | 判定 |
|---|---|---|---|---|
| PSNR | **52.58 dB** | 52.83 dB | ≥ 40 dB | ✅ 超 12.6 dB |
| max\|Δ\| | **0.0536** | 0.0473 | ≤ 0.08 | ✅ |
| mean\|Δ\| | **0.00347** | 0.00336 | — | ✅ |
| rmse | **0.00470** | 0.00457 | — | ✅ |

> 真机与黄金仿真器只差 **0.25 dB** ⇒ 硬件跑的就是仿真器验过的那套语义（打包/布局/层序/注意力/归一化），没有额外硬件级偏差。
> 参考文件：`_bundle/reference_image.npy`（`(1,3,512,512)` f32，统计与 `stage_summary.json` 的 `conv_out` 差 ≤4e-5，确认是同一 `latent.bin` 的 torch fp32 解码）。

**解读**：min/max 的差全部落在该量级的 1 个 bf16 台阶之内（19 附近 ULP=0.125、232 附近 ULP=1.0），
这是“板上存 bf16、参考是 fp32”的必然差异；mean/std 差 1e-4–3e-3 说明分布完全对上。
⇒ 排除了审查文档担心的两类 bug（只算一半块的半图、NCHW 暂存越界），那类错误会让统计差几个数量级。

`b1` / `b2a` 没有直接参考统计，但夹在已验证的 a 与 b2b 之间，链上任何一段算错都会让 b2b 的 min/max/std 变形。

## 3. 性能（段级端到端，含部署与裸字节 IO）

```text
pkg_a 0.9s · pkg_b1 1.0s · pkg_b2a 3.9s · pkg_b2b 0.3s · pkg_c 28.4s   total 34.6s
```

- **c 段占 82%**：27 个 kernel、6 个 resnet + 上采样 + norm + conv_out，且激活最大（`@act` 748 MiB）⇒ 优化的唯一着力点。
- 小段（b2b 0.3 s）里**进程/部署开销占主导**，不是 kernel 时间。
- ⛔ 这不是逐 kernel 时间，也不是带宽；VAE 这条线没有 `debug.stamp` 插桩。

## 4. 还没做的（要补的判据）

| 项 | 缺什么 | 怎么补 |
|---|---|---|
| ~~逐像素 PSNR / max\|Δ\|~~ | **已完成**：真机 **52.58 dB / 0.0536**（判据 ≥40 dB / ≤0.08）| — |
| `b1` / `b2a` 的直接参考 | 仓库 `stage_summary.json` 未收录这两层 | 用 `compare_stages.py` 对 torch 逐层 dump 比 |
| 逐 kernel 时间 / 带宽 | 无插桩 | 需要给 kernel 加 `debug.stamp`（工作量中等） |

## 5. 复跑

```bash
tar -xzf sd_vae_onboard.tar.gz -C ~ && cd ~/sdvae_pkg
dort device -i 0 | grep "Dev  :"     # 要 OS up  RT up
bash run_on_board.sh . 0             # 已带 --board-lock external
```
