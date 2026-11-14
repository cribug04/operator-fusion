# SD1.5 VAE 解码器 · NPU 真机验证手册

**对象**：SD1.5 VAE decoder（全 NPU，5 段 CalVX 程序）：输入 latent `[4,64,64]`，输出 512×512 RGB 图像。
**目的**：在 Calculet NPU 板卡上把 5 段 calbin 依次跑完，得到图像，和 torch/diffusers fp32 黄金参考比对，判定移植是否成功。
**现状**：平台侧已全通（5/5 设备遍编译、docc calbin、doctor 拒收 0、反汇编 0 unknown、构建逐字节确定）；本地黄金仿真与参考 **PSNR 52.83 dB / max|Δ| 0.0473**。真机数值是本手册要补的最后一步。

---

## 1. 前置条件

| 项 | 要求 |
| --- | --- |
| 板卡 | 任意一块可用的 Calculet NPU 板（程序 `chiplets = 1, cores = 2`，双核 CCU） |
| 板端 | `dort` 可用（工具链 0.4.1）；驱动已装（`/dev/calculet*` 出现）；板子健康：`OS up  RT up` |
| 主机 | 能 SSH 到板子的 Linux 机器；`dort`、`python3` + `numpy`（PIL 可选，仅用于存 PNG） |
| 文件 | `sd_vae_onboard_pkg.tar.gz`（约 107 MB，内容见 §2） |
| 资源 | 板子空闲（一次 5 段串行约几十秒~几分钟，取决于板子/排队；部署参数块 4.7–33 MB/段） |

---

## 2. 上板包内容与校验

```
sd_vae_onboard_pkg/
├── cal_a/  cal_b1/  cal_b2a/  cal_b2b/  cal_c/   # docc calbin 目录（各含 param_blk0.bin 与 pkg_*/ 下的 CCU ELF）
├── in_x.bin          # 段 A 的输入：latent ⇒ border bf16 [68][68][16] 再补零到 decl 大小，163,840 B
├── latent.bin        # 原始 z（f32 [4,64,64]，= randn/0.18215），留档
├── ref_image.npy     # torch/diffusers fp32 参考图 [3,512,512] f32（未 clamp）
├── ref_image.png     # 参考图（人工看）
├── host.py           # 主机侧工具：input / image / stage / compare / raw
├── run_on_board.sh   # 一键脚本：5 段串起来 + 出图 + 对比 + 记录耗时
├── MANIFEST.txt      # 文件清单（相对路径 + 字节数）
└── SHA256SUMS.txt    # 包内文件 sha256
```

解包后先校验：

```bash
tar xzf sd_vae_onboard_pkg.tar.gz && cd sd_vae_onboard_pkg
sha256sum -c SHA256SUMS.txt          # 期望全部 OK
```

关键文件 sha256（与 SHA256SUMS.txt 一致，可人工核对）：

| 文件 | 字节 | sha256 |
| --- | --- | --- |
| `in_x.bin` | 163,840 | `cf3eaee6bdb2c5cb05d6dc45d47a2467d172865dae591140e3360ee1b3784ea8` |
| `latent.bin` | 65,536 | `79ed341043fe4be1f398e53a3af03d190d0a189fbc23034570acba4e8330a031` |
| `ref_image.npy` | 3,145,856 | `b7cc6839afa2b5a6d0d16b73bc061216d5cc4fcab032e39f3a3f7e8442bce126` |
| `run_on_board.sh` | 1,789 | `c1d51225c8cbee87e2553423fd4ee3433dee73c89489ea4fc62236939e2c2b56` |
| `host.py` | 5,362 | `0afe3eca2ab3b65aff1c8c3b731c77be0bd2bc577f73dcda91fd35005c725a54` |

> `in_x.bin` 是**已经按设备布局摆好的输入**（border NHWC bf16，前 4 通道是 z，其余 0；文件补零到 163,840 B 正好等于程序里 `in_x` 的声明大小，dort `--raw-in` 要求大小一致）。⛔ 不要拿 `latent.bin` 直接当输入喂。

---

## 3. 板子健康检查（每次上板前）

```bash
dort device -i 0
```

要求输出里有 **`Dev : ... OS up   RT up`**。

不是 `RT up` 时（照仓库 AGENTS.md 的恢复流程，**只有 `reprobe` 会重启板上 RT**）：

```bash
dort device reprobe -i 0 --yes        # SOFT_RESET，等 15 秒
sudo rmmod calculet_pci && sudo insmod ~/cal-pcie-driver/calculet_pci.ko
# reprobe 后设备节点可能变成 /dev/calculet1，这条 insmod 是归位；再等 RT up 再跑
dort device -i 0 | grep "Dev  :"
```

> ⛔ 只做 `rmmod/insmod` 而不 `reprobe`，会一直停在 `OS up / RT DOWN`，所有 calbin 报 `configure 失败：NoDevice`。
> 如果是 16 芯粒验收机或共享板，不要自己复位，交给人处理。

---

## 4. 运行（两种方式，任选）

### 4.1 一键（推荐）

```bash
cd sd_vae_onboard_pkg
./run_on_board.sh . 0
```

脚本按顺序跑 5 段，任何一段 `rc≠0` 立即停；产物：`out_a.bin`、`out_b1.bin`、`out_b2a.bin`、`out_b2b.bin`、`out_c.bin`、`image.npy`、`image.png`、`compare.json`、`timing.txt`。

### 4.2 逐段手跑（排查时用）

> `dort calbin run` 的裸字节模式：`--raw-in` 把文件**原样**写进输入张量，`--raw-out` 把输出张量**原样**写出；输入/输出张量名就是程序里 `npu::decls!` 声明的 `in_x` / `out_y`。

| # | 命令（在包目录里执行） | 输入 | 输出 | 输出字节 |
| --- | --- | --- | --- | --- |
| 1 | `dort calbin run -i 0 -f cal_a -m pkg_a --raw-in in_x.bin --raw-out out_a.bin --in-tensor in_x --out-tensor out_y` | `in_x.bin` **163,840 B**（图像 147,968 + 补零） | `out_a.bin` | 4,734,976 |
| 2 | `dort calbin run -i 0 -f cal_b1 -m pkg_b1 --raw-in out_a.bin --raw-out out_b1.bin --in-tensor in_x --out-tensor out_y` | `out_a.bin` 4,734,976 B | `out_b1.bin` | 17,842,176 |
| 3 | `dort calbin run -i 0 -f cal_b2a -m pkg_b2a --raw-in out_b1.bin --raw-out out_b2a.bin --in-tensor in_x --out-tensor out_y` | `out_b1.bin` 17,842,176 B | `out_b2a.bin` | 17,842,176 |
| 4 | `dort calbin run -i 0 -f cal_b2b -m pkg_b2b --raw-in out_b2a.bin --raw-out out_b2b.bin --in-tensor in_x --out-tensor out_y` | `out_b2a.bin` 17,842,176 B | `out_b2b.bin` | 69,222,400 |
| 5 | `dort calbin run -i 0 -f cal_c -m pkg_c --raw-in out_b2b.bin --raw-out out_c.bin --in-tensor in_x --out-tensor out_y` | `out_b2b.bin` 69,222,400 B | `out_c.bin` | 8,536,064 |

每段的输入/输出**字节数必须完全匹配上表**（错一个字节就是拿错文件/上一段没跑）。段间直接传原始字节（同一 border NHWC bf16 布局，段程序开头会把输出边框清零，所以不需要主机再做处理）。

可选：部署自检（不推理，确认部署量不是 0）：
```bash
dort calbin run -i 0 -f cal_a -m pkg_a --deploy-only --dump-layout
# 看 deploy 的字节数与各 section 落点；"deployed 0 bytes" 说明 calbin/ELF 不对，先别推理
```

各段参数块大小（部署时写入，失败首先看这里）：a 21,483,520 B；b1 33,452,032 B；b2a 28,729,344 B；b2b 4,722,688 B；c 12,292,096 B。

---

## 5. 出图与数值对比

```bash
python3 host.py image out_c.bin image.npy image.png
python3 host.py compare image.npy ref_image.npy compare.json
cat compare.json
```

- `out_c.bin` 是 `[516][516][16]` 的 border bf16（`[H+4][W+4][P]`，P=16），**只读内部 `[2:514, 2:514, 0:3]`**；`host.py image` 会正确取 512×512 RGB，**不要自己按 3 通道硬解析**（第 4~16 通道是补齐的 0）。
- 中间段自检（可选；验证每段输出量级对不对，避免最后才发现）：

```bash
python3 host.py stage out_a.bin 64 512      # 段 A：64×64×512
python3 host.py stage out_b1.bin 128 512    # 段 B1
python3 host.py stage out_b2a.bin 128 512   # 段 B2A
python3 host.py stage out_b2b.bin 256 512   # 段 B2B
python3 host.py raw out_c.bin               # 字节数 + sha256
```

**参考量级**（本地黄金仿真同程序跑出来的期望值；板卡数值应“量级一致”，不要求逐位）：

| 段 | 输出内部 min / max / mean / std |
| --- | --- |
| a | -19.00 / 18.875 / 0.0376 / 1.5964 |
| b1 | -79.00 / 64.00 / -0.0326 / 6.4385 |
| b2a | -98.00 / 73.50 / -0.2113 / 5.4225 |
| b2b | -233.0 / 214.0 / -0.8094 / 18.6220 |
| c（只取 3 通道图像） | -1.8672 / 1.6484 / -0.0898 / 0.7379 |

> 若中间段量级差很多（比如全 0、全 NaN、或 10× 以上），不必等最后：先只跑到该段，用 `host.py stage` 看，再按 §7 排查。

---

## 6. 通过判据

| 项 | 通过标准 | 本地仿真基准 |
| --- | --- | --- |
| 5 段 | 每段 `rc = 0`；板子保持 `OS up RT up` | — |
| 图像数值 | `compare.json`：**PSNR ≥ 40 dB 且 max_abs ≤ 0.08**、mean_abs ≤ 0.02 | PSNR **52.83 dB**，max 0.0473，mean 0.00336 |
| 图像外观 | `image.png` 与 `ref_image.png` 视觉一致（随机 latent 解码出的抽象图；结构/色调相同即可） | 两张图肉眼无差 |
| 确定性（可选） | 同一条命令重跑一次：两次 `image.npy` 的 max\|Δ\| = 0（或 `raw out_c.bin` 内部区 sha256 相同） | 仿真确定性已验 |
| 记录 | `timing.txt`（每段耗时 + 总耗时）与 `compare.json` 归档 | — |

---

## 7. 排错表

| 症状 | 最可能原因 | 处理 |
| --- | --- | --- |
| `dort device -i 0` 无设备 / `未发现 Calculet 设备` | 驱动未装、板子没插好、RT 没起 | 先 `lspci \| grep -i calculet`；再按 §3 恢复（reprobe + rmmod/insmod） |
| `configure 失败：NoDevice` | 板子 RT DOWN | §3 恢复流程；不要只 rmmod/insmod |
| `deployed 0 bytes` / `flying/flying` 挂死 | calbin 目录不完整（缺 ELF/参数块）或 `-m` 子模型名不对 | 核对 `ls cal_a/pkg_a` 的 ELF；子模型名必须是 `pkg_a`…`pkg_c`；用 `--deploy-only --dump-layout` 先看 |
| `raw-in` 尺寸不符 / 读输入报错 | 文件拿错或大小不对（最常见：喂成 147,968 B 的裸图像、`latent.bin`、或上一段没用 `-m` 指定子模型） | 第一段必须喂**补零到 163,840 B** 的 `in_x.bin`；对照 §4.2 的字节数表 |
| 某段输出全 0 | 上一段没真正跑完 / 输入文件是 0 字节 / 段链断开 | 从第 1 段逐段重跑；`ls -l out_*.bin` 看大小 |
| 图像是噪声状条纹/花屏 | 段链中某段失败但 rc 被忽略；或输入布局不对 | 逐段 `host.py stage` 对照 §5 量级；确认 `in_x.bin` 的 sha256 |
| 图像大体对但边缘/角落异常 | 输入/输出边框没清零（旧版程序才会） | 用本包（程序会在开头清 `out_y` 边框）；不要自己拼中间文件 |
| PSNR 30–40 dB | 硬件 `exp/rsqrt` 近似 + bf16 的累积误差 | 属正常范围下限；只要 ≥40 dB 即通过。若 <30 dB 按“花屏”逐段查 |
| 重复跑第二张图结果异常 | 输出/输入 decl 未开 `double_buffer`（doctor 有 ping/pong 警告） | 一张图跑一次 `dort calbin run`（每次都会重新部署）；连续多张需重新生成带 `double_buffer` 的包 |
| 段 5 很慢 / 内存告警 | 段 c 的 `@act` 620 MiB（最大），512² 特征图 | 属预期；看 `timing.txt`，只要没挂就让它跑完 |

---

## 8. 记录模板（请照抄归档）

```
板卡：________（型号/序列号）  固件/驱动：________  dort：0.4.1  日期：____
SD1.5 VAE 上板包 sha256（tar.gz）：________
每段：
  a   rc=__  耗时=__ s    out_a.bin   bytes=______  sha256=______
  b1  rc=__  耗时=__ s    out_b1.bin  bytes=______  sha256=______
  b2a rc=__  耗时=__ s    out_b2a.bin bytes=______  sha256=______
  b2b rc=__  耗时=__ s    out_b2b.bin bytes=______  sha256=______
  c   rc=__  耗时=__ s    out_c.bin   bytes=______  sha256=______
compare.json：psnr=____ dB  max_abs=____  mean_abs=____  rmse=____
图像：image.png（归档）  vs  ref_image.png（视觉一致：是/否）
确定性（重跑一次）：max|Δ|=____
结论：PASS / FAIL（原因：________）
产物归档路径：board_out_YYYYMMDD/（包含 5 个 out_*.bin、image.npy/png、compare.json、timing.txt、dort 输出日志）
```

---

## 9. 归档与回传

1. 把 `out_c.bin`（8.5 MB）、`image.npy/png`、`compare.json`、`timing.txt`、`dort ... > log 2>&1` 的日志拷到 `<包目录>/board_out_YYYYMMDD/`；
2. 把该目录提交到 Gitee 仓库 `sd_vae/evidence/board_out_YYYYMMDD/`（或贴关键数值 + sha256 到提交说明）；
3. 如果 PSNR 达到 40 dB 以上，就把 `SD_VAE_ON_NPU.md` §1 的“真机数值：❌ 未验证”改成实测值，并把本手册 §8 记录附在提交里。

---

## 10. 附：为什么判据是这套

- 参考是 diffusers 0.43 fp32 在 CPU 上跑的**逐层 hook 黄金参考**（确定性：同输入两次 decode 逐 bit 相同；权重 248/248 与 safetensors 逐 bit 对齐）。
- 我们本地“黄金仿真器”按 `step.rs` 的算子顺序与布局执行同一份程序（同 border/NHWC 布局、同 bf16 舍入点、同权重块），与参考 PSNR 52.83 dB ⇒ 说明打包/布局/层序/注意力/归一化都对；真机只多出“硬件近似指令（exp/rsqrt ≤1 ULP）与两核同步”两项差异，所以判据留到 40 dB / 0.08。
- 中间段 PSNR 比最终图低是正常的：这些张量值域到 ±233，bf16 表示误差就大；到最后一层归一化+卷积后回到 [-1.9,1.65]，误差也被压回 0.05 以内。
