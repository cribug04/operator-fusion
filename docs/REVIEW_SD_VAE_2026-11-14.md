# operator-fusion 仓库审查（master @ cccd266，附带 jax/validation-fixes @ 77b8351）

审查日期 2026-11-14。审查环境：Mac 上只读检出 master；Linux x86_64（OrbStack）装有 calculet 工具链 rustc 1.98.1 / docc 0.4.1 / dort 0.4.1，三仓库 calBLAS 32d2bb8 · calDNN 1e06c41 · npu 172aad2。**没有真机，没有模拟器**，所有结论来自源码、文档、以及在本机实际执行的 cargo / docc / 探针实验。没有安装 torch 和 SD 权重，所以 `emulate.py` 的 52.83 dB 没有复跑，只审了代码。

---

## 1. 结论

**现在拿去真机跑，SD VAE 五段程序大概率出错图，不是因为硬件，而是程序本身有两个确定存在的错误**：

1. 四个按核分块的 calBLAS 模板（`nhwc_to_nchw!` / `nchw_to_nhwc!` / `upsample2!` / `transpose2d!`）被包在 `if core_rank() == 0` 里只由 0 号核调用，但模板内部用 `halves!()` 按 `num_cores()` 分块，而 docc 把 `num_cores()` 折成程序的核数 2。结果 **每次调用只处理前一半数据块**。我用探针程序在 docc 上证实了这一点（§3 实验 A），官方 yolo26 程序也是让两核都调 `upsample2`。全部 5 段共 66 处调用受影响，所有 30 个 GroupNorm 的输入、3 个上采样、注意力的 Kᵀ 都是残缺的。
2. 生成器给 NCHW 暂存区分配的大小是 **0 字节**（`gen.py:265` 匹配的算子种类 `'gn'` 不存在）。b1 / b2a / c 三段的 NCHW 暂存落在声明的 `@act` 之外，分别越界写 4 / 16 / 128 MiB；a 段碰巧与 q 缓冲同址且生命期不冲突。

黄金仿真器 `emulate.py` 按张量名字执行，不建模地址、核分工和 VB，所以 **52.83 dB 对这两个错误完全没有检出能力**。它证明的是层序、权重打包、通道布局与 bf16 舍入点这几件事是对的，这部分价值是真实的。

平台侧数字可信：我在 Linux 上从仓库源码重建 5 段，`.cvx` 大小与 sha256、ccu0/ccu1 ELF 大小、doctor 结果与 README / STAGE_EVIDENCE.md **逐字节一致**。文档总体诚实，真机未验证写得很清楚；个别数字和路径陈旧（§6）。

两处修法我已在生成器副本里改好并重新生成了 5 段，a 段和 c 段在 docc 上重新编译打包的结果见 §3 实验 C。

Resize 分支 jax/validation-fixes 自洽，与 master 合并无冲突（`git merge-tree` 干净）。

---

## 2. 发现清单

严重度：**阻断** = 不改真机必错或越界；**重要** = 影响正确性或可信度；**建议** = 数值/工程改进；**文档** = 文字与事实不符。

| # | 严重度 | 文件:行 | 问题 | 证据 | 建议修法 |
| --- | --- | --- | --- | --- | --- |
| 1 | **阻断** | `sd_vae/gen/gen.py:800-816`；生成物如 `sd_vae/kernels/a/step.rs:24,30,64`、`b1/step.rs:83,85`、`c/step.rs:90` | `nhwc_to_nchw!` / `nchw_to_nhwc!` / `upsample2!` / `transpose2d!` 被包在 `if core_rank() == 0` 里只由 0 号核调用。这四个模板内部都用 `calblas::halves!(…)`（`calBLAS/src/kernels/vision.rs:87-91`、`:412`、`:498`、`:619` 附近）按 `core_rank()/num_cores()` 分块。docc 把 `num_cores()` 折成程序 invocation 的核数 2，于是 0 号核只处理前一半块，另一半从不被写。5 段共 66 处调用（a 11、b1 14、b2a 12、b2b 2、c 27）。 | §3 实验 A：无 `cores` 属性的 kernel 里写 `halves!(64)`，编出的 ccu0 ELF 与手写 `bn = 32`（两核版）**逐字节相同**（sha `287f42f8…`），与手写 `bn = 64`（单核版）不同（`ba729fb4…`）。官方 `calBLAS/examples/yolo26/yolo26n.cvx` 中 `call … @y26_up_20x20_256_p384_384` 不带任何核号分支，两核都调后 `barrier.core.sync`。队友自己在 `sd_vae/SMOKE.md:441` 已把这一点列为“没有数值手段确认”。 | 去掉这四类调用外面的 `if core_rank() == 0 { }`，让两核都调（后面已有 `sync_cores()`）。`fill16k!`、`k_crop`、`k_pad` 不用 `halves!`，保持单核调用即可。§3 实验 B 证明 docc 接受无 `cores` 属性的 kernel 由两核调用（拒收 0，ccu1 得到函数体）。 |
| 2 | **阻断** | `sd_vae/gen/gen.py:264-270` | `max_nchw` 只在 `op['kind'] == 'gn'` 时更新，但算子种类只有 `gn_stats` / `gn_apply`，于是 NCHW 暂存区 `nchw_scratch_1` 大小为 0。分配器给它一个 0 字节的 slot，其后的 slot 或 `@act` 末尾与它同址。 | 运行 `gen.py` 后 `plan.json`：a 段 `nchw_scratch_1 size 0 offset 14270464`（与 `q_11` 同址）；b1 `size 0 offset 32112640 = act_bytes`；b2a `offset 53592064 = act_bytes`；c `offset 650330112 = act_bytes`。即 b1 / b2a / c 的 `nhwc_to_nchw!` 写到 `@act` 声明之外 4 / 16 / 128 MiB（c 段 `up_blocks.3.resnets.0.norm1` 需要 512²×256×2 B）。`docs/SD_VAE_ON_NPU.md:69` 写的“最大 134 MiB”正是 128 MiB = 134,217,728 B，说明早期版本算过、改名为 `gn_stats` 后失效。 | `gen.py:265` 改成 `if op['kind'] == 'gn_stats':`（key 形如 `('gn_stats', hw, c, g)`，`key[2]*key[1]*2` 正确）。修后 `@act`：a 39,436,288 · b1 36,306,944 · b2a 70,369,280 · b2b 不变 · c 784,547,840 B。文档数字要跟着改，calbin 要重新 docc。 |
| 3 | 重要 | `sd_vae/gen/emulate.py` 整体；`docs/SD_VAE_ON_NPU.md:22`“剩下的不确定只剩硬件 kernel 本身的行为” | 仿真器以张量**名字**为键执行，不建模 `@act` 偏移、slot 复用、核分工、VB 容量，所以 #1、#2 这类错误它检不出。文档把 52.83 dB 解释成“程序语义已验到只剩硬件行为”，范围说大了。 | `emulate.py:73-78 buffer()`、`:98-107 layout()` 按名字存取；`gn_stats` 直接 `torch.mean`。 | 文档改为“验证了层序、权重打包/布局、通道窗口、bf16 舍入点；不覆盖地址分配、核分工、硬件近似指令”。可加一个“按 `plan.json` 偏移把所有缓冲放进一条一维 numpy 内存”的仿真模式（Resize 分支 `onboard/resize_reference.py::emulate_flat` 是同思路的小例子），能直接抓到 #2。 |
| 4 | 重要 | `sd_vae/gen/pack.py:112-129`；`docs/SD_VAE_ON_NPU.md:50,86` | `post_quant_conv` 折进 `conv_in` 不是严格等价：原模型是先算 `z' = W_pq z + b_pq`，再对 `z'` 做零填充卷积；折叠后 `b_eff` 把全部 9 个抽头的 `W_in·b_pq` 都加上，但在 64×64 最外一圈像素上，落在图外的抽头本应贡献 0。误差 = Σ(图外抽头) `W_in_tap · b_pq`，只影响边缘 252 个像素。 | 数学推导；a 段仿真 max\|Δ\| 0.212 发生在何处未知（没有权重无法复算）。 | 待验证项：用 `emulate.py` 的 stage a 输出与参考做差，看最大误差是否集中在最外圈。若是，a 段改成先跑一个 1×1 conv（4→4，`pin=16`）再 conv_in，或者保留折叠但把误差写进文档。 |
| 5 | 重要 | `sd_vae/onboard/package_all.sh:9-11,16,19,23` | 脚本写死 `/root/sd_vae/cal_*`、`REF="$W/../ref"`、`$ROOT/SD_VAE_REAL_DEVICE_VERIFICATION.md`。仓库里没有 `sd_vae/ref/`（参考只有 `sd_vae/reference/REFERENCE.md`，`latent.bin` 在 `sd_vae/evidence/`），手册在 `docs/`。从仓库检出直接跑会失败。 | `ls sd_vae` 无 `ref/`；`docs/SD_VAE_ON_NPU.md:36` 的路径说明承认工作区与仓库布局不同，但脚本没改。 | 用参数或环境变量传 calbin 目录和参考目录；`REF` 指向 `sd_vae/evidence`，手册路径改 `docs/`。 |
| 6 | 重要 | `docs/SD_VAE_REAL_DEVICE_VERIFICATION.md:53,96,166`；`sd_vae/gen/host.py:41` | “dort `--raw-in` 要求大小一致（等于 decl）”是推断。`dort calbin run --help` 只说“把这个文件原样写进输入张量”，没有说必须等长。 | `dort calbin run --help` 输出（本机 0.4.1）。 | 标为待验证；补零到 decl 大小本身无害，保留即可，但排错表不要把“尺寸不等”写成确定报错原因。 |
| 7 | 建议 | `sd_vae/gen/gen.py:461-479`（`emit_gn_stats`） | 方差用 `E[x²] − E[x]²`，f32 顺序累加，一组最多 1,048,576 个元素（c 段 `conv_norm_out`：4 通道 × 512²）。均值远大于标准差时有抵消误差；`rsqrt_approx` 是硬件近似。仿真器用 torch 精确均值/方差和精确 `sqrt`，没有建模这两点。 | 代码；`emulate.py:109-123`。 | 两遍法（先均值再 Σ(x−mean)²）或分块 Welford；仿真器加一个“模拟近似 rsqrt”开关做敏感性分析。属精度风险，不是阻断。 |
| 8 | 建议 | `sd_vae/gen/gen.py:617-619,643-657`；`emulate.py:180-185` | 注意力里 kernel 在 bf16 上做 `s − max`、`exp_approx`，输出用 **bf16 的行和** `lb` 做 bf16 除法；仿真器用 f32 的 `s − m`、精确 `exp`、f32 行和除法。每行输出有最多约 2⁻⁸ 的系统性缩放差，仿真没有体现。 | 代码对照。 | 仿真器把 `l` 也舍成 bf16 再除，`s−m` 先舍 bf16；或 kernel 改在 f32 里除（`out.div` 在 f32 cache 上再 cvt）。 |
| 9 | 建议 | `sd_vae/gen/pack.py:139-145`；`sd_vae/STAGE_EVIDENCE.md:36`；`docs/SD_VAE_ON_NPU.md:16` | “自校验 138 个张量逐字节回读一致”只是把刚写进 `blk` 的字节再读出来和刚算出的 `expect` 比较，只能发现偏移重叠，不能发现布局/转置/折叠算错。文档读起来像是独立校验。 | 代码：`checks.append((name, off, arr.nbytes, arr.tobytes()))` 后 `blk[off:off+nbytes] == expect`。 | 文档改成“偏移不重叠校验”；真正的布局校验是 `emulate.py` 从参数块读回权重再卷积这一步（它确实做了，可以明说）。 |
| 10 | 建议 | `sd_vae/onboard/run_on_board.sh:20-29`；`docs/SD_VAE_ON_NPU.md` §8“性能未知” | `timing.txt` 记录的是整条 `dort calbin run` 的墙钟时间，含部署 4.7–33 MB 参数块、进程启动、输入输出搬运。它不是 kernel 延迟，不能用来报性能。 | 脚本内容。 | 文档明确写“端到端含部署耗时”；性能要等确认 dort 的计时口径。 |
| 11 | 建议 | `sd_vae/gen/pack.py:103,113-114`；`gen.py:179-180,188` | 注意力权重用旧键名 `query/key/value/proj_attn`。SD1.5 官方 safetensors 确实用旧名，diffusers 加载时转换；若换成 `to_q/to_k/to_v/to_out.0` 命名的导出文件，`pack.py` 会 KeyError。 | `src[p['src'] + '.weight']` 直接索引。 | 打包时做一次键名映射回退。 |
| 12 | 建议 | `sd_vae/gen/emulate.py:65-70` | `gn_params()` 按旧的 `[rows][8]` 布局读 GN 参数，是死代码；现行 `gn_apply` 用 `[C][16]`。容易误导后续维护。 | 全文件无调用处。 | 删除。 |
| 13 | 建议 | `onboard/package_all.sh`、`onboard/run_on_board.sh`（Resize） | 文件模式 100644，没有可执行位；`sd_vae/onboard/*.sh` 是 100755。 | `git ls-files -s`。 | `chmod +x` 并提交，或文档统一写 `bash xxx.sh`。 |
| 14 | 文档 | `docs/SD_VAE_ON_NPU.md:58` | c 段 `@act` 写 “650.3 MiB”，实际 650,330,112 B = 620.2 MiB（650.3 是 MB）。同表其它段用 MiB。真机手册 `:172` 写 620 MiB 是对的。 | 换算。 | 统一单位；修 #2 后改为 748.2 MiB / 784,547,840 B。 |
| 15 | 文档 | `docs/SD_VAE_ON_NPU.md:69` | “NCHW 暂存 … 最大 134 MiB” 与当前生成器行为（0 字节）不符，见 #2。 | `plan.json`。 | 修 #2 后改为 128 MiB（134,217,728 B）。 |
| 16 | 文档 | `docs/SD_VAE_ON_NPU.md:39,161`；`README.md`“sd_vae/evidence/ 参考图/仿真图/逐段对比 JSON/参考统计” | 文档列出的 `_recon/work/sd_vae/ref/`（`dumps_index.json`、`layer_table.json`、`reference_image.{bin,npy}`）与 `emu_out/` 的 `emulated_image.npy` **不在仓库里**。仓库只有 `sd_vae/evidence/{latent.bin, reference_image.png, emulated_image.png, compare.json, stage_compare.json, stage_summary.json}` 和 `sd_vae/reference/REFERENCE.md`。§6 复现命令引用的 `reference_image.bin` 也拿不到。 | `ls sd_vae/evidence sd_vae/reference`。 | 把 `ref_image.npy`（3 MB）入库，或在文档写清哪些只在上板包里。 |
| 17 | 文档 | `docs/SD_VAE_ON_NPU.md:95`、`sd_vae/SMOKE.md:434`、`docs/SD_VAE_ON_NPU.md:69` | 正文把“布局/上采样 kernel 是 1 核、包在 `if core_rank()==0`”写成定论，而 `SMOKE.md:441` 自己承认这是未验证推断；`:69` 又说“两核分”。三处互相矛盾。 | 原文。 | 按 #1 的实验结论统一改写。 |
| 18 | 文档 | `sd_vae/gen/host.py:31` | 注释“4 KiB 对齐到 163,840”：147,968 按 4 KiB 对齐是 149,504；163,840 是 16 KiB（`FILL_BLK`）对齐。 | 算术。 | 改注释。 |
| 19 | 文档 | `sd_vae/onboard/sd_vae_onboard_pkg.tar.gz.sha256:1`；`docs/SD_VAE_ON_NPU.md:36`；`docs/CALCULET_RECON.md` 新增两行 | 含个人机器路径 `/mnt/c/Users/10463/...`、`C:\Users\10463\AppData\...`。不是密钥，但是私有信息；仓库公开时建议去掉。扫描未发现 token / 密码 / 私钥。 | `grep -rn "Users/10463"`。 | sha256 文件只留哈希和文件名。 |
| 20 | 文档 | `docs/SD_VAE_REAL_DEVICE_VERIFICATION.md:17` | “一次 5 段串行约几十秒~几分钟”没有任何测量依据。 | 无数据。 | 改为“未测”。 |

### 核对无误的项（正面结论，便于对外引用）

- 5 段源码重建：`.cvx` 246,826 / 161,321 / 71,967 / 23,085 / 278,729 B，sha256 与 `STAGE_EVIDENCE.md` §1 五个值完全一致；ccu0/ccu1 ELF 1,414,648/1,049,156 · 968,536/714,864 · 398,884/145,552 · 31,160/29,820 · 1,528,008/959,116 B 一致；5 段 doctor 拒收 0、警告 2（ping/pong 同址）。参数块大小合计 100,679,680 B 与 `w` decl 一致。
- 层表与 diffusers `AutoencoderKL`（`block_out_channels=[128,256,512,512]`, `layers_per_block=2`）一致：conv_in 4→512；mid 2 resnet + 1 注意力；up_blocks 通道 512→512→512→256→128，各 3 个 resnet，前 3 块带 nearest×2 + conv；`up_blocks.2.resnets.0`（512→256）与 `up_blocks.3.resnets.0`（256→128）带 1×1 shortcut；conv_norm_out GN(32,128)+SiLU；conv_out 128→3（补 16）。注意力残差用 `proj_attn` 的 1×1 conv 带 `res=x`（GN 之前的输入）实现，与 diffusers `residual_connection=True`、`rescale_output_factor=1` 一致；`SCALE = 0.044194174 = 1/√512`。
- `pack.py` 权重布局 `[k²][CI][CO]`，`flat[(ky·K+kx), ci, co] = W[co, ci, ky, kx]`，与 `conv2d!` 的 `rd!(w, [K*K*CI, CO])` 视图及 `@wtap` 取 `[(tap)*CI + r0, c0]` 一致；bf16 RNE 位运算正确；GN 参数 `[C][16]` 与 `k_gna` 的 `gg.sub([c,0],[1,16])` 一致；bias f32 `[CO_pad]` 一致。折叠公式 `W_eff` 的 einsum 下标正确（边缘例外见 #4）。
- 段间字节数链：`out_y` 与下一段 `in_x` 都由 `img_size()` 算出，4,734,976 → 17,842,176 → 17,842,176 → 69,222,400 → 8,536,064 B，与真机手册 §4.2 表一致。`dort calbin run` 的 `-f/-m/-i/--raw-in/--raw-out/--in-tensor/--out-tensor/--deploy-only/--dump-layout` 都是 0.4.1 的真实选项。
- `host.py image` 取 `[2:514, 2:514, 0:3]` 正确；`host.py input` 把 z 放进前 4 通道、边框留零正确；`latent.bin` 就是 `/0.18215` 后的 z，与参考一致。
- 分配器的生命期分析（`finish()`）对 `res` 输入有记录，残差缓冲不会被提前复用；各图像 slot 在程序开头整块清零，`conv2d!` / `nchw_to_nhwc!` / `upsample2!` / `k_pad` 只写内部，边框保持为零；每段开头清 `out_y` 使段间裸字节衔接成立。
- Resize 分支：`kernels/`、`patches/onboard_kit.patch`、Linux 构建副本三方逐行一致；`DESC` 加入后 `.cvx` 哈希不变；与 master 合并无冲突。

---

## 3. 实验记录（本机 Linux，docc 0.4.1）

**实验 A：无 `cores` 属性的 kernel 里 `num_cores()` 折成多少。** 五个只差一行的 crate，kernel 都是 `#[npu::kernel] pub fn probe(y)`，循环 `for i in 0..bn` 写 `[b0+i]` 行；程序 `cores = 2`，`if core_rank()==0 { probe(...) }`。

| 变体 | `(b0, bn)` 的写法 | ccu0 ELF sha256 前 16 位 |
| --- | --- | --- |
| h | `calblas::halves!(64)` | `287f42f8eccc858d` |
| c1 | `(0, 64)` | `ba729fb43908adc2` |
| c1b | `halves!` 展开式手写、`num_cores` 代入 1 | `ba729fb43908adc2` |
| c2 | `(core_rank()*32, 32)` | `287f42f8eccc858d` |
| c2b | `halves!` 展开式手写、`num_cores` 代入 2 | `287f42f8eccc858d` |

`h` 与 `c2/c2b` 逐字节相同，与 `c1/c1b` 不同；反汇编 diff 对 c2b 为 0 行。结论：`num_cores()` = 2。五个变体 ccu1 ELF 全同（kernel 只在 ccu0 上）。

**实验 B：无 `cores` 属性的 kernel 由两核调用是否被 docc 接受。** 把 `h` 的 `if core_rank()==0` 去掉：`from-calvx` exit 0，doctor 拒收 0、警告 1；ccu1 ELF 变为 `f8a405bddfaeb394`（7,144 B，得到了函数体）。

**实验 C：修后的生成器能否编译打包。** 对 `gen.py` 做 #1、#2 两处修改，重新生成 a、c 两段并在 Linux 上编译、docc、doctor：结果见本文末尾“实验 C 结果”。

**实验 D：5 段原样重建**（见 §2 正面结论第一条）。

---

## 4. “疑似错误”专列（待验证）

| 项 | 怀疑 | 验证条件 |
| --- | --- | --- |
| #4 `post_quant_conv` 折叠的边缘误差 | 最外圈 252 像素有系统性偏差 | 需要 SD1.5 safetensors + torch：比较 a 段仿真输出与参考的误差图，看是否集中在边缘；或直接算 `Σ_{图外抽头} W_in·b_pq` 的量级 |
| #2 的越界写在板上落到哪 | `submodel_memory_reserved_info.txt` 只列 `param_blk0`/`in_x`/`out_y` 的 vaddr，`@act` 的落点不在其中；越界 4–128 MiB 可能写进空闲保留区（侥幸无害）或别的张量 | 板上 `dort calbin run --deploy-only --dump-layout` 看 workspace 实际落点；无论如何应先修 #2 |
| #7 `rsqrt_approx` / #8 `exp_approx`、bf16 除法的真实精度 | 30 层 GN 累积后能否守住 40 dB 判据 | 需要真机或模拟器；或向主办方要 KS1 近似指令的误差规格 |
| `dort --raw-in` 是否要求文件长度等于 decl | 手册当作确定事实 | 真机上喂一个 147,968 B 的文件看是否报错 |
| 所有 `#[npu::kernel]` 单核 kernel 的 `core_rank()` 在只有 0 号核调用时是否恒为 0 | 实验 A 只证明 `num_cores()` 折成 2，`core_rank()` 行为符合预期（否则 c2 变体也对不上），但没有在真机上验证 | 真机探针：按手册 SMOKE.md:441 的建议写一行已知数据读回 |
| `conv2d!` 在 `pin = (16, 0)`、`cin = 16` 的首层是否被 docc 的 `aligned_args` 接受 | docc 已过，但 `kabi.json` 的运行时前置条件没有核 | 真机部署时按 `kabi.json` 核实参 |

---

## 5. 上板前必改项

按“不改必挂”排序：

1. **#1 两核调用**：现状每个 GN 的 NCHW 输入只有前一半块是新数据（后一半是上一次的残留或零），每次上采样只写上半图，Kᵀ 只有一半——图像一定错，而且错得不像“精度问题”，像花屏/半图。修法是删掉 66 处 `if core_rank() == 0 { … }` 包裹（生成器 3 处改动），保留 `fill16k` / `k_crop` / `k_pad` 的单核调用。
2. **#2 NCHW 暂存 0 字节**：b1 / b2a / c 越界写。生成器 1 行改动。修后 c 段 `@act` 748 MiB，要确认板上保留区放得下（`dram_rsvd` 12.86 GB，应可）。
3. 修完 1、2 后 **重新跑 `emulate.py`**（数值应不变，因为仿真器不受这两处影响）、重新 docc 5 段、更新 STAGE_EVIDENCE 里 `.cvx`/ELF/`@act` 数字和上板包。
4. 上板第一步不要直接跑 5 段。先跑 a 段并用 `host.py stage out_a.bin 64 512` 看量级，再用 `in_x` 的已知输入做一个最小探针（比如只跑 `k_fill0` + `nhwc_to_nchw!` + `nchw_to_nhwc!` 往返，期望逐位等于输入）确认实验 A 的结论在硬件上成立。
5. 判据 40 dB / 0.08 建立在仿真器 52.83 dB 之上，而仿真器对 #7、#8 两类硬件近似没有建模，留的余量是否够只能真机见。如果结果在 30–40 dB，不要按排错表当“正常范围下限”放过，先对照中间段。

---

## 6. 文档与代码不一致清单

| 位置 | 文档说 | 实际 |
| --- | --- | --- |
| `docs/SD_VAE_ON_NPU.md:58` | c 段 `@act` 650.3 MiB | 650,330,112 B = 620.2 MiB；修 #2 后 784,547,840 B |
| `docs/SD_VAE_ON_NPU.md:69` | NCHW 暂存最大 134 MiB | 当前生成器分配 0 B；应为 128 MiB |
| `docs/SD_VAE_ON_NPU.md:69` vs `:95` vs `SMOKE.md:434,441` | “两核分” / “cores=1 包在 if 里” / “未验证” | 见 #1、#17 |
| `docs/SD_VAE_ON_NPU.md:39,161`、`README.md` | `ref/` 下有 `reference_image.{bin,npy}`、`layer_table.json`、`dumps_index.json`；`emu_out/*.npy` | 仓库中不存在；只有 png/json/latent.bin |
| `docs/SD_VAE_ON_NPU.md:16`、`STAGE_EVIDENCE.md:36` | 打包器“逐字节回读一致”作为自校验 | 只是写后读回，不校验布局 |
| `docs/SD_VAE_ON_NPU.md:22` | 仿真后“不确定只剩硬件 kernel 行为” | 地址/核分工/VB 不在仿真范围 |
| `docs/SD_VAE_REAL_DEVICE_VERIFICATION.md:53,96,166` | `--raw-in` 要求等长 | dort 帮助未说明 |
| `docs/SD_VAE_REAL_DEVICE_VERIFICATION.md:17` | 约几十秒到几分钟 | 未测 |
| `sd_vae/onboard/package_all.sh` | 可从仓库复现上板包 | 路径指向工作区与 `/root/sd_vae`，仓库内不可运行 |
| `sd_vae/gen/host.py:31` | 4 KiB 对齐 | 16 KiB |
| `sd_vae/gen/emulate.py:65` | — | 死代码 `gn_params`（旧布局） |
| `README.md` 2.5 节 | 数字与 STAGE_EVIDENCE 一致 | 核对一致（正面） |

---

## 7. 加分项建议（按投入产出）

1. **先修 #1、#2，再做一个“地址级仿真”**（半天）：把 `emulate.py` 改成按 `plan.json` 的 `offset` 在一条一维 bf16 数组上读写，并按 `halves!` 规则模拟两核各写一半。这一步能把本次两个阻断级错误在没有板子的情况下抓出来，也是后续 UNet 移植最值钱的基础设施。
2. **真机最小探针先行**（有板后 30 分钟）：三段式小程序验证 `num_cores()` 语义、`nhwc_to_nchw!` 往返逐位、`k_fill0` 清零。通过后再跑 5 段，排错范围立刻缩小。
3. **把 Resize 接进 VAE 的 b1/b2b 上采样作为可选替换**：现在 nearest ×2 来自 `upsample2!`；Resize 分支的 `resize_bilinear2!` 同样是 border 无关的 NHWC 布局（无边框），要接进去需要加边框版本。这能把两条线合成一个故事（Resize 服务于 SD），但只有在 1、2 完成且真机跑通后才值得做。

---

## 实验 C 结果

对 `gen.py` 做 #1（去掉四类模板调用外的 `if core_rank() == 0`）与 #2（`'gn'` → `'gn_stats'`）两处修改后重新生成并在 Linux 上编译、`docc calbin from-calvx`、`doctor`：

| 段 | build | `.cvx` B | from-calvx | doctor | ccu0 ELF B | ccu1 ELF B（修前） |
| --- | --- | --- | --- | --- | --- | --- |
| a | exit 0 | 243,366 | exit 0 | 拒收 0，警告 2 | 1,414,672 | 1,411,672（修前 1,049,156） |
| c | exit 0 | 270,253 | exit 0 | 拒收 0，警告 2 | 1,528,008 | 1,525,764（修前 959,116） |

ccu1 的 ELF 变大，说明布局/上采样/转置 kernel 现在确实进入了 1 号核；两段都没有触发 riscvc 21 位分支链接错误。修改后的生成器副本：`gen_fixed.py`（与原文件相比只有 4 处差异，见 `gen_fixed.diff`）。b1 / b2a / b2b 只重新生成未编译。数值仿真未重跑（本机无 torch 与权重）；按仿真器的实现方式，这两处修改不会改变它的输出。
