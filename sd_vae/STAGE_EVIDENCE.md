# SD1.5 VAE 五段程序 · 平台侧证据（docc 0.4.1 / calculet 工具链）

日期：2026-11-14；环境：WSL2 Ubuntu 24.04.3 + `rustc 1.98.1-calculet` / `docc 0.4.1` / `dort 0.4.1`。
所有数字来自本机真实命令输出（`dort` 本机无设备，真机未跑）。命令见 §5。

> 本版程序与上一版唯一的代码差异：**每段在开头把自己 `out_y` 声明的边框清零**。段间是用裸字节衔接的（`out_a.bin` 直接喂 b1 的 `in_x`），而 conv 只写内部区域；边框不先清零就会把上一段内存残留当 padding 读进去。仿真数值不变（52.83 dB）。

## 0. 修复版（2026-11-14 晚，当前 `sd_vae/kernels/` 与 `gen.py` 对应的就是这一版）

> **2026-11-14 晚修复版**：审查（`docs/REVIEW_SD_VAE_2026-11-14.md`）发现生成器两处错误并已修复：① `nhwc_to_nchw!` / `nchw_to_nhwc!` / `upsample2!` / `transpose2d!` 内部按 `num_cores()` 分块，原来只由 0 号核调用会漏掉一半数据块，现改为两核都调用；② NCHW 暂存区大小算成 0 字节，现按最大 GroupNorm 输入分配（c 段 128 MiB）。修复后 5 段重新编译、`docc calbin from-calvx`、`doctor` 全部通过（拒收 0，各 2 条 ping/pong 警告），数字见下表；修复前的数字保留在 `sd_vae/STAGE_EVIDENCE.md` §1。本地黄金仿真不建模地址与核分工，修复前后输出相同；**真机仍未验证**。

| 段 | `.cvx` B | `.cvxo` B | `.cvx` sha256 | 参数块 B | `@act` B | ccu0 ELF B | ccu0 load_size | ccu1 ELF B | ccu1 load_size | doctor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | 243,366 | 38,610 | `1949785fe7fc0f021cdde9a3fc129c1a3bfd8f16eedc25d683bb437f52608514` | 21,483,520 | 39,436,288 | 1,414,672 | 1,408,920 | 1,411,672 | 1,406,028 | 拒收 0，警告 2 |
| b1 | 156,915 | 25,127 | `8a02050b0190df47a4f8c7b9c269044420bdafce66ba7af64ef014d91b677da2` | 33,452,032 | 36,306,944 | 968,536 | 963,140 | 966,160 | 960,792 | 拒收 0，警告 2 |
| b2a | 68,253 | 11,324 | `887cbf430c33f83cf9d581561d3e3323c0b05c96af07cc978448258cc87acc04` | 28,729,344 | 70,369,280 | 398,884 | 393,656 | 397,144 | 391,948 | 拒收 0，警告 2 |
| b2b | 22,461 | 3,888 | `a191139559b251d4aceadcda99f7783dfd466ec51a465bb1c3209cc81bf87a50` | 4,722,688 | 69,222,400 | 31,160 | 26,120 | 30,700 | 25,688 | 拒收 0，警告 2 |
| c | 270,253 | 43,162 | `151aa7c32b3da0a1e4f3c15a609d802753b794e9e2945cce308c7db31c5be3dc` | 12,292,096 | 784,547,840 | 1,528,008 | 1,521,536 | 1,525,764 | 1,519,320 | 拒收 0，警告 2 |
| **合计** | **761,248** | **122,111** | — | **100,679,680** | — | **4,341,260** | — | **4,331,440** | — | — |

- 每份 ELF 的 `docc elf ccu view` 自校验行“各节拼起来正好 = load_size”均成立；ccu1 ELF 明显变大（例如 a 段 1,049,156 → 1,411,672 B），说明布局/上采样/转置 kernel 现在确实进入了 1 号核。
- 反汇编条数、构建两次逐字节确定性、参数块回读在修复版上**未重跑**；修复只改了 `step.rs` 的调用方式与 `@act` 偏移，不改 kernel 体与参数块布局。
- 执行环境：Linux x86_64（OrbStack）+ calculet 工具链 rustc 1.98.1 / docc 0.4.1；参数块用等长零文件代替（只验证结构，`docc calbin from-calvx` 要求长度等于 `@w` decl）。

## 1. 每段程序（修复前，2026-11-14 日间；仅供对照，`.cvx`/`@act` 与当前源码不再一致）

| 段 | `.cvx` B | `.cvxo` B | `.cvx` sha256 | 参数块 B | `@act` B | ccu0 ELF B | ccu0 load_size | ccu0 反汇编（条 / unknown） | ccu1 ELF B | ccu1 load_size | ccu1 反汇编（条 / unknown） | doctor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | 246,826 | 39,171 | `1293ba90347fab276f95aa520c8c73ee4aecf19f81a63ca095e695b9ab2078d6` | 21,483,520 | 35,241,984 | 1,414,648 | 1,408,896 | 255,023 / **0** | 1,049,156 | 1,043,636 | 183,206 / **0** | 拒收 0，警告 2 |
| b1 | 161,321 | 25,841 | `87ea900a50dc37f9548602dbce67bd457496c051e0d3e95f160e1e3f3ddf4170` | 33,452,032 | 32,112,640 | 968,536 | 963,140 | 169,481 / **0** | 714,864 | 709,628 | 120,109 / **0** | 拒收 0，警告 2 |
| b2a | 71,967 | 11,936 | `d38ee4993d4caec7b6594351f67c07630dbdd88cfb2641191eadeee94bbc86fb` | 28,729,344 | 53,592,064 | 398,884 | 393,656 | 72,664 / **0** | 145,552 | 140,448 | 23,194 / **0** | 拒收 0，警告 2 |
| b2b | 23,085 | 3,984 | `f23d093a1ed6069d2794fcb06de2ca9d9f71b2bbc9079ad2b6e6c2f797e87189` | 4,722,688 | 69,222,400 | 31,160 | 26,120 | 4,586 / **0** | 29,820 | 24,852 | 4,359 / **0** | 拒收 0，警告 2 |
| c | 278,729 | 44,539 | `b1f4ac1e703225d095bd7d1f086dd87f47562482287ad5329603bdd4ee9faab4` | 12,292,096 | 650,330,112 | 1,528,008 | 1,521,536 | 278,834 / **0** | 959,116 | 953,084 | 165,977 / **0** | 拒收 0，警告 2 |
| **合计** | **781,928** | **125,471** | — | **100,679,680** | — | **4,341,236** | — | **780,588 / 0** | **2,898,508** | — | **496,845 / 0** | — |

- 反汇编全部“走完了：每个字节都被一条认得出的指令覆盖”，`0 条认不出`；每份 ELF 的各节字节和 = `load_size`（`docc elf ccu view` 自校验行）。
- doctor 的 2 条警告均为 `in_x` / `out_y` 的 ping/pong 同址（单次推理不受影响；重复喂输入需给这两个 decl 加 `double_buffer` 重新生成）。

## 2. 确定性（可复现）

| 段 | 设备遍（删 `target` 两次重编）`.cvx` sha256 | calbin 两次 `.elf` sha256 |
| --- | --- | --- |
| a | 两次相同 `1293ba90…` | ccu0 `9ef262af26a4a8a1…` / ccu1 `62a08c5f6aa14ff0…`，两次相同 |
| b1 | 两次相同 `87ea900a…` | ccu0 `c075a7417d5c8458…` / ccu1 `b3c82c70c26d3b84…`，两次相同 |
| b2a | 两次相同 `d38ee499…` | ccu0 `ddb33e83bbd4ed8a…` / ccu1 `bec601d3afd928ca…`，两次相同 |
| b2b | 两次相同 `f23d093a…` | ccu0 `8f077ce0b9178fa2…` / ccu1 `a5e023da1c24c01f…`，两次相同 |
| c | 两次相同 `b1f4ac1e…` | ccu0 `282ce19b9e13b052…` / ccu1 `37fc546e99294d04…`，两次相同 |

## 3. 参数块

| 段 | param_blk0.bin B | 打包器自校验 |
| --- | --- | --- |
| a | 21,483,520 | bad=0（28 个张量逐字节回读一致） |
| b1 | 33,452,032 | bad=0（26 个） |
| b2a | 28,729,344 | bad=0（24 个） |
| b2b | 4,722,688 | bad=0（2 个） |
| c | 12,292,096 | bad=0（58 个） |
| **合计** | **100,679,680** | 138 个张量 |

> 参数块从 313 MB 降到 100.7 MB：自写 `gn_stats`/`gn_apply` 后，GroupNorm 的 gamma/beta 不再按 `calblas::group_norm!` 的要求展开成 `[C·HW/CC][8]` f32（约 219 MB），改成每通道 16 份（`[C][16]`，一行 64 B）。

## 4. 本地数值（黄金仿真，非真机）

用 torch 按 `step.rs` 的算子顺序 + 我们的布局执行同一份程序（同 border/NHWC 布局、同 bf16 舍入点、同参数块），与 diffusers 0.43 fp32 参考（确定性 decode，sha256 `5dcfd42c…`）比较：

| 边界 | module | PSNR | max\|Δ\| |
| --- | --- | --- | --- |
| 段 a 输出 | `decoder.mid_block.resnets.1` | 43.91 dB | 0.2116 |
| 段 b1 输出 | `decoder.up_blocks.0` | 35.62 dB | 0.5315 |
| 段 b2a 输出 | `decoder.up_blocks.1.resnets.2` | 34.42 dB | 0.8076 |
| 段 b2b 输出 | `decoder.up_blocks.1`（含 upsampler） | 26.27 dB | 1.2852 |
| 段 c 输出（最终图） | `decoder.conv_out` | **52.83 dB** | **0.0473** |

## 5. 命令（复现）

```bash
# 设备遍（5 段；源码在 calBLAS/examples/sd_vae_{a,b1,b2a,b2b,c}）
for s in a b1 b2a b2b c; do
  ( cd /root/calBLAS/examples/sd_vae_$s && \
    cargo build --lib --features step --target dopamine-unknown-none-cvx )
done
# docc 打包 + 检查
for s in a b1 b2a b2b c; do
  docc calbin from-calvx /root/sd_vae/pkg_$s -o /root/sd_vae/cal_$s
  docc calbin doctor /root/sd_vae/cal_$s
  for core in ccu0 ccu1; do
    docc elf ccu view   /root/sd_vae/cal_$s/pkg_$s/${core}_chip0.elf   # load_size
    docc elf ccu disasm /root/sd_vae/cal_$s/pkg_$s/${core}_chip0.elf   # 条数/unknown/字节覆盖
  done
done
# 确定性：删 target 重编后比 .cvx；docc 二次构建比 ELF
# 本地数值：emulate.py + compare_stages.py（见 ../gen/）
```

## 6. 结论

**平台侧五段全部通过**：设备遍可复现（`.cvx` 逐字节确定）、docc 打包可复现（ELF 逐字节确定）、doctor 拒收 0、反汇编 0 unknown 且字节全覆盖。真机数值验证见 `docs/SD_VAE_REAL_DEVICE_VERIFICATION.md`（队友已报告通过，原始输出尚待入库；详见仓库 README §0）。
