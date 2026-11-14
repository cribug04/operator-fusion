# SD1.5 VAE 五段程序 · 平台侧证据（docc 0.4.1 / calculet 工具链）

日期：2026-11-14；环境：WSL2 Ubuntu 24.04.3 + `rustc 1.98.1-calculet` / `docc 0.4.1` / `dort 0.4.1`。
所有数字来自本机真实命令输出；命令在文末。

## 1. 每段程序

| 段 | `.cvx` B | `.cvxo` B | `.cvx` sha256（前 16） | 参数块 B | `@act` B | ccu0 ELF B | ccu0 load_size | ccu0 反汇编（条 / unknown） | ccu1 ELF B | ccu1 load_size | ccu1 反汇编（条 / unknown） | doctor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | 246,653 | 39,144 | `a5562bb7721383c8` | 21,483,520 | 35,241,984 | 1,414,604 | 1,408,852 | 255,016 / **0** | 1,049,156 | 1,043,636 | 183,206 / **0** | 拒收 0，警告 2 |
| b1 | 161,147 | 25,814 | `d73855e1ce461449` | 33,452,032 | 32,112,640 | 968,488 | 963,092 | 169,473 / **0** | 714,864 | 709,628 | 120,109 / **0** | 拒收 0，警告 2 |
| b2a | 71,793 | 11,909 | `997783b1fffe517a` | 28,729,344 | 53,592,064 | 398,836 | 393,608 | 72,656 / **0** | 145,552 | 140,448 | 23,194 / **0** | 拒收 0，警告 2 |
| b2b | 22,917 | 3,959 | `7ceb0215d9681845` | 4,722,688 | 69,222,400 | 31,112 | 26,072 | 4,574 / **0** | 29,820 | 24,852 | 4,359 / **0** | 拒收 0，警告 2 |
| c | 278,556 | 44,512 | `8ad397b0001dad32` | 12,292,096 | 650,330,112 | 1,527,952 | 1,521,480 | 278,824 / **0** | 959,116 | 953,084 | 165,977 / **0** | 拒收 0，警告 2 |
| 合计 | 780,066 | 125,338 | — | 96,279,680 | — | 4,340,992 | — | 780,543 / 0 | 2,898,508 | — | 496,845 / 0 | — |

- 反汇编全部“走完了：每个字节都被一条认得出的指令覆盖”，`0 条认不出`；每份 ELF 的各节字节和 = `load_size`（`docc elf ccu view` 自校验行）。
- doctor 的 2 条警告均为 `in_x` / `out_y` 的 ping/pong 同址（单次推理不受影响；重复喂输入需加 `double_buffer`）。

## 2. 确定性

| 段 | 设备遍两次（删 target 重编）`.cvx` sha256 | calbin 两次 `.elf` sha256 |
| --- | --- | --- |
| a | 同（`a5562bb7…`） | ccu0 `7a91b619…` / ccu1 `62a08c5f…`，两次相同 |
| b1 | 同（`d73855e1…`） | ccu0 `81022ffa…` / ccu1 `b3c82c70…`，两次相同 |
| b2a | 同（`997783b1…`） | ccu0 `ce1e7980…` / ccu1 `bec601d3…`，两次相同 |
| b2b | 同（`7ceb0215…`） | ccu0 `768954ee…` / ccu1 `a5e023da…`，两次相同 |
| c | 同（`8ad397b0…`） | ccu0 `b76375c9…` / ccu1 `37fc546e…`，两次相同 |

## 3. 参数块

| 段 | param_blk0.bin B | 打包器自校验 |
| --- | --- | --- |
| a | 21,483,520 | bad=0（28 个张量逐字节回读一致） |
| b1 | 33,452,032 | bad=0（26 个） |
| b2a | 28,729,344 | bad=0（24 个） |
| b2b | 4,722,688 | bad=0（2 个） |
| c | 12,292,096 | bad=0（58 个） |
| 合计 | **96,279,680** | 138 个张量 |

> 参数块从 313 MB 降到 96 MB 的原因：自写 `gn_stats`/`gn_apply` 后，GroupNorm 的 gamma/beta 不再需要按 `calblas::group_norm!` 的要求展开成 `[C·HW/CC][8]` f32（约 219 MB），改成每通道 16 份（`[C][16]`，一行 64 B）。

## 4. 命令（复现）

```bash
# 设备遍（5 段）
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
    docc elf ccu disasm /root/sd_vae/cal_$s/pkg_$s/${core}_chip0.elf   # 条数/unknown/覆盖
  done
done
# 确定性：删 target 重编后比 .cvx；docc 二次构建比 ELF
```

## 5. 结论

**平台侧五段全部通过**：设备遍可复现（`.cvx` 逐字节确定）、docc 打包可复现（ELF 逐字节确定）、doctor 拒收 0、反汇编 0 unknown 且字节全覆盖。真机数值验证见 `../docs/SD_VAE_ON_NPU.md` §7（等待板卡访问）。
