# Resize 双核版 `resize_bilinear2_tiled2!`（2026-11-14 晚）

**状态**：设备端编译 ✅ · docc calbin ✅ · doctor 拒收 0 ✅ · CPU 分块分配检查 ✅ · **真机：与单核逐字节相同 ✅** · 墙钟比值 huge 1.10 / big 1.04（含固定开销）· **kernel 净加速比未测 ❌**

真机结果见 §8；§1–§7 写于上板之前，保持原样。

路线 A 之前交付的 Resize 是单核程序（`#[npu::program(cores = 1)]`）。这一版把通道分块版改成两核各算一半通道块，
目的只有一个：在真机上和单核版**同形状、同输入**对比，得到一个有实测支撑的加速比。
上板前本文件不出现加速数字；上板后的数字只在 §8，并带口径。

## 1. 做法

| 项 | 内容 |
| --- | --- |
| 宏 | [`kernels/resize_bilinear2_tiled2.rs`](../kernels/resize_bilinear2_tiled2.rs)（calBLAS 补丁中追加到 `src/kernels/vision.rs` 末尾） |
| kernel 属性 | `#[npu::kernel(cores = 2, numerics = "ks1")]`，程序 `#[npu::program(chiplets = 1, cores = 2)]`，入口里**两核都调** |
| 分工 | `NC = C/CC` 个通道块，核 r 处理 `r*NC/2 .. (r+1)*NC/2`；`base = core_rank()*PER; for k in 0..PER { xoff = (base+k)*CC; … }` |
| 计算体 | 直接复用 `resize_bilinear2_tiled!` 的 `@pair` / `@h` 分支，数值与单核分块版逐位相同 |
| 核间 | 各块读写的 DDR 区域互不重叠（输入只读，输出按 `cc*CC*2 B` 偏移分开），不需要 `sync_cores` |
| 新增约束 | `NC % 2 == 0`（每核整数个块），其余同分块版 |

为什么能确定要两核都调：VAE 审查里的探针实验证明 docc 把 `num_cores()` 折成程序核数、只由 0 号核调用分块模板会漏一半
（[REVIEW_SD_VAE_2026-11-14.md](REVIEW_SD_VAE_2026-11-14.md) §3）。这里同一机制反过来用：两核都调、各拿一半。

## 2. 三个新程序

| example | 形状 | cc | 核 | 用途 |
| --- | --- | --- | --- | --- |
| `resize_probe_big2` | 32×32×256 ⇒ 64×64×256 | 64（NC=4，每核 2 块） | 2 | 与已真机验证的 `resize_probe_big` 同形状，先验正确性 |
| `resize_probe_huge` | 128×128×256 ⇒ 256×256×256 | 32（NC=8） | 1 | 单核计时基线（输入 8 MiB、输出 32 MiB，kernel 时间才盖得过 dort 的固定开销） |
| `resize_probe_huge2` | 同上 | 32（每核 4 块） | 2 | 双核计时 |

`W*CC*56 = 128*32*56 = 229,376 B ≤ 384 KiB`，满足中间 cache 约束。

## 3. 平台侧证据（Linux calc，calculet 工具链，2026-11-14 20:10）

四个 example 全部 `cargo build … --target dopamine-unknown-none-cvx` rc=0、`docc calbin from-calvx` rc=0、`docc calbin doctor` **拒收 0 条、警告 2 条**（与以往相同的 ping/pong 两条）。
`.cvx` 中 `cores = 2` 出现在程序属性与 kernel 属性各一次、`core_rank` 查询 1 次。

| example | `.cvx` sha256 前缀 | ccu0 ELF | ccu1 ELF |
| --- | --- | --- | --- |
| `resize_probe_big`（旧，单核） | `5a22650a3f29335a`（**与已上板版本逐字节相同**） | 86,568 B | 6,872 B（空壳） |
| `resize_probe_big2` | `e13fba7920c6efd2` | 86,572 B | **86,620 B** |
| `resize_probe_huge` | `fc7f8dc099101e0d` | 13,356 B | 6,872 B（空壳） |
| `resize_probe_huge2` | `172a4ee9bf89a02d` | 13,356 B | **13,400 B** |

ccu1 ELF 从 6,872 B 的空壳长到与 ccu0 同量级，是两核都拿到 kernel 体的直接证据（与 VAE 修复时的现象一致）。
`resize_probe.cvx` 重编 sha 仍为 `536b406a6ec2512b`，旧的两个真机包不受这次改动影响。
完整哈希与 doctor 日志见 [`evidence/resize_dualcore/`](../evidence/resize_dualcore/)。

## 4. CPU 侧检查

- [`tests/test_resize_dualcore_split.py`](../tests/test_resize_dualcore_split.py)（3 项）：复刻宏的块分配公式，5 种形状下两核块集合不重不漏、各占一半；
  双核分块结果与单核分块地址仿真、与 kernel 次序参考逐位相同；并保留一个反例（只由 0 号核调用时只覆盖前一半块）。
- [`onboard/gen_resize_huge_case.py`](../onboard/gen_resize_huge_case.py)：固定种子生成 128×128×256 输入，期望值由三路算法交叉验证
  （kernel 次序 / 独立坐标公式 / 分块地址仿真）后写出，16,777,216 个输出元素一致。文件 40 MiB 不入库，板机现场生成（约 2.5 分钟）：

```
input.bin:    8,388,608 B  sha256 32ad08ef8e67489ac0cdd935ad762aee985d3ea3363d04e4d82307ccbf751f00
expected.bin: 33,554,432 B  sha256 5c1d4c42450b96ce276fc03bd6ca9735ce59d95d32a6eefcf844bc6f3bb46762
```

（numpy 版本不同时随机数可能不同，sha 对不上不影响有效性：期望值总是由同一份输入算出。）

## 5. 真机怎么跑（给上板的人）

```bash
# 开发机：重新打包（会多出 resize_big2 / resize_huge / resize_huge2 三个 calbin，旧的四个不变）
cd onboard && CALBLAS=~/bohack/calBLAS bash package_all.sh
# 生成大尺寸用例（任意有 python3+numpy 的机器，约 2.5 分钟）
python3 gen_resize_huge_case.py            # 写到 cases/resize_huge/
# 板卡主机：单核 vs 双核各跑 5 次，自动比对数值并汇总时间
DEVICE=0 N=5 bash run_resize_perf.sh       # 结果在 perf_out/timing.tsv
```

脚本做的判定：每个程序 N 次输出逐字节一致；`resize_big2` 与 `resize_big`、`resize_huge2` 与 `resize_huge` 输出**逐字节相同**；
与 CPU 期望 `--mode exact` 一致。

## 6. 报告规则

- 时间是 `dort calbin run` 整条命令的墙钟时间，**包含进程启动、模型部署、输入输出搬运**。只能报"同形状单核/双核的比值"，并注明含固定开销。
- 若 `resize_big` 对比看不出差别（2 MiB 输出，kernel 时间可能只有毫秒级），这是预期内的，用 `huge` 对的数字。
- 若双核**没有**更快，照实写。可能原因包括 DDR 带宽已被单核占满，这本身也是一个结论。
- 真机结果出来前，README、海报、PPT 里这一项一律写"双核版：编译通过、真机待测"。

## 7. 已知限制

1. 只分通道块，不分行；`C/CC` 必须是偶数，`C ≤ 128` 的小通道数场景（如 32×32×16）仍只有单核版。
2. 没有 kernel 内计时（`stamp()` 槽的读取方式未查到），拿不到净 kernel 时间。
3. 单核 `huge` 基线的 ccu0 ELF（13 KB）比 `big`（86 KB）小，说明 docc 对 NC=8 的外层循环没有完全展开；两版 huge 的代码形态一致，比较仍然公平，但与 `big` 之间不可横向比。

## 8. 真机结果（2026-11-14 晚，队友上板；报告 [`evidence/resize_dualcore_board_report.md`](../evidence/resize_dualcore_board_report.md)，原始件 [`evidence/board_2026-11-14/resize_dualcore/`](../evidence/board_2026-11-14/resize_dualcore/)）

环境：calculet-MS-A2，2 芯粒，Driver 0.9.3 / DRT 74832e51，dort 0.4.1；程序来源 `jax/resize-dualcore` @ `2ddef94`，板机重编的四个 `.cvx` 哈希与 §3 全部一致；`--board-lock external`，CalRT 流带未开启。

**数值（`--mode exact`）**

| 用例 | 元素 | 结果 | 输出 sha256 前 16 |
| --- | --- | --- | --- |
| resize_big（单核） / resize_big2（双核） | 1,048,576 | 都与 CPU 期望逐位相同；`cmp` single == dual | `6fb683176318acd3`（与第一次上板的单核输出同哈希） |
| resize_huge（单核） / resize_huge2（双核） | 16,777,216 | 同上 | `5c1d4c42450b96ce`（= §4 里 `expected.bin` 的哈希，说明板机现场生成的用例与本机逐字节一致） |

**墙钟（`dort calbin run` 整条命令，N=5，`timing.tsv`）**

| 用例 | 中位数 | 比值（单核/双核） |
| --- | --- | --- |
| resize_big / big2 | 0.006 s / 0.006 s | 1.04 |
| resize_huge / huge2 | 0.052 s / 0.047 s | 1.10 |

**怎么解读**

- 功能结论成立：双核版可以直接替换单核分块版，输出逐字节相同。
- 性能结论**不成立**：1.10 倍是含进程启动、部署、40 MiB 输入输出搬运的墙钟比，5 次样本的离散度（huge 0.048–0.053 s）和差值同量级。海报、PPT、README 一律写"真机逐位正确；墙钟比 1.10，含固定开销，不构成性能结论"。
- 报告里"两点模型估 kernel 约 10 ms → 5 ms、固定开销约 42 ms"是推算，而且与 `big` 整条命令只有 6 ms 自相矛盾（固定开销不可能大于总时间），**不要引用**。要净时间只能给 kernel 加 `stamp()` 插桩，或把形状放大到 kernel 时间明显盖过开销。

