# resize_bilinear2：Calculet NPU 双线性上采样算子（交付说明）

**日期**：2026-11-13
**真机验证**：见 [REAL_DEVICE_VERIFICATION.md](REAL_DEVICE_VERIFICATION.md)（含逐条命令、判定标准、排错表）
**状态**：设备遍编译 ✅ · docc calbin 打包 ✅ · calbin doctor ✅ · 构建确定性 ✅ · CCU ELF 反汇编 ✅ · CPU 参考 ✅ · **真机运行 ❌（本机无板卡/模拟器）**
**代码仓库**：`calculet-repos/calBLAS`（宏追加在 `src/kernels/vision.rs`，新增 `examples/resize_probe/`）
**补丁**：`_recon/work/resize/artifacts/resize_bilinear2.patch`（280 行，7 个文件，干净 diff）
**证据/产物**：`_recon/work/resize/artifacts/`（构建日志、calbin、ELF 反汇编、CPU 参考、输入/期望数据）

---

## 1. 选题与语义

**新算子：`resize_bilinear2`** —— NHWC、bf16、2× 双线性上采样：

```
输入  x: [H][W][C] bf16（行主序、紧排）
输出  y: [2H][2W][C] bf16
语义  = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
```

- 输出像素 `(oy, ox)` 的源坐标 `u = (ox+0.5)/2 - 0.5`，边缘 clamp 到 `[0, W-1]`（行方向同理）。
- 权重只有 `0.25 / 0.75` 两种（×2 在 align_corners=False 下退化为四邻域加权组合）。
- 中间按 **f32** 计算（每次 mul/add 各舍一次，RNE），最后一次性舍回 bf16；CPU 参考严格复刻同一次序。
- 约束：`C ≤ 128`（f32 中间 cache 一行 ≤ 512 B），`H, W ≥ 3`，尺寸全为编译期常量。

**为什么选它**（依据 `CALCULET_RECON.md` 的勘察）：
1. 仓库里 Nearest ×2 已有（`upsample2!`），**Bilinear 完全不存在**（三仓库 grep 零命中）；
2. SD/SAM/多尺寸输入的预处理与 VAE 上采样都需要它；
3. 纯取数 + 加权，无矩阵乘，48h 新手团队可控；
4. 复用 `upsample2!` 的 NHWC 视图/分核写法，不引入新 API。

---

## 2. 代码结构

| 文件 | 内容 |
| --- | --- |
| `calBLAS/src/kernels/vision.rs`（追加） | `#[macro_export] macro_rules! resize_bilinear2 { $vis $name: h=…, w=…, c=… }`，展开出一个 `#[npu::kernel(numerics="ks1")]` |
| `calBLAS/examples/resize_probe/Cargo.toml` | 独立 example crate（自带 `[workspace]`，不影响 calBLAS 主工作区） |
| `calBLAS/examples/resize_probe/src/kernels.rs` | 特化：`calblas::resize_bilinear2!(pub resize_bilinear2: h=32, w=32, c=16);` |
| `calBLAS/examples/resize_probe/src/step.rs` | `npu::decls! { in_x: input 32768 B, double_buffer; out_y: output 131072 B, double_buffer }` + `#[npu::program(chiplets=1, cores=1)] pub fn step()` |
| `calBLAS/examples/resize_probe/verify.sh` | 一键：编译 → `docc calbin from-calvx` → `doctor` → ELF 查看 |
| `_recon/work/resize/reference.py` | CPU 参考（numpy，严格复刻运算次序）+ 输入/期望数据生成 |
| `_recon/work/resize/resize_bilinear2_macro.rs` | 宏源码（与 vision.rs 中追加内容相同，便于审阅） |

**关键实现点（真实 API，全部来自仓库）**：

- 地址/视图：`Ptr::bind_strided::<Read/Write, 3>`、`View::sub`（与 `upsample2!` 相同写法）；
- 行内插值：`VCache::mul/add/copy`、`VCache::rows`、标量 `f32` 操作数（`Operand<f32>`）；
- 类型转换：`VCache::cvt(_, Cvt::default())`（bf16 ↔ f32，RNE）；
- 核数：`cores = 1` + 无分支三段展开（原因见 §3.2 的 docc 限制）；
- 写入：4 个交错的 2D 写视图（行 2iy+a、列 2j+b），行距 2C，起点 `a·2W·C + b·C`（等价 `upsample2!` 的 4 视图，去掉边框）。

> 设计上**没有用任何猜测 API**：所有原语调用都能在 `npu/src/lib.rs` 找到签名，写法与 `upsample2!`/`conv2d!` 一致。

---

## 3. 构建与打包（已在 WSL2 上真实执行）

### 3.1 命令

```bash
# 设备遍：一份 CalVX 程序（.cvx 汇编 + .cvxo 二进制）
cd ~/calBLAS
cargo build --manifest-path examples/resize_probe/Cargo.toml --lib --features step \
  --target dopamine-unknown-none-cvx
# => examples/resize_probe/target/dopamine-unknown-none-cvx/npu/resize_probe.cvx  (35,160 B)
# => .../resize_probe.cvxo (6,655 B)

# docc：CalVX ⇒ calbin（目录里放 step.cvx + 空的 param_blk0.bin）
rm -rf /tmp/resize_pkg /tmp/resize_cal && mkdir -p /tmp/resize_pkg
cp examples/resize_probe/target/dopamine-unknown-none-cvx/npu/resize_probe.cvx /tmp/resize_pkg/step.cvx
: > /tmp/resize_pkg/param_blk0.bin
docc calbin from-calvx /tmp/resize_pkg -o /tmp/resize_cal     # exit 0
docc calbin doctor /tmp/resize_cal                             # exit 0（拒收 0 条）
```

### 3.2 遇到的真实编译错误与修复（不是猜的）

| # | 真实报错 | 根因 | 修复 |
| --- | --- | --- | --- |
| 1 | `error[E0391]: cycle detected when checking if H is a trivial const` | 宏内层 `@pair` 又定义了 `const H = $h`，而调用处传的是外层 `H`（自引用） | `@pair` 调用改传原始宏字面量 `$h/$w/$c` |
| 2 | `docc calbin from-calvx`：`calvx-to-calt：… for 体里的 calvx.if：运行时条件、带结果的 if 还没有…纯标量的分支请写成 s.select` | kernel 里 `if iy == 0 { 0 } else { iy - 1 }` 生成 scf.if 带结果，docc 当前不支持 | 改为**无运行时分支**：第 0 行、`for iy in 1..H-1`（常量边界）、最后一行三段展开，行号 clamp 用常量/线性表达式 |

### 3.3 验证证据（全部 ✅ 本机实测）

| 验证项 | 结果 | 证据文件 |
| --- | --- | --- |
| 设备遍编译 | `Finished`，`.cvx` 35,160 B / `.cvxo` 6,655 B | `artifacts/build.log`、`.cvx`、`.cvxo` |
| docc calbin 打包 | `wrote /tmp/resize_cal`，exit 0 | `artifacts/from-calvx.log` |
| calbin doctor | **拒收 0 条**、解不开的稳态流 0 份；2 条 ping/pong 地址警告（见 §6） | `artifacts/doctor.log` |
| 产出 ELF | ccu0 19,560 B、ccu1 6,872 B；`docc elf ccu view`：`load_size=14608 = .text(14100)+.nops(104)+.rodata(404)` | `artifacts/ccu0_view.txt` |
| CCU 反汇编 | `2849 条：676 宽 / 10 custom / 2163 标量；0 条认不出；每个字节都被覆盖` | `artifacts/ccu0_chip0.elf.s`、`artifacts/ccu0_disasm.err` |
| 构建确定性 | 两次 `docc calbin from-calvx` 的 calbin **逐字节相同**，`docc calbin diff`：`一致：清单逐字段相同` | `artifacts/sha256_build1.txt`、`sha256_build2.txt` |
| CPU 参考正确性 | kernel 次序的 f32 结果 vs 独立 float64 数学参照：全部 65,536 个输出**bf16 舍入 100% 一致**，max偏差 0.996 ULP | `artifacts/reference.py`、`input_32x32x16_bf16.bin`、`expected_64x64x16_bf16.bin` |

生成的 CalVX 关键片段（节选，`artifacts/resize_probe.cvx`）：

```
.func resize_bilinear2(r0: ptr ddr, r1: ptr ddr)
    ...
    vec.load / q.cvt / vec.mul 0.75 / vec.mul 0.25 / vec.add / q.cvt / vec.store
    ...
.func step()
    mem.addr r0 = @in_x
    mem.addr r1 = @out_y
    call r0, r1, @resize_bilinear2
```

---

## 4. CPU 参考与测试数据

`_recon/work/resize/reference.py`（Windows Python 3.13 + numpy 2.5.3 已运行）：

```
input  : H=32 W=32 C=16 bf16, range [-1, 1]
output : 64x64x16 bf16
kernel-order f32 vs math f64: max|d| = 1.953e-03, mean|d| = 4.341e-04
max|d| / max(ulp) = 0.996 ULP(bf16)
exact-match ratio vs f64-derived rounding: 1.000000
```

生成的二进制（**bf16 原始位型**，可直接给 `dort calbin run --raw-in/--raw-out`）：

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `input_32x32x16_bf16.bin` | 32,768 B | 确定性随机输入（`np.random.default_rng(20261113)`，落 bf16 网格） |
| `expected_64x64x16_bf16.bin` | 131,072 B | 按 kernel 运算次序 + bf16 RNE 的期望输出 |

---

## 5. 真机运行步骤（未来有板时照抄）

```bash
# 板卡健康（calBLAS/AGENTS.md:91-101）
dort device -i 0 | grep "Dev  :"        # 要 OS up   RT up

# 跑我们的 calbin（与 yolo26 相同的 raw 输入/输出接口）
dort calbin run -i 0 -f /tmp/resize_cal -m resize_pkg \
  --raw-in input_32x32x16_bf16.bin --raw-out out.bin \
  --in-tensor in_x --out-tensor out_y

# 逐元素比较（bf16 位型可直接比；或转 f32 算 max diff）
python -c "
import numpy as np
a=np.fromfile('out.bin',dtype=np.uint16); b=np.fromfile('expected_64x64x16_bf16.bin',dtype=np.uint16)
print('elements',a.size,'exact match',(a==b).mean())
# bf16->f32 看最大误差
f=lambda x:((x.astype(np.uint32)<<16).view(np.float32))
d=np.abs(f(a)-f(b)); print('max|d|',d.max(),'mean|d|',d.mean())
"
```

预期：若设备执行与 CPU 参考同一次序，应逐位相同（只涉及 f32 mul/add/cvt，无 exp/rsqrt 等硬件近似）。

---

## 6. 已知限制（诚实清单）

1. **真机未验证**：本机无 NPU 板卡、无 `cvx` 模拟器（calvxus 不在三个赛题仓库里）。已完成的是**平台侧编译/打包/静态验证**，不是数值验证。
2. **单核**（`cores = 1`）：为了绕开 docc「for 体内带结果 if」限制并先拿到正确版本；性能未优化。
3. 只支持 **×2、NHWC、C ≤ 128**；任意尺寸/倍率、NCHW、双三次插值未实现。
4. **ping/pong 警告**：`mem.decl` 已标 `double_buffer`，但 docc 对本程序仍把 ping/pong 分到同一地址（doctor 通过、仅警告）；多次推理前需确认主机侧缓冲切换策略。
5. `docc elf kabi` 对本 kernel 报 `calt-bind` 错误（docc 自己的诊断说「多半是 docc 自己的缺陷」）；`docc calbin from-calvx` 路线不受影响，calbin 的 CCU ELF 已生成。
6. ~~未接入 calDNN 图/引擎表~~ **2026-11-14 已接入**：`ResizeDesc` + `Op::Resize` + `Graph::resize` + calBLAS `Resize` 引擎 + 探针引擎表，主机侧 `Plan::new` / `bind` 通过、5 项测试通过；但 calDNN 的计划发射路径官方尚未打通，真机仍走整程序路线。详见 [CALDNN_INTEGRATION.md](CALDNN_INTEGRATION.md)。

---

## 7. 下一步（按优先级）

1. **真机数值验证**（需要板卡）：按 §5 跑 `dort calbin run`，与 `expected_64x64x16_bf16.bin` 比对；若不一致，先用更小的尺寸（如 8×8×16）定位。
2. **cores = 2 版本**：按行分核；需要重新设计 clamp（当前靠三段展开，分核后边界行归属需固定）。
3. **通用尺寸**：把 `2×` 扩成 `H_out × W_out`（坐标/权重运行时计算，注意 docc 的标量控制流限制），或至少支持整数倍率 `SX/SY` 编译期参数。
4. ~~接入 calDNN~~ 已完成（2026-11-14），见 [CALDNN_INTEGRATION.md](CALDNN_INTEGRATION.md)。
5. **接入 CV 模型**：优先对接 YOLO26n 的预处理/上采样段或 SD 的 VAE 上采样段（真实权重 + 端到端对比）。
6. **性能**：行流水复用（相邻输出行共享输入行插值结果，可减少 2/3 的行内插值计算）。

---

## 8. 复现环境（本次已搭好）

- Windows 11 家庭版 + **WSL2（WSL 3.0.5，Ubuntu 24.04.3，kernel 6.18.58.1-microsoft-standard-WSL2）**
- rustup 1.29.3 + calculet 工具链（`rustc 1.98.1 (48a229cea 2026-09-01)`、`cargo 1.98.1`、`docc 0.4.1`、`dort 0.4.1`）
- 三仓库（`/root/{calBLAS,calDNN,npu}`，commit 见 `CALCULET_RECON.md` §3.2）
- 安装/修复脚本留档：`_recon/wsl/`（`env_setup.sh`、`install_wsl_msi.ps1`、`WSL_RESUME_PLAN.md`、日志）

---

## 9. 文件清单（本交付）

```
CALCULET_RECON.md                              # 勘察报告（§15/§16 有环境修复与最终结果）
RESIZE_OPERATOR.md                             # 本文件
calculet-repos/calBLAS/src/kernels/vision.rs   # 追加 resize_bilinear2! 宏
calculet-repos/calBLAS/examples/resize_probe/  # 独立 example（Cargo.toml, src/, README.md, verify.sh）
_recon/work/resize/
├── resize_bilinear2_macro.rs                  # 宏源码（审阅用）
├── reference.py                               # CPU 参考 + 数据生成
├── input_32x32x16_bf16.bin                    # 真机测试输入
├── expected_64x64x16_bf16.bin                 # 真机测试期望输出
├── apply_and_build.sh / finalize.sh / ...     # 本机执行的脚本（可复现）
└── artifacts/
    ├── resize_bilinear2.patch                 # 干净补丁（应用回 calBLAS）
    ├── resize_probe.cvx / .cvxo               # 设备遍产物
    ├── calbin/                                # docc calbin（含 2 个 CCU ELF）
    ├── build.log / from-calvx.log / doctor.log / kabi.log
    ├── ccu0_chip0.elf.s / ccu1_chip0.elf.s    # 反汇编
    ├── sha256_build1.txt / sha256_build2.txt  # 确定性对比
    └── reference.py / input/expected .bin
```

---

## 10. 通道分块版 `resize_bilinear2_tiled!`（2026-11-13 晚新增，支持大 C）

原版受限于“f32 中间 cache 一行 ≤512 B” ⇒ `C ≤ 128`。分块版把 C 切成 `NC = C/CC` 块顺序处理，
每块中间 cache 只占 `W × CC`，从而支持大通道数（SD/SAM 特征图常见 C=256/320/1280）。

- **宏**：`calculet-repos/calBLAS/src/kernels/vision.rs` 追加的 `resize_bilinear2_tiled!`
- **example**：`calculet-repos/calBLAS/examples/resize_probe_big/`（32×32×**256** → 64×64×256，`CC=64`）
- **约束**：`CC >= 32 && CC % 32 == 0 && C % CC == 0`（块偏移是 64 B 整数倍，docc 可证对齐）、
  `H, W >= 3`、`W*CC*56 B <= 384 KiB`（12 份 f32 + 4 份 bf16 cache）
- **结构**：`for cc in 0..NC`（常数轮数，同 `conv2d!` 的 nb 外层）内嵌“第 0 行 / `for iy in 1..H-1` / 最后一行”，仍无运行时条件分支。

### 本机验证（WSL2 + calculet 工具链）

| 项 | 结果 |
| --- | --- |
| 设备遍编译 | ✅ `resize_probe_big.cvx` 40,369 B / `.cvxo` 7,118 B |
| `docc calbin from-calvx` | ✅ exit 0（ccu0 86,568 B / ccu1 6,872 B） |
| `docc calbin doctor` | ✅ **拒收 0 条**（2 条 ping/pong 警告） |
| CCU 反汇编 | ✅ 15,997 条：4,177 宽 / 10 custom / 11,810 标量，**0 unknown**；`load_size 81,612` = 各节之和 |
| 构建确定性 | ✅ 两次 calbin 逐字节相同 |
| CPU 参考 | ✅ 与独立 float64 参照 **bf16 舍入 100% 一致**（max 1.953e-3，≤0.996 ULP） |
| 真机数值 | ❌ 未验证（无板/无模拟器） |

数据：`input_32x32x256_bf16.bin`（524,288 B）+ `expected_64x64x256_bf16.bin`（2,097,152 B），
真机命令同 §5（把 calbin 目录/子模型名/decl 名换成 `resize_big_cal` / `resize_big_pkg`）。

### 仍未做的（诚实清单）

1. **行流水复用**：每行仍重复计算 3 行的行内插值；已分析出两种实现（三 cache 轮换 / 循环体三重展开），
   但**无板无法数值验证**，故暂缓（详见 `OPTIMIZATION_CANDIDATES.md` §8.3）。
2. **cores=2**：当前仍单核；双核需要重新设计边界行归属与常量核号分支。
3. **W 分块**：`W*CC*56 B <= 384 KiB` 之外的宽度仍不支持；超大 W 需要横向分块（边界列单独处理）。
4. 任意尺寸/倍率、NCHW、进 calDNN 图（Engine 注册）仍是后续项。

## 11. 双核版 `resize_bilinear2_tiled2!`（2026-11-14 晚新增）

两核各算一半通道块，数值与分块版逐位相同，配三个计时用程序；编译 / docc / doctor 已过，**真机未跑、无性能数据**。见 [RESIZE_DUALCORE.md](RESIZE_DUALCORE.md)。
