# SD1.5 VAE 解码器：calBLAS 视觉模板冒烟 + docc 0.4.1「21 位分支」bug 记录

> 平台侧（编译 / 打包级）验证。**没有 NPU 真机、没有 calvxus 模拟器** ⇒ 数值与性能未验证。
> 全部结论来自真实命令输出；源码在本文同目录 `smoke/`，日志在 `smoke/logs/`。
> 环境：WSL2 Ubuntu / root，`rustc 1.98.1-calculet`（toolchain channel=`calculet`）、`docc 0.4.1`、`cargo 1.98.1`；三仓库 `/root/{calBLAS,calDNN,npu}`。

---

## 0. 结论先行

### 0.1 冒烟程序（`smoke/example/`）

一份整步程序（`npu::decls!` + `#[npu::program(chiplets = 1, cores = 2)]`）里串起 16 个 kernel：
6 个视觉模板 + `fill16k` + `transpose2d` + 手写单头 flash 注意力 + 4 个手写搬运 kernel + 入口。

| 项 | 结果 |
|---|---|
| `cargo build --lib --features step --target dopamine-unknown-none-cvx` | **exit 0，0 warning** |
| 产物 | `target/dopamine-unknown-none-cvx/npu/sd_vae_smoke.cvx` **178,116 B**、`.cvxo` **29,058 B** |
| `docc calbin from-calvx` | **exit 0** |
| `docc calbin doctor` | **exit 0，拒收 0 条，警告 2 条**（`in_img`/`out_y` 的 ping=pong 同址；单次推理无影响） |
| `param_blk0.bin` | **10,014,720 B**（= `.cvx` 里 `@w` 的 `mem.decl`，必须同尺寸） |
| ccu0_chip0.elf | **1,088,452 B**，load_size **1,082,540**（.text 1,074,244 + .nops 104 + .rodata 8,192） |
| ccu1_chip0.elf | **725,776 B**，load_size **720,292**（.text 711,996 + .nops 104 + .rodata 8,192） |

形状覆盖：`conv2d 4(补)→512 k3s1 @64²`、`nhwc↔nchw + group_norm(512ch/32组/HW=4096)`、`conv 512→512 k3s1 @64²`、`1×1 conv 512→1536`、`upsample2 ×2 512ch（64²→128²）`、`conv 512→256 k3s1 @128²`、`T=4096/D=512 单头 flash attention`、`border NHWC ⇄ 紧排搬运（P=1536/512）`。

### 0.2 形状 → 模板可用性总表

| 形状 | 模板/写法 | 结论 |
|---|---|---|
| conv 4→512, k3s1, 64²，输入 NHWC 边框 | `conv2d!`，**`cin` 补到 16 即可**（`pin=(16,0)` 实测 docc 通过） | ✅ 可用（补到 16 省一半首层算力） |
| conv 4→512，补到 32 通道 | `conv2d!`，`cin=32, pin=(32,0)` | ✅ 可用（更保守，64 B 行距） |
| conv 512→512 k3s1 @64² | `conv2d!` | ✅ 可用（FAST 路，VB 225 KB） |
| conv 512→512 k3s1 @128² | `conv2d!` | ✅ 可用（**慢路**：FAST 条件 `(3·132+16)·512·2 = 421,888 > 384 KB`） |
| conv 1×1 512→1536 @64²（qkv 投影） | `conv2d!`，`k = 1, act = none` | ✅ 可用（1×1 只读内部） |
| conv 512→256 k3s1 @128² | `conv2d!` | ✅ 可用（慢路） |
| GroupNorm 512ch/32组/**HW=4096(64²)** | `group_norm!` | ✅ 可用 |
| GroupNorm 512ch/32组/**HW=12288** | `group_norm!` | ✅ 可用（.text 831,812 B，接近上限） |
| GroupNorm 512ch/32组/**HW=16384(128²)** | `group_norm!` | ❌ **docc 0.4.1 挂**（riscvc 21 位分支 bug，见 §4）。要么换 §4 的绕法，要么用 2 段手写 `gn_stats`/`gn_apply` |
| NHWC 边框 ⇄ NCHW ⅹ2（64²/128²，512 通道） | `nhwc_to_nchw!` / `nchw_to_nhwc!` | ✅ 可用（128² 在 stage 程序里也验证过） |
| upsample2 ×2，**c=512** @64²→128² | `upsample2!` | ✅ **文档写 c ≤ 256，但 c=512 实测 docc 通过**（本冒烟程序就在用）。保险起见也可拆两次 c=256（指针偏移 +256 通道） |
| k[4096,512] ⇒ kᵀ[512,4096] | `transpose2d!` | ✅ 可用（rows/cols 是 32 的倍数） |
| 边框清零（整块填常数） | `fill16k!` | ✅ 可用；**`n` 必须是运行期值**（常量循环体会被展开成 1 MiB 代码，同 §4 bug） |
| 单头 flash attention T=4096, D=512 | `attention!` ❌ / 手写 kernel ✅ | `attention!` 头维上限实测 **256**（见 §2.9）；D=512 必须手写（本冒烟里有一份可编译 + docc 通过的写法） |
| 单头 flash attention T=4096, **D=256** | `attention!` | ✅ 可用（doctor 拒收 0） |

---

## 1. 目录

```
smoke/
├── example/        # 冒烟整步程序（16 个 kernel 一份 .cvx）
│   ├── Cargo.toml  # 独立 [workspace]，features = ["step"]
│   └── src/{lib.rs, kernels.rs, step.rs}
├── attn_probe/     # attention! 头维探针：缺省 D=128；--features d256 / d512
│   ├── Cargo.toml
│   └── src/{lib.rs, kernels.rs, step.rs}
├── docc_bug_gn/    # docc 21 位分支 bug 的最小复现 + SRAM 绕法（同一 crate 两个 feature）
│   ├── Cargo.toml
│   └── src/{lib.rs, kernels.rs, step.rs}
└── logs/           # 全部命令的原始输出
```

放进 calBLAS 的位置：`/root/calBLAS/examples/sd_vae_smoke/{example→同级的 Cargo.toml/src…}`，`docc_bug_gn`、`attn_probe` 是它的子目录（各自独立 `[workspace]`）。
⚠ 本 example 的依赖路径按 `/root/calBLAS/examples/sd_vae_smoke/` 写（`calblas = ../..`、`caldnn = ../../../calDNN`、`npu = ../../../npu`；子目录多一层 `..`）。

---

## 2. 各模板的确切可用调用写法

全部实例化写法 = `calblas::宏名!(pub 函数名: 参数…)`，**生成的 Rust 函数名就是 `函数名`**（在 `step()` 里按这个名字调用；没有别名）。

公共约定（见 `/root/calBLAS/src/kernels/vision.rs` 顶部）：

- 卷积 / 池化 / 上采样：**NHWC bf16 + 四周 2 像素边框**，图 = `[H+4][W+4][P]`，`P` = 像素行距（通道数向上取整到 32 的倍数）；边框 = 0。
- `pin` / `pout`: `(P, 窗口偏移)`：`P` 是整图的通道行距，窗口偏移是像素内偏到第几个通道（编译期常数，0/16/…）。
  **指针实参本身必须给到 32 通道（64 B）对齐处**；kernel 的指针参数 docc 要求 64 B 对齐。
- NCHW 一侧（`group_norm!` / `nchw_to_nhwc!` / `nhwc_to_nchw!`）是**紧排** `[C][H·W]`、无边框。
- 卷积权重 bf16 `[k·k][Cin][Cout]`（BN 已折入），bias f32 `[Cout]`；`weights = ci_major` 时权重是 `[Cin][k·k][Cout]`。
- 每个 `count = 2` 的 kernel 调用后必须 `sync_cores()`；`cores = 1` 的 kernel 包在 `if core_rank() == 0 { … }` 里，后面也跟 `sync_cores()`。
  ⛔ 同一块地址既当 `cores = 2` kernel 的输入又当输出（原地）。

### 2.1 `conv2d!`

```rust
calblas::conv2d!(
    pub 名字:
    h = H, w = W, cin = CI, cout = CO, k = 3, s = 1,   // k ∈ {1,3}，s ∈ {1,2}；CI、CO 必须是 16 的倍数
    pin = (P_in, 0), pout = (P_out, 0),                // 行距（≥ 通道数，32 的倍数）；窗口偏移一般 0
    act = silu                                          // 或 act = none
    // 残差：再加 `, res = (P_res, 0)`（同形的图，act 之后加）
    // 权重布局：[Cin][k·k][Cout]：加 `, weights = ci_major`
);
```

**缓冲区大小**（主机侧分配/声明用）：

- 输入：`(H·s + 4) × (W·s + 4) × P_in × 2` B（**注意 s=2 时输入是 2 倍尺寸**）
- 输出：`(H + 4) × (W + 4) × P_out × 2` B
- 权重：`k·k × CI × CO × 2` B；bias：`CO × 4` B

**本冒烟里的四条确切写法**（`smoke/example/src/kernels.rs`）：

```rust
// conv_in：latent [64][64][4] ⇒ 512，k3 s1 silu，4 个真实通道补到 32（输入图 [68][68][32]）
calblas::conv2d!(pub vae_conv_in_64x64_4to512_k3s1_p32_512_silu: h = 64, w = 64, cin = 32, cout = 512, k = 3, s = 1, pin = (32, 0), pout = (512, 0), act = silu);
// 对照（补到 16，实测 docc 也通过；输入图 [68][68][16]，权重 9×16×512×2 = 147,456 B）
calblas::conv2d!(pub k_conv_p16_64_16to512_k3: h = 64, w = 64, cin = 16, cout = 512, k = 3, s = 1, pin = (16, 0), pout = (512, 0), act = silu);
// mid resnet 3×3：512 → 512 @64²（FAST 路）
calblas::conv2d!(pub vae_conv_mid_64x64_512to512_k3s1_silu: h = 64, w = 64, cin = 512, cout = 512, k = 3, s = 1, pin = (512, 0), pout = (512, 0), act = silu);
// 注意力 qkv 投影的融合形状：1×1 512 → 1536 @64²
calblas::conv2d!(pub vae_qkv_64x64_512to1536_k1s1_none: h = 64, w = 64, cin = 512, cout = 1536, k = 1, s = 1, pin = (512, 0), pout = (1536, 0), act = none);
// 上采样后的 3×3：512 → 256 @128²（慢路）
calblas::conv2d!(pub vae_conv_up_128x128_512to256_k3s1_silu: h = 128, w = 128, cin = 512, cout = 256, k = 3, s = 1, pin = (512, 0), pout = (256, 0), act = silu);
```

**结论 / 坑**：

1. **`cin` 只要补到 16**（模板断言 `CI % 16 == 0 && CO % 16 == 0`）。`pin=(16,0)` 我们在 docc 0.4.1 上实测**通过**（`load_size 13,252 B`、doctor 拒收 0）—— 之前按「64 B 行距」推断必须补到 32 是**多余的**；docc 接受 32 B 行距的输入，首层可以只补到 16。
2. `P_out` 取 32 的倍数；`pout` 的窗口偏移 `YO % 32 != 0` 时写回退化成一行一行写（能编，只是慢）。
3. `CO = 1536` 时内部按 `NC = 32` 分 48 个通道块；`Cout` 是 16 的倍数即可，不必 32。
4. FAST 路条件（模板内 `vision.rs:122`）：`s == 1 && CI ≤ 1024 && (k·(W+4) + 16)·CI·2 ≤ 384 KB`；128²×512 的 conv 放不下 ⇒ 走慢路（性能差异，影响整 VAE 的 128²/256² 层）。
5. 读数：`m.mma` 的 `M` 固定 16 行、`NC = Cols(CK,CO).nc()`（本仓已支持，调用方不用管）。

### 2.2 `group_norm!`

```rust
calblas::group_norm!(pub 名字: hw = H·W, c = C, groups = G, eps = 1.0e-06f32, act = silu);
```

- NCHW 紧排：x / y 各 `C·HW × 2` B。
- **w / b 不是每通道 1 个 f32**，而是主机展开成 **f32 `[C·HW/CC][8]`**：一行 8 个相同的值（32 B），行号 = 该通道在 `[C][HW]` 展平后 / CC；`CC` 由 `Cols::norm_cols(n, hw)` 定（hw ≥ 128 时取 128）。
  - 例：hw=4096、c=512、g=32 ⇒ `CC=128`、`R=CG·HW/CC=512`、`NH=norm_parts(R,CC)=4`、`RP=128`；w/b 各 `C·HW/CC = 16384` 行 × 8 × 4 B = **524,288 B**。
  - 例：hw=16384、c=512、g=32 ⇒ `R=2048`、`NH=16`、`RP=128`（**这个形状触发 docc bug**，见 §4）。
- 约束：`C % G == 0`、`HW % CC == 0`、`RP % 8 == 0`（模板里有 const assert）。
- kernel 是 **`cores = 2`**：组按核分（`halves!(G)`），两核都调 + `sync_cores()`；⛔ 原地（x、y 别用同一地址）。
- 本冒烟里的确切写法：

```rust
calblas::group_norm!(pub vae_gn_512ch_32groups_hw4096_silu: hw = 4096, c = 512, groups = 32, eps = 1.0e-06f32, act = silu);
```

### 2.3 `nhwc_to_nchw!` / `nchw_to_nhwc!`

```rust
calblas::nhwc_to_nchw!(pub 名字: h = 64, w = 64, c = 512, p = 512);   // 边框 NHWC 内部 ⇒ NCHW [C][HW] 紧排
calblas::nchw_to_nhwc!(pub 名字: h = 64, w = 64, c = 512, p = 512);   // 反向
```

- 约束（模板 const assert）：`C % 32 == 0`、`P % 32 == 0`、`C ≤ P`、`(H·W) % 32 == 0`、`W % 32 == 0 || 32 % W == 0`。
- **只写/只读内部**：NHWC 一侧的边框不会被写 ⇒ 目标 NHWC 缓冲**先 `fill16k!` 清零**；源 NHWC 的边框不读。
- NCHW 一侧是 `[C][H·W]` 紧排（无 P 行距）。
- kernel 没有 `cores` 属性（默认 1）：包在 `if core_rank() == 0` 里调用（内部 `halves!` 按该 kernel 的核数分块；这是推断，未数值验证 —— 见 §6 的未验证项）。

本冒烟写法：

```rust
calblas::nhwc_to_nchw!(pub vae_nhwc2nchw_64x64_512_p512: h = 64, w = 64, c = 512, p = 512);
calblas::nchw_to_nhwc!(pub vae_nchw2nhwc_64x64_512_p512: h = 64, w = 64, c = 512, p = 512);
```

### 2.4 `upsample2!`（最近邻 ×2）

```rust
calblas::upsample2!(pub 名字: h = 64, w = 64, c = 512, pin = 512, pout = 512);
```

- ⚠ `pin` / `pout` 是**标量行距**（不是 tuple！与 `conv2d!` 不同），一般都给整图通道数。
- 输入 `[H+4][W+4][P_in]`、输出 `[2H+4][2W+4][P_out]`，只写内部；输出边框要预先清零 ⇒ 输出缓冲先 `fill16k!`。
- 函数没有 `cores` 属性（默认 1）⇒ 包在 `if core_rank() == 0` 里调用。
- **文档写 `c ≤ 256`，但 `c = 512` 实测 docc 通过**（本冒烟程序里就是 512 通道，from-calvx/doctor 全过）。若生成器想保守，可拆两次 `c = 256`：两次调用分别把 x/y 指针偏移 +256 通道（`+512` 字节，64 B 对齐），`pin/pout` 仍给整图行距 512。
- 缓冲：输入 `(H+4)·(W+4)·P_in·2` B，输出 `(2H+4)·(2W+4)·P_out·2` B。64²×512 ⇒ 输出 132×132×512×2 = **17,842,176 B**。

### 2.5 `transpose2d!`

```rust
calblas::transpose2d!(pub 名字: rows = 4096, cols = 512);   // x [R][C] ⇒ y [C][R]，都紧排 bf16
```

- 约束：`R % 32 == 0 && C % 32 == 0`。
- 用途：把 k `[T, D]` 转成 kᵀ `[D, T]` 给手写注意力（每块 Kᵀ 是 `[D_chunk, 128]` 连续行）。
- 默认 1 核 ⇒ 包在 `if core_rank() == 0` 里调用；写完 `sync_cores()` 再给两核的注意力读。

### 2.6 `fill16k!`（整块填常数，多用边框清零）

```rust
calblas::fill16k!(pub 名字: value = 0.0f32);
// 调用：名字(ptr, n)   —— n 是「多少个 16 KiB 块」，运行期 usize 标量（index 寄存器）
```

- 每块 = 32 行 × 512 B = 16,384 B bf16，全填同一个值；**缓冲字节数必须整除 16,384**。
- ⚠ **`n` 必须是运行期值**：如果写成常量循环体，docc 会把循环展开成约 1 MiB 代码，撞上 §4 的 riscvc 21 位分支 bug。本冒烟里 `n` 直接传字面量也**没问题**（它作为 kernel 标量实参进寄存器，不是常量循环），但**不要**在 `fill16k!` 模板里把 `n` 改成 `const`。
- 块数 = `缓冲字节 / 16384`。本项目用到：4,734,976 B ⇒ 289 块；17,842,176 B ⇒ 1089 块；14,204,928 B ⇒ 867 块。
- 默认 1 核 ⇒ 包在 `if core_rank() == 0` 里；跟 `sync_cores()`。

### 2.7 手写单头 flash 注意力（T=4096、D=512，任务 C）

`attention!` 模板头维放不下（§2.9），所以在 example 里手写了一个：

```rust
#[npu::kernel(cores = 2, numerics = "ks1")]
pub fn vae_attn_4096x512(q: Ptr<Ddr,bf16>, kt: Ptr<Ddr,bf16>, v: Ptr<Ddr,bf16>, o: Ptr<Ddr,bf16>)
```

- 布局：q / v / o **紧排 `[T, D]` bf16**（行距 D）；kT 是 `transpose2d!` 转好的 `[D, T]`（行距 T）。q 块按 `halves!(T/16)` 分两核（每核 128 块）。
- 结构（**这是 docc 能过、也是行为正确的关键**）：
  1. **第一遍**（只算分数取行最大）：对 key 块 `kb∈[0,32)` 每块 128 key：`sacc[16,128]` f32 清零，头维 4 块 `dk∈[0,4)` 各 128 维做 `mma(lq[16,128], rk[128,128])` 累加；`sb = bf16(sacc)·1/√512`，行最大累加到 `m`。
  2. **第二遍 a**（只算 p 和行和）：重算分数 → `p = exp(sb − m)`（bf16 `exp_approx`）、`l += 行和`，并把 p **全部存进 VB** 的 `pall[512,128]` bf16（128 KB，VB 384 KB 够）。
  3. **第二遍 b**（只有 4 个输出累加器）：对 `kb∈[0,32)`：`lp.load_v(pall.rows(kb*16,16))`，4 个 D 块各 `rv[128,128]` + `acc_d.mma(lp, rv)`；循环结束 `o[:, 块] = acc / l`（bf16）按 `[16, 128]` 连续行写回。
- **⛔ 硬约束（踩过的坑，docc `calt-fate` 拒）**：**不要把「分数累加串」和「输出累加器串」放进同一个循环**。docc 报：
  `calt-fate：calt.mma 要从 GOB 里接着上一条累加（目的块 [16, 128] F32），但队头不是它的串：它在队里排第 1（从 0 数起），队里在飞的 2 串 … GOB 是 FIFO，init_acc 取走的是最早进去的那一串，起串、循环体、收尾里各串的 m.mma 次序要完全一致，否则两串的值互换，悄悄算错`
  实测：同一个第二遍循环里同时有分数链 + 4 个 acc（5 串）、分数链 + 2 个 acc（3 串）、甚至分数链 + 1 个 acc（2 串）**都会被拒**；把 p 落到 VB、把 PV 累加拆成「单独一段、只有 4 个固定次序 acc」（与 `conv2d!` FAST 路的 4 累加器同构）就过。
- 本写法编译 + docc 均通过（就是 §0.1 那份程序的一部分）；数值未验证。

### 2.8 手写搬运 kernel（border NHWC ⇄ 紧排）

`border2tight!` / `tight2border!` 两个本地宏（`smoke/example/src/kernels.rs`），形状 H=W=64、P=1536/512 各一个实例：

- 视图：源/目的各自 `[H, W·P]`（行距分别是 `(W+4)·P` / `W·P`），每行 `W·P` 个元素连续；按 512 B（256 个 bf16）一块 `vcache.load` → `store`。
- 一次搬一整行像素：`CC=256`（512 B = vcache 一行上限），每行 `W·P/CC` 块。
- `tight2border` 的目标边框由调用方先 `fill16k!` 清零。
- 单核（默认 1 → `if core_rank() == 0`）；docc 通过（在 §0.1 的程序里）。

### 2.9 `attention!` 模板的头维上限（实证数字）

`calblas::attention!` 的 rhs / acc 是 `mcache::<bf16>([D, KB])` / `mcache::<f32>([16, D])`（KB=128），D 一大就放不进 GIB1（docc 报的 GIB1 容量是 **65,536 B**）。用 `smoke/attn_probe/` 实测（单头、T=4096、`mask = keys` 全 0 掩码）：

| 头维 D | 设备遍 | `docc calbin from-calvx` | 结果 |
|---|---|---|---|
| 128 | pass | **exit 0** | ccu0/ccu1 ELF 各 157,348 B，load_size 152,280，.text 149,452；doctor 拒收 0 / 警告 5 |
| 256 | pass | **exit 0** | ELF 各 153,948 B，load_size 148,864，.text 146,780；doctor 拒收 0 / 警告 10（**上限比文档以为的 128 大一档**） |
| 512 | pass | **exit 1** | `error: CalVX ⇒ ELF：出 ELF 失败：calt-place：gib1 放不下：要到 131072 字节，只有 65536（Gib(1) [8, 512, 16] 行距 [16384, 32, 2] Bf16 @0x0）` |

- **规则**：`D × KB × 2 = D × 256 B ≤ 65,536` ⇒ **D ≤ 256**（D=256 刚好 65,536 B 通过）。
- 所以 SD VAE 的 mid attention（D=512、单头、T=4096）**不能直接用 `attention!`**，要用手写 kernel（§2.7），或者拆成「4 个头 D=128」——但那会改变语义（softmax 不是按 128 维分头做的），只能走 §2.7。

---

## 3. 冒烟程序的布局与尺寸（可直接照抄）

`smoke/example/src/step.rs`：

```rust
npu::decls! {
    in_img:  input,     bytes = 295936;      // [68][68][32] bf16（4 真实通道 + 28 个 0）
    out_y:   output,    bytes = 8921088;     // conv 512→256 输出 NHWC [132][132][256]
    w:       weights,   bytes = 10014720;    // 卷积权重 + bias + GN 展开的 w/b
    act:     workspace, bytes = 111329280;
}
#[npu::program(chiplets = 1, cores = 2)]
pub fn step() { … }
```

`@w` 偏移（每块 4096 对齐；全部按字节）：

| 偏移 | 内容 | 大小 |
|---|---|---|
| 0 | conv_in 权重 `[9][32][512]` bf16 | 294,912 |
| 294,912 | conv_in bias f32 `[512]` | 2,048 |
| 299,008 | GroupNorm w f32 `[16384][8]` | 524,288 |
| 823,296 | GroupNorm b f32 `[16384][8]` | 524,288 |
| 1,347,584 | conv_mid 权重 `[9][512][512]` | 4,718,592 |
| 6,066,176 | conv_mid bias | 2,048 |
| 6,070,272 | qkv 权重 `[1][512][1536]` | 1,572,864 |
| 7,643,136 | qkv bias `[1536]` | 6,144 |
| 7,651,328 | conv_up 权重 `[9][512][256]` | 2,359,296 |
| 10,010,624 | conv_up bias `[256]` | 1,024 |
| **10,014,720** | 合计 = `@w` 声明 | |

`@act` 偏移（每块 4096 对齐）：`A_H0=0`（conv_in 输出）`4,734,976`；`A_NCHW1=4,734,976`；`A_NCHW2=8,929,280`；`A_H1=13,123,584`；`A_H2=17,858,560`；`A_QKV=22,593,536`（14,204,928 B）；`A_UP=36,798,464`（17,842,176 B）；attn 的 `A_Q=54,640,640 / A_K / A_KT / A_V / A_O` 各 4,194,304 B；搬运的 `A_T1536=75,612,160`（12,582,912 B）`/A_T512/A_B1536/A_B512`；合计 111,329,280 B。

程序顺序：conv_in → NHWC→NCHW → GroupNorm → NCHW→NHWC → conv_mid → qkv 1×1 → upsample2 → conv_up → transpose2d(k⇒kT) → 单头 attention → 搬运（border⇒tight P=1536/512）→ 搬运（tight⇒border P=1536/512，目标先清零）。

---

## 4. docc 0.4.1 / riscvc 0.2.17 的「21 位分支」bug（128² group_norm）

### 4.1 现象（报错原文）

对 128² 的 group_norm（`hw = 16384, c = 512, groups = 32`）单独一次调用，`docc calbin from-calvx` 失败：

```
error: CalVX ⇒ ELF：出 ELF 失败：riscvc：链接：main0.o：.text 处到 `.LBB1_1` 的距离 -1117504 超出 21 位（类型 17）
```

（最小复现里是 `-1115512`；不同程序略有 ±1 KB 抖动，量级恒定 ≈ 1.1 MB。）

- RISC-V 的 21 位有符号分支范围 = ±1,048,575 B；riscvc 没有插跳板，直接超范围报错。
- **不是 `.cvx` 太大**：最小复现的 `.cvx` 只有 66 KB 左右；是 **docc 把常量上界的循环全展开**后**单个 calvx.func 的 .text** 太大。
- `docc calbin doctor`、`--no-params`、`--ddr/--sram/--entries` 都绕不过（这个错发生在出 ELF 阶段）；docc 二进制里没有 unroll/relax 的环境变量或选项。

### 4.2 触发条件：`group_norm!` 的 `for part in 0..NH` 被全展开

模板里 `NH = Cols::norm_parts(R, CC)` 是**编译期常量**，docc 把「三遍 × NH 段」整段展开。单函数 `.text` 实测（同一个最小 step，只换 hw）：

| hw | R = (C/G)·HW/CC | NH | ccu0 `.text` | 结果 |
|---|---|---|---|---|
| 4,096 (64²) | 512 | 4 | **282,484 B** | ✅ from-calvx exit 0 |
| 8,192 | 1,024 | 8 | **560,716 B** | ✅ exit 0 |
| 12,288 | 1,536 | 8 | **831,812 B** | ✅ exit 0 |
| 16,384 (128²) | 2,048 | 16 | **≈1,116,476 B**（= 报错距离） | ❌ exit 1 |
| 65,536 (256²) 推算 | 8,192 | 64 | ≈4.4 MB | ❌（同理必挂） |

- 拟合：`.text ≈ 545 B × R`（282,484/512 = 551.7；560,716/1024 = 547.6；831,812/1536 = 541.5；1,116,476/2048 = 545.2）。
- 推论：**`R > ~1,900` 就危险**；即 `(C/G)·HW/128 > 1900`。对本 VAE（C/G = 16、CC = 128）：**HW > 15,360 的 GroupNorm 都会挂**，所以 128² 起就必须换写法。
- `act = none` 也一样挂（展开的是循环体，不是 silu）。

### 4.3 最小复现（`smoke/docc_bug_gn/`）

```rust
// kernels.rs
calblas::group_norm!(pub k_gn_16384_512_32_silu: hw = 16384, c = 512, groups = 32, eps = 1.0e-06f32, act = silu);
// step.rs
npu::decls! { x: input, bytes = 16777216; y: output, bytes = 16777216; w: weights, bytes = 1048576; }
#[npu::program(chiplets = 1, cores = 2)]
pub fn step() { k_gn_16384_512_32_silu(x().cast(), w().at(0).cast(), w().at(524288).cast(), y().cast()); }
```

```bash
cd /root/calBLAS/examples/sd_vae_smoke/docc_bug_gn
cargo build --lib --features step --target dopamine-unknown-none-cvx     # exit 0
rm -rf /tmp/p /tmp/c && mkdir -p /tmp/p
cp target/dopamine-unknown-none-cvx/npu/docc_bug_gn.cvx /tmp/p/step.cvx
truncate -s 1048576 /tmp/p/param_blk0.bin
docc calbin from-calvx /tmp/p -o /tmp/c
# => exit 1：riscvc：链接：main0.o：.text 处到 `.LBB1_1` 的距离 -1115512 超出 21 位（类型 17）
```

### 4.4 绕法 A（已实测通过）：把 NH 做成**运行期**值 —— 从 SRAM 读

**关键**：docc 会内联内核并做常量传播，所以「普通标量实参」和「`num_cores()` 现场算」都**无效**（会把上界折回常量 16 再展开，报错距离与展开版逐字节相同）；必须让上界**不可常量传播**——从 SRAM 读：

```rust
#[npu::kernel(cores = 2, numerics = "ks1")]
pub fn gn(..., nh: npu::Ptr<npu::Ram, i32>) {          // ⚠ 用 i32；u32 会报
    let nh = nh.load() as usize;                        //   "failed to legalize unresolved materialization"
    // 其余与 group_norm! 完全一样：const RP = R/NH 不变（缓存形状仍是编译期），
    // 只把三处 `for part in 0..NH` 换成 `for part in 0..nh`
}
```

调用侧：程序加 `l2: ram, bytes = 64;`，GN 调用最后传 `l2().at(0).cast()`；主机把 `NH`（本形状 = 16，即 `R/RP`）写进该 SRAM 槽（或从 input 用 `copy_i32!` 抄进去，规避「host 只写 input」的限制）。

实测（`smoke/docc_bug_gn --features step,sram`）：

| 项 | 结果 |
|---|---|
| `cargo build` | exit 0（0 warning） |
| `docc calbin from-calvx` | **exit 0** |
| `docc calbin doctor` | **拒收 0 条**，警告 2 |
| ccu0 / ccu1 ELF | 各 **33,940 B**（原来是 ~1.12 MB 的文本），load_size **28,976**，`.text` **27,456**（缩小 **40×**），`.rodata` 1,416 |
| 反汇编 | 5,216 条，**0 条 unknown**，字节全覆盖 |

**同一个绕法用到完整 stage b2**（6 个 128² GN + 全部其它 op，`/root/calBLAS/examples/sd_vae_b2_bisect/` 里做过）：`from-calvx` exit 0、`doctor` 拒收 0；ccu0 ELF 359,056 B（load_size 353,768）、ccu1 ELF 104,840 B（load 99,716）。日志在 `smoke/logs/fullb2_workaround_*.log`。

> 结论：如果改模板（`/root/calBLAS/src/kernels/vision.rs` 的 `group_norm!`），最小改动 = 签名加 `nh: Ptr<Ram, i32>`，`for part in 0..nh`，`const RP = R/NH` 不动。数值逐位不变（累加/归约次序不变，只是循环不展开）。

### 4.5 绕法 B（最终采用）：手写 `gn_stats` + `gn_apply`

最终 SD VAE 生成器没有改 calBLAS 模板，而是把 group_norm 换成两个手写 kernel（都在段内用**运行期循环上界**）：

- `gn_stats`：按组求每通道 mean / rstd（两遍法，或 sum/sumsq），写进 workspace 的 `[C][2]` f32；
- `gn_apply`：读统计量做归一 + 仿射（+ silu），运行期循环上界。

代码与生成器：`_recon/work/sd_vae/gen/gen.py` 的 `emit_gn_stats` / `emit_gn_apply`。

### 4.6 试过但**无效**的绕法（省得再试）

| 尝试 | 结果 |
|---|---|
| `group_norm!` 的 NH 当普通标量实参传（`nh: usize`） | ❌ 报错距离与常量展开版**逐字节相同**（docc 内联 + 常量传播折回 16） |
| 在内核里写 `let nh = NH / 2 * num_cores();` | ❌ 同样被折（docc 知道 invocation `cores = 2`） |
| `Ptr<Ram, u32>` 读上界 | ❌ `failed to legalize unresolved materialization`（u32 不行，**i32 可以**） |
| `--no-params` | ❌ 仍在出 ELF 阶段挂 |
| `--ddr/--sram/--entries/...` | ❌ 只影响地址/入口，不影响代码生成 |
| docc 环境变量 / 隐藏选项 | ❌ 二进制里没有 unroll/relax 相关的开关 |

---

## 5. 精确命令（可复制复跑）

```bash
export PATH=/root/.cargo/bin:$PATH RUSTUP_HOME=/root/.rustup CARGO_HOME=/root/.cargo

# ① 设备遍（cwd 必须是 example 目录：registry 配置按 cwd 找）
cd /root/calBLAS/examples/sd_vae_smoke
cargo build --lib --features step --target dopamine-unknown-none-cvx
# => target/dopamine-unknown-none-cvx/npu/sd_vae_smoke.cvx（178,116 B）/ .cvxo（29,058 B）

# ② 打包 + docc（目录名 = calbin 里的子模型名）
rm -rf /tmp/smoke_pkg /tmp/smoke_cal && mkdir -p /tmp/smoke_pkg
cp target/dopamine-unknown-none-cvx/npu/sd_vae_smoke.cvx /tmp/smoke_pkg/step.cvx
truncate -s 10014720 /tmp/smoke_pkg/param_blk0.bin     # 必须 == .cvx 里 @w 声明
docc calbin from-calvx /tmp/smoke_pkg -o /tmp/smoke_cal   # exit 0
docc calbin doctor /tmp/smoke_cal                          # exit 0，拒收 0，警告 2

# ③ ELF / 反汇编覆盖
docc elf ccu view   /tmp/smoke_cal/smoke_pkg/ccu0_chip0.elf          # load_size / 各节
docc elf ccu disasm /tmp/smoke_cal/smoke_pkg/ccu0_chip0.elf --census  # 条数 / unknown / 覆盖面

# ④ 头维上限探针
cd /root/calBLAS/examples/sd_vae_smoke/attn_probe
cargo build --lib --features step,d256 --target dopamine-unknown-none-cvx   # D=256 ⇒ docc 过
cargo build --lib --features step,d512 --target dopamine-unknown-none-cvx   # D=512 ⇒ docc 报 GIB1

# ⑤ docc bug 最小复现 + 绕法
cd /root/calBLAS/examples/sd_vae_smoke/docc_bug_gn
cargo build --lib --features step       --target dopamine-unknown-none-cvx  # 复现（docc 挂）
cargo build --lib --features step,sram  --target dopamine-unknown-none-cvx  # 绕法（docc 过）
```

doctor 摘要（冒烟程序，原文见 `smoke/logs/smoke_doctor.log`）：

```
  权重块  param_blk0.bin 1 片，共 10014720 字节
     ccu   ccu0_chip0.elf         dram  chip ...  1088452 字节
     ccu   ccu1_chip0.elf         dram  chip ...  725776 字节
  文件    9 个，共 11830037 字节
  检查    ⛔ 拒收 0 条  ⚠ 警告 2 条  ⛔ 解不开的稳态流 0 份
   ⚠  tensor `in_img` ... ping and pong the same address ...（out_y 同）
```

---

## 6. 给「整 VAE 生成器」的注意事项

1. **缓冲区对齐**：`npu::decls!` 的每个 decl 基址由运行时对齐（本仓按 4 KiB 观察），但**指针参数要 64 B 对齐**；所有 `act().at(offset)` 的偏移都取 4096 的倍数最稳（本冒烟全部如此），至少也要 64 B。
2. **参数指针**：`@w` 里各权重/bf16 与 bias/f32 的偏移 64 B 对齐（本冒烟 4096）。`rc`：设备端 `usize` 只有 32 位，**权重偏移 > 4 GiB 要用 `Ptr::at(u64)`**（不是 `add`）。
3. **`param_blk0.bin` 尺寸必须等于 `.cvx` 里 `@w` 的 `mem.decl`**（`docc calbin from-calvx` 会校）；只做结构检查可用 `--no-params`。
4. **conv 首层**：4 个真实通道补到 **16** 就够（`cin=16, pin=(16,0)` 实测 docc 通过）；补到 32 只是行距更保守。第一层权重 `[9][16][512]`（其余 12 个输入通道的权重给 0）。
5. **GroupNorm**：128² 及以上**不能用现成 `group_norm!`**（§4）；用 `gn_stats`/`gn_apply` 手写两段（运行期循环上界）。注意 w/b 的主机排布口径（`[C·HW/CC][8]`）只对模板成立；自写 kernel 请只传每通道 `[C]` 的 f32。
6. **单核 / 双核**：整步程序 `#[npu::program(chiplets = 1, cores = 2)]`。`cores = 2` 的 kernel（`conv2d!`、`group_norm!`）两核都调 + `sync_cores()`；模板里没有 `cores` 属性的（`nhwc_to_nchw!`、`nchw_to_nhwc!`、`upsample2!`、`transpose2d!`、`fill16k!`）是 1 核，包在 `if core_rank() == 0` 里 + `sync_cores()`。
7. **⛔ 原地运算**：`cores = 2` 的 kernel 不能输入=输出（两核各写一半、读全量会撕裂）。`group_norm` 的 x/y、conv 的 x/y 都必须分开。
8. **布局转换**：`nchw_to_nhwc!` / `upsample2!` 只写内部 ⇒ 目标 NHWC 缓冲**预清零**（`fill16k!`）。`nhwc_to_nchw!` 只读内部，不用清。
9. **注意力**：D=512 单头别用 `attention!`（§2.9，上限 256）；用手写 kernel 时**分数链与输出累加器链必须分循环**（§2.7），这是 docc `calt-fate` 的硬约束，不是可选优化。
10. **搬运**：border NHWC ⇄ 紧排用 §2.8 的写法（512 B 一块、一整行一次搬）；别用「[1, 大列] 视图」，vcache 一行上限 512 B。
11. **未验证项（本机没有真机/模拟器）**：
    - 数值、性能全未验证；
    - `cores = 1` 的 kernel 在 `cores = 2` 程序里调用时，其内部 `halves!` 用的是该 kernel 的核数还是整个 invocation 的核数，**没有数值手段确认**（按 docc 的「cores = 1 的 kernel 只在 ccu0」推断前者，但没验证）——这是整 VAE 里 `upsample2`/布局转换的实际风险点，建议上板前用一个「两核同步写回一行已知数据再读回」的探针先确认；
    - docc 的 `aligned_args` 前置条件（主机发射前要核这些实参是 64 的倍数）本冒烟全由常量满足，真机上要照 `kabi.json` 核。
12. **性能提示（未实测）**：128²×512 的 conv 因 VB 384 KB 限制走慢路（`fast = false`）；单头 attention 在 docc 上能过，但它的 PV 累加每 key 块要读 4 次 V（每次 128×128），且 D 方向分块导致 mma 数 ×4；`vae_attn` 的分数部分算了两遍（两遍 softmax 结构）。

---

## 7. 复跑清单 / 日志索引

| 日志 | 内容 |
|---|---|
| `logs/smoke_build.log` | 冒烟设备遍：exit 0，0 warning |
| `logs/smoke_fromcalvx.log` | `docc calbin from-calvx`：exit 0 |
| `logs/smoke_doctor.log` | `doctor`：拒收 0 / 警告 2 / ELF 尺寸 |
| `logs/hw{4096,8192,12288}_fromcalvx.log` | docc 21 位 bug 的通过档（.text 282,484 / 560,716 / 831,812） |
| `logs/hw16384_fromcalvx_FAIL.log` | 失败档原文（距离 -1,116,476 超出 21 位） |
| `logs/bug_gn_build.log` / `logs/bug_gn_fromcalvx_FAIL.log` | 最小复现：build 0 / docc 1（`-1115512`） |
| `logs/bug_gn_sram_build.log` / `logs/bug_gn_sram_fromcalvx.log` | 绕法 A：build 0 / docc 0 / .text 27,456 |
| `logs/fullb2_workaround_*.log` | 绕法 A 用到完整 stage b2：from-calvx 0、doctor 拒收 0 |
| `logs/attn_d128_fromcalvx.log` / `attn_d128_doctor.log` | `attention!` D=128：docc 0、doctor 0/警告 5 |
| `logs/attn_d256_build.log` / `attn_d256_fromcalvx.log` | D=256：build 0、docc 0 |
| `logs/attn_d512_build.log` / `attn_d512_fromcalvx_FAIL.log` | D=512：build 0、docc 1（GIB1 放不下） |
