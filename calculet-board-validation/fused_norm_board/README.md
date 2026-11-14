# 题一：融合残差加法与 RMSNorm（`add_rms_norm`）

把 `s = a + b`（残差，后面几层还要用）与 `y = rms_norm(s) · w` 合成一个 kernel：`s` 加完留在 vcache 里直接用，
不再走一趟 DDR 读回。

## 改了什么

| 位置 | 内容 |
| --- | --- |
| `src/kernels/norm.rs` | 新增 `calblas::add_rms_norm!` 模板；逐句复用 `rms_norm!` 的归一步骤，只在前面多读一个输入、加一次、写一次残差 |
| `src/engines.rs` | 新增 `FusedNormDesc` 与融合引擎 `AddRmsNorm`：一条 `Step { covers: 2 }` 认下图里**相邻**的 `Pointwise(Add)` 与 `Norm(Rms)` |
| `examples/fused_norm/` | 特化（`rows = 4, n = 4096, eps = 1e-6`，对齐 `examples/qwen3_8b` 的多 token prefill）、引擎表、图、测试与演示 |

数值上融合与两步**逐步同序**：bf16 加一次、`sum → ×1/n → +eps → bf16 → rsqrt.approx → ×w` 的顺序完全一致，
因此预期两版设备输出逐位一致，需在真机上分别与 CPU 参考比较，再检验两版输出是否逐位一致。

## 跑法

```bash
cd calBLAS

# 设备遍：出 CalVX（.cvx 可读汇编 / .cvxo 目标文件）
cargo build -p fused_norm --lib --target dopamine-unknown-none-elf
ls target/dopamine-unknown-none-elf/npu/        # fused_norm.cvx、fused_norm.cvxo

# 主机遍：编排两种方案、打印作业与搬运量
cargo run -p fused_norm

# 测试
cargo test -p fused_norm
```

## 已验证的结果

设备遍产出的 `fused_add_rms_norm` 序列（`fused_norm.cvx`）：`vec.load a → vec.load b → vec.add → vec.store res →
vec.mul(s,s) → q.cvt f32 → vec.reduce ×2 → ×1/n → +eps → q.cvt bf16 → q.rsqrt.approx → vec.mul → vec.load w → vec.mul → vec.store y`。

主机遍（`cargo run -p fused_norm`，4×4096 bf16，每张矩阵 32 KiB）：

```
图：5 个张量、2 个运算（[4 × 4096] bf16，每张矩阵 32 KiB）

融合：1 个作业，工作区 DDR 32768 B、SRAM 0 B
  fused_add_rms_norm(%0, %1, %2, %3, %4)

两步：2 个作业，工作区 DDR 32768 B、SRAM 0 B
  residual_add(%0, %1, %3)
  hidden_rms_norm(%3, %2, %4)

一趟 DDR 搬运（不含主机读写）：融合 139264 B、两步 172032 B ⇒ 融合省 32768 B（19.0%）
```

测试 6 个全过：融合成 1 个作业且参数次序正确、不融合时是 2 个作业、eps 不匹配或权重缺失、长度错误时引擎拒绝融合、
归一化输入不是残差时不融合和图形状。

## 已知限制

- 特化表只有 `rows = 4, n = 4096, eps = 1e-6` 一档；换形状要再加一条（`add_rms_norm!` + `FusedNormDesc`）。
- 只认 `NormKind::Rms`、不带偏置；`Pointwise` 必须是 `Add`，且归一化直接吃残差那个中间张量。
- 残差仍写 DDR（后续层要用），省下的是归一化再读一遍的那 32 KB 和一次作业发射。
- 融合引擎必须排在 `Binary` / `RmsNorm` 之前，`FirstFit` 才先试它（见 `examples/fused_norm/src/engines.rs`）。

## 上板验收

两份 CalBin 均使用 1 个芯粒和 2 个核。现有 kernel 由 0 号核执行，两版用同一个程序入口和屏障。
输入和输出均为小端 BF16 连续数组。

| 张量 | 所属缓冲区 | 字节偏移 | 大小 |
| --- | --- | ---: | ---: |
| a | in_raw | 0 | 32768 |
| b | in_raw | 32768 | 32768 |
| w | in_raw | 65536 | 8192 |
| residual | out_raw | 0 | 32768 |
| y | out_raw | 32768 | 32768 |

在仓库根目录生成两份 CalBin 和测试输入。输出目录应是一个新目录。

```bash
export PATH=/root/.cargo/bin:$PATH
bash examples/fused_norm/pack.sh ./board
```

上板时进入产物目录，选择分配给团队的设备下标。随产物附带 `verify.py`。

```bash
device=0
dort device -i "$device"
for variant in fused split; do
    dort calbin run -i "$device" -f "$variant/cal" -m "$variant" \
      --in-tensor in_raw --out-tensor out_raw \
      --raw-in input.bin --raw-out "$variant.bin" || break
done
python3 verify.py check input.bin fused.bin split.bin
```

测试输入包含整行抵消、普通非零输入、全零输入和接近 eps 主导的低幅输入，权重包含正值、负值和零。
验收脚本对两版分别检查残差 BF16 字节和独立 CPU RMSNorm 公式，再检查两版设备输出是否逐字节一致。
CPU 公式在残差 BF16 舍入后用高精度计算，暂用相对容差 3.5% 或绝对容差 0.002。该阈值待现场核验，
用于覆盖设备的 BF16 中间舍入与近似 rsqrt。

### 性能记录

4×4096 形状下，算子张量读写的静态估算为融合 139264 B、两步 172032 B。两版均写出 32768 B 的残差。
主机图的内部工作区各 32768 B。上板程序直接将残差写到输出缓冲区，两版各使用 73728 B 输入及
65536 B 输出。Plan 报告的 SRAM 工作区为 0 B。VB 峰值和实际 DDR 事务量待现场核验。

现场用同一设备、相同输入和相同调用次数，先验正确性，再分别记录编译、加载、预热和稳态执行。
`dort calbin run --raw-in` 用于验证单次调用，整个命令的耗时含部署和主机 I/O。
稳态比较应在一次部署后重复推理，用板上计时记录重复次数、中位数和 P95。
