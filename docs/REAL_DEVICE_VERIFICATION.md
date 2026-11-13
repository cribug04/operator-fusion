# 真机验证操作手册（Calculet NPU：Resize 算子 + stride-2 卷积原型）

**版本**：2026-11-13 22:10
**适用对象**：拿到 Calculet NPU 板卡、需要在真机上验证本仓库算子的同学
**前置工具版本（本机实测）**：calculet 工具链 `rustc 1.98.1`、`docc 0.4.1`、`dort 0.4.1`；Python 3 + numpy（对比脚本用）
**一句话流程**：**编译（开发机）→ `docc` 打 calbin → 板卡健康检查 → `dort calbin run` 跑 → 拉回输出 → 和参考逐元素比 → 记录**。

---

## 0. 这份手册验证什么、不验证什么

| 验证对象 | 验证内容 | 通过标准（详见 §7） |
| --- | --- | --- |
| `resize_bilinear2`（双线性 ×2，32×32×16→64×64×16） | 真机输出 vs CPU 参考 | 逐位相同（允许极少数 1 ULP，见 §7.4） |
| `resize_bilinear2_tiled`（通道分块，32×32×256→64×64×256） | 真机输出 vs CPU 参考 | 同上 |
| `conv2d_s2_fast`（stride-2 FAST 原型，80×80×64→40×40×64） | ① vs 同形状慢路 `conv2d_s2_ref`；② vs CPU 参考 | ① **必须逐位相同**；② 宽松（cos ≥ 0.9999、≤4 ULP） |

**已完成的（不需要真机）**：Rust 设备遍编译、`docc calbin from-calvx`、`docc calbin doctor`、CCU ELF 反汇编 100% 可识别、两次构建逐字节相同、CPU 参考（numpy）。
**只有真机能做的**：① 数值正确性；② 性能；③ 反复运行的稳定性（多次推理/多 token）。

---

## 1. 你需要准备什么（逐项自检）

| # | 条件 | 自检命令 | 期望 |
| --- | --- | --- | --- |
| 1 | Linux x86_64 开发机（装工具链） | `uname -m` | `x86_64` |
| 2 | rustup + calculet 工具链 | `rustc +calculet --version` | `rustc 1.98.1 (…)` |
| 3 | cargo | `cargo --version` | `cargo 1.98.1` |
| 4 | docc / dort | `docc --version; dort --version` | `docc 0.4.1` / `dort 0.4.1` |
| 5 | 三个仓库同级 | `ls -d calBLAS calDNN npu` | 三个目录都在 |
| 6 | 板卡主机能打开设备 | `dort device -i 0` | 出现 `OS up   RT up` |
| 7 | 板卡主机有 Python3+numpy（或把输出拉回开发机比） | `python3 -c "import numpy; print(numpy.__version__)"` | 有版本号 |
| 8 | 有 `~/cal-pcie-driver/calculet_pci.ko`（只在需要恢复板卡时用） | `ls ~/cal-pcie-driver/calculet_pci.ko` | 文件存在 |

> **开发机 / 板卡主机可以是同一台**。若板子在另一台机器上：开发机负责 §§4 编译打包，板卡主机只放 `onboard_cal/`（calbin）+ `cases/`（输入数据）+ `compare_bf16.py`。

---

## 2. 材料清单（本工作区里已备好）

```
REAL_DEVICE_VERIFICATION.md                 ← 本手册
calculet-repos/calBLAS/                     ← 代码（vision.rs 已含 3 个新宏；4 个 example）
  src/kernels/vision.rs
  examples/resize_probe/                    ← 用例 1（bilinear ×2 小图）
  examples/resize_probe_big/                ← 用例 2（通道分块 C=256）
  examples/conv2d_s2_probe/                 ← 用例 3（stride-2 FAST 原型）
  examples/conv2d_s2_ref/                   ← 用例 3 的同形状慢路对照
_recon/work/onboard/                        ← 真机验证工具包
  package_all.sh                            ← 开发机一键：编译 4 个 example + docc 打包 + doctor
  run_on_board.sh                           ← 板卡主机一键：健康检查 + 4 用例 ×3 次 + 对比
  compare_bf16.py                           ← bf16 原始输出对比工具（需要 numpy）
  gen_conv_s2_case.py                       ← 重新生成卷积测试数据（可选；cases 已附带）
  cases/                                    ← 测试数据（已生成，sha256 见 §3.2）
  artifacts/onboard_kit.patch               ← 代码补丁（若你从干净仓库开始，用它应用）
```

文件清单/尺寸/名字的速查表见 **§10 附录 A**。

---

## 3. 测试数据（已生成，可直接用）

### 3.1 四个用例的输入/输出

| 用例 | 输入文件 | 输入大小 | 权重/参数 | 期望输出 | 输出大小 |
| --- | --- | --- | --- | --- | --- |
| `resize` | `cases/resize_small/input.bin`（32×32×16 bf16） | 32,768 B | 无（`param_blk0.bin` 空文件） | `cases/resize_small/expected.bin`（64×64×16） | 131,072 B |
| `resize_big` | `cases/resize_big/input.bin`（32×32×256） | 524,288 B | 无 | `cases/resize_big/expected.bin`（64×64×256） | 2,097,152 B |
| `conv_s2_fast` / `conv_s2_ref` | `cases/conv_s2/input.bin`（padded NHWC 164×164×64） | 3,442,688 B | `cases/conv_s2/param_blk0.bin`（9×64×64 bf16 权重 + 64 f32 bias） | `cases/conv_s2/expected.bin`（80×80×64） | 819,200 B |

> 说明：`resize` 的输入/期望是**位型文件**（bf16 uint16 小端）；卷积的期望输出是 80×80×64（kernel 输出 buffer 是 84×84×64，只有内部 80×80 被写，比较时裁掉外圈 2 像素）。

### 3.2 sha256 核对（分发时先验一下，避免拿错文件）

| 文件 | 字节数 | sha256 |
| --- | --- | --- |
| `cases/resize_small/input.bin` | 32,768 | `4d7f911771392b0158a22412eeb265772904d873dc5f5e38ac140e3ad9262898` |
| `cases/resize_small/expected.bin` | 131,072 | `725b91e13d946710c48eb194a11f7a0e25fcccde5df2d563cb2a585b90ba2cff` |
| `cases/resize_big/input.bin` | 524,288 | `4b2f77ff5b00b04eccf4061cd8504acef3add42adb326c1afddc055952e49bcf` |
| `cases/resize_big/expected.bin` | 2,097,152 | `6fb683176318acd3b0736307bec0490dc210e5a8696062cb974c22852f9e40ff` |
| `cases/conv_s2/input.bin` | 3,442,688 | `d16c0bd6455fa3a384a876c9d81a575d70421bf29a3a5535baba534d4b5a87a4` |
| `cases/conv_s2/param_blk0.bin` | 73,984 | `d0f273a96ea70ad6a5df7cbef2608a19aeb980cd815ff41e2edaa6afa26c2f55` |
| `cases/conv_s2/expected.bin` | 819,200 | `d0a092068fc050ea076c96fc309bf8a6de1e890106dfa4e5c475fd3eea8efbc2` |

### 3.3 想自己重新生成卷积数据（可选）

```bash
python3 -m pip install numpy                   # 若未装
python3 gen_conv_s2_case.py                    # 默认写到 ./conv_s2_case/
# 把生成物覆盖到 cases/conv_s2/（或直接用 gen 出来的目录）
cp conv_s2_case/input_164x164x64_bf16.bin cases/conv_s2/input.bin
cp conv_s2_case/param_blk0.bin            cases/conv_s2/param_blk0.bin
cp conv_s2_case/expected_80x80x64_bf16.bin cases/conv_s2/expected.bin
```

> 随机种子固定（`20261113+2`），重复生成结果一致；sha256 应与 §3.2 相同。

---

## 4. 编译 + docc 打包（在**开发机**上）

### 4.1 放置代码（二选一）

**方式 A（推荐）：用本工作区现成的 calBLAS** —— 直接把 `calculet-repos/` 整个拷到 Linux 上（或 `scp -r`）。

**方式 B：从干净 clone 开始，应用补丁**

```bash
# 三个仓库同级
for r in calBLAS calDNN npu; do git clone http://115.159.82.89/git/calculet/$r.git; done
cd calBLAS
git apply /path/to/onboard_kit.patch      # 补丁：8 个 example 文件 + vision.rs 的 3 个宏
git status --short                        # 应看到修改：src/kernels/vision.rs；新增 4 个 examples/
```

### 4.2 一键打包（推荐）

```bash
cd /path/to/onboard          # 即 _recon/work/onboard/
export CALBLAS=$HOME/calBLAS # 你的 calBLAS 仓库根目录
export CASE_DIR=$PWD/cases
export OUT=$PWD/onboard_cal
bash package_all.sh
```

脚本会做：4 次 `cargo build --target dopamine-unknown-none-cvx` → 4 次 `docc calbin from-calvx` → 4 次 `docc calbin doctor`，并打印所有真机命令。

**期望的最后输出**：

```
################ 打包完成 ################
calbin 目录：
  .../onboard_cal/resize
  .../onboard_cal/resize_big
  .../onboard_cal/conv_s2
  .../onboard_cal/conv_s2_ref
```

每个 doctor 日志（`onboard_cal/logs/doctor-*.log`）里应有：

```
检查    ⛔ 拒收 0 条  ⚠ 警告 2 条  ⛔ 解不开的稳态流 0 份
```

### 4.3 手动逐条（想看清每一步时）

```bash
cd $HOME/calBLAS
# ① 设备遍编译（每个 example 单独编译；工具链自动用 rust-toolchain.toml 的 calculet）
cargo build --manifest-path examples/resize_probe/Cargo.toml  --lib --features step --target dopamine-unknown-none-cvx
cargo build --manifest-path examples/resize_probe_big/Cargo.toml --lib --features step --target dopamine-unknown-none-cvx
cargo build --manifest-path examples/conv2d_s2_probe/Cargo.toml  --lib --features step --target dopamine-unknown-none-cvx
cargo build --manifest-path examples/conv2d_s2_ref/Cargo.toml    --lib --features step --target dopamine-unknown-none-cvx

# ② 准备 calbin 目录（子模型名 = 目录名）：
OUT=/tmp/onboard_cal; mkdir -p $OUT/pkg/{resize_pkg,resize_big_pkg,conv_s2_pkg,conv_s2_ref_pkg}
cp examples/resize_probe/target/dopamine-unknown-none-cvx/npu/resize_probe.cvx          $OUT/pkg/resize_pkg/step.cvx
cp examples/resize_probe_big/target/dopamine-unknown-none-cvx/npu/resize_probe_big.cvx  $OUT/pkg/resize_big_pkg/step.cvx
cp examples/conv2d_s2_probe/target/dopamine-unknown-none-cvx/npu/conv2d_s2_probe.cvx    $OUT/pkg/conv_s2_pkg/step.cvx
cp examples/conv2d_s2_ref/target/dopamine-unknown-none-cvx/npu/conv2d_s2_ref.cvx        $OUT/pkg/conv_s2_ref_pkg/step.cvx
: > $OUT/pkg/resize_pkg/param_blk0.bin ; : > $OUT/pkg/resize_big_pkg/param_blk0.bin      # resize 无权重
cp cases/conv_s2/param_blk0.bin $OUT/pkg/conv_s2_pkg/param_blk0.bin                      # 卷积：真实权重
cp cases/conv_s2/param_blk0.bin $OUT/pkg/conv_s2_ref_pkg/param_blk0.bin

# ③ 打包 + 自检（逐个）
for m in resize resize_big conv_s2 conv_s2_ref; do
  docc calbin from-calvx $OUT/pkg/${m}_pkg -o $OUT/$m
  docc calbin doctor $OUT/$m | tail -8
done
```

> ⚠️ `param_blk0.bin` 的长度必须等于程序里 `@w` 声明的字节数（resize=0；conv=73,984），否则 docc 报 `读不了 …/param_blk0.bin` 或长度不符。
> 子模型名就是 pkg 目录名（`resize_pkg` 等）——真机 `-m` 参数用它，`doctor` 输出里也能看到。

---

## 5. 板卡健康检查（在**板卡主机**上）

```bash
dort device -i 0 | grep "Dev  :"      # 必须看到 OS up   RT up
```

- 若显示 `OS up` 但 **`RT DOWN`**（或 `configure 失败：NoDevice`），按下面的顺序恢复：
  ```bash
  dort device reprobe -i 0 --yes      # SOFT_RESET；只有这一步会重启板上 RT
  sleep 15
  sudo rmmod calculet_pci && sudo insmod ~/cal-pcie-driver/calculet_pci.ko
  sleep 5
  dort device -i 0 | grep "Dev  :"    # 等到 OS up  RT up
  ```
- ⛔ **共享板/排队环境不要自行 reprobe**：先问负责人。⛔ 只做 `rmmod/insmod` 救不回 `RT DOWN`（见 `calBLAS/AGENTS.md:91-101`）。
- `dort overview` 可看设备概况；打不开 `/dev/calculet0` 时注意权限（可能需要 sudo 或把用户加进设备组）。

---

## 6. 上板运行（在**板卡主机**上）

### 6.1 先把 calbin/输入拷到板卡主机（若不同机）

```bash
# 在开发机：
scp -r onboard_cal cases compare_bf16.py  <user>@<board-host>:~/onboard/
```

### 6.2 一键跑（推荐）

```bash
cd ~/onboard
DEVICE=0 CAL_ROOT=$PWD/onboard_cal CASE_DIR=$PWD/cases OUT=$PWD/board_out bash run_on_board.sh
```

它会：健康检查 → 4 个用例各跑 3 次（保存 sha256）→ 自动对比 → 打印 `VERDICT`。
输出在 `board_out/`：`out_resize.bin`、`out_resize_big.bin`、`out_conv_fast.bin`、`out_conv_ref.bin`、`logs/`。

### 6.3 手动逐条（推荐第一次按这个走，便于核对）

**用例 1：resize（32×32×16 → 64×64×16）**

```bash
dort calbin run -i 0 -f onboard_cal/resize -m resize_pkg \
  --raw-in cases/resize_small/input.bin --raw-out out_resize.bin \
  --in-tensor in_x --out-tensor out_y
ls -l out_resize.bin        # 必须是 131072 字节
```

参数含义：`-i 0` 设备号；`-f` calbin 目录；`-m` 子模型名（doctor 里看到的 `resize_pkg`）；`--raw-in/--raw-out` 原始二进制输入/输出；`--in-tensor/--out-tensor` 程序里的 decl 名（`in_x`/`out_y`）。

**用例 2：resize 通道分块（32×32×256 → 64×64×256）**

```bash
dort calbin run -i 0 -f onboard_cal/resize_big -m resize_big_pkg \
  --raw-in cases/resize_big/input.bin --raw-out out_resize_big.bin \
  --in-tensor in_x --out-tensor out_y
ls -l out_resize_big.bin    # 必须是 2097152 字节
```

**用例 3A：conv stride-2 FAST 原型**

```bash
dort calbin run -i 0 -f onboard_cal/conv_s2 -m conv_s2_pkg \
  --raw-in cases/conv_s2/input.bin --raw-out out_conv_fast.bin \
  --in-tensor in_x --out-tensor out_y
ls -l out_conv_fast.bin     # 必须是 903168 字节（84×84×64 bf16）
```

**用例 3B：同形状慢路对照**

```bash
dort calbin run -i 0 -f onboard_cal/conv_s2_ref -m conv_s2_ref_pkg \
  --raw-in cases/conv_s2/input.bin --raw-out out_conv_ref.bin \
  --in-tensor in_x --out-tensor out_y
ls -l out_conv_ref.bin      # 必须是 903168 字节
```

**重复性检查**：每个用例再跑 2 次（换 `--raw-out` 文件名），三次输出 `sha256sum` 应完全相同。若不同：先怀疑板卡/程序状态（按 §9 排查），不要先改代码。

> `dort calbin run` 的 flag 随 dort 版本可能略有差异；不确定时先 `dort calbin run --help` 核对（本文按仓库 `examples/yolo26/deploy/DEPLOY.md:44` 的用法）。

---

## 7. 对比与判定

### 7.1 通用对比工具

```bash
python3 compare_bf16.py <实际输出> <期望输出> [选项]
# 选项：
#   --actual-shape H,W,C     把实际输出按形状 reshape（卷积必须）
#   --expected-shape H,W,C
#   --crop y0:y1,x0:x1       对实际输出裁剪（卷积裁掉外圈 2 像素 → 2:82,2:82）
#   --expected-crop ...      对期望输出裁剪（fast vs ref 时两边都裁）
# 输出：元素数、逐位相同比例、最大/平均 |diff|（f32）、bf16 ULP 近似、余弦、VERDICT
```

### 7.2 用例 1/2：resize 对 CPU 参考

```bash
python3 compare_bf16.py out_resize.bin     cases/resize_small/expected.bin --label resize
python3 compare_bf16.py out_resize_big.bin cases/resize_big/expected.bin   --label resize_big
```

**判定**：

| 结果 | 含义 | 处理 |
| --- | --- | --- |
| `VERDICT: EXACT`（100% 逐位） | ✅ 完全正确 | 记录 PASS |
| `NEAR`：逐位 ≥99.9% 且 max ≤2 ULP | ✅ 可接受（原因见 §7.5：f32 中间运算合同是 rnu、numpy 是 RNE；`0.75/0.25` 权重使 tie 极少） | 记录 PASS，注明差异元素数 |
| 逐位 <99%，或 max >2 ULP | ❌ FAIL | 按 §9 排查（先查输入/输出大小、bf16 位型、板卡状态） |

### 7.3 用例 3：conv fast vs 慢路（**最严格**）

```bash
python3 compare_bf16.py out_conv_fast.bin out_conv_ref.bin \
  --actual-shape 84,84,64  --crop 2:82,2:82 \
  --expected-shape 84,84,64 --expected-crop 2:82,2:82 \
  --label conv_fast_vs_ref
```

**判定**：必须 `VERDICT: EXACT`（逐位相同）。
理由：两条路径的 9 个 tap、mma 累加次序、bias、SiLU 宏完全相同，唯一区别是 lhs 取自 VB（fast）还是 DDR（slow）；若出现差异，说明 fast 原型的地址映射/奇偶有 bug（本次已在开发期修正过一次，这正是这个对照要防的）。

### 7.4 用例 3：conv 对 CPU 参考（宽松）

```bash
python3 compare_bf16.py out_conv_fast.bin cases/conv_s2/expected.bin \
  --actual-shape 84,84,64 --crop 2:82,2:82 --expected-shape 80,80,64 --label conv_fast_vs_cpu
python3 compare_bf16.py out_conv_ref.bin  cases/conv_s2/expected.bin \
  --actual-shape 84,84,64 --crop 2:82,2:82 --expected-shape 80,80,64 --label conv_ref_vs_cpu
```

**判定（两个都应满足）**：`cosine ≥ 0.9999` 且 `max diff (bf16 ULP) ≤ 4`。
原因：CPU 参考是 f64 累加 + 近似 SiLU，而硬件是 TU f32 分块累加 + bf16 近似 `exp/div`，**不可能逐位**；这条只用于排除"整块算错/布局错"，严格判据是同硬件慢路对照（§7.3）。

### 7.5 为什么 resize 可以接近逐位、conv 不能

- resize 全链路只有 f32 乘加 + 一次 bf16 舍入，权重是精确的 0.25/0.75；唯一潜在差异是 docc 给 f32 逐元素运算挂的 numerics 合同是 `rnu`（ties-away）而 numpy 默认 RNE —— 只有"恰好半个 ULP"才分叉，概率极低。
- conv 的 TU 是**分块 f32 累加**，求和次序与 numpy 不同；SiLU 用 bf16 近似 `exp`/`div`（KS1 硬件近似，`rsqrt/exp` ≤1 ULP）；这两者都会让结果与 CPU 参考差几 ULP。**同硬件慢路对照**则不受这些影响，所以它是严格判据。

---

## 8. 记录模板（每次真机验证都填）

```md
## 真机验证记录
- 日期/操作者：
- 板卡：设备号 ___，型号 ___，固件 ___（dort board -i 0）
- 代码版本：calBLAS commit ___（git rev-parse --short HEAD），补丁 sha256 ___
- 工具链：rustc ___，docc ___，dort ___
- 命令：dort calbin run -i 0 -f ___ -m ___ --raw-in ___ --raw-out ___ --in-tensor in_x --out-tensor out_y
- 输出 sha256（3 次）：run1 ___ / run2 ___ / run3 ___（一致？是/否）
- 对比结果：
  | 用例 | 逐位相同率 | max diff (ULP) | cosine | 结论 |
  | resize | | | | |
  | resize_big | | | | |
  | conv_fast_vs_ref | | | | |
  | conv_vs_cpu | | | | |
- 结论：PASS / FAIL（原因）：
- 异常/备注：
```

---

## 9. 排错表（按症状查）

| 症状 | 可能原因 | 处理 |
| --- | --- | --- |
| `dort` 报 `未发现 Calculet 设备` | 板卡未上电 / 驱动未加载 / 设备号不对 | `dort device` 看全部设备；`ls /dev/calculet*`；换 `-i` |
| `RT DOWN` / `configure 失败：NoDevice` | 板上 RT 挂了 | 按 §5 流程 reprobe + rmmod/insmod；共享板先问负责人 |
| `model not found` / 子模型不存在 | `-m` 名字和 calbin 里不一致 | 用 `docc calbin doctor <calbin目录>` 输出里的 `── <名字>` 核对 |
| `tensor not found` / 输入名不对 | `--in-tensor/--out-tensor` 与 decl 名不符 | decl 名在 example 的 `src/step.rs`：`in_x`、`out_y` |
| `--raw-in` 报大小不符 | 输入文件大小 ≠ decl 字节数 | resize：32,768 / 524,288；conv：3,442,688（见 §10 附录 A） |
| 输出文件大小不对 | `--raw-out` 未写全/程序 decl 改过 | 期望：131,072 / 2,097,152 / 903,168 |
| 输出全 0 | 输入没写进去 / 参数块是空占位 / 板卡刚复位 | 检查 raw-in；conv 必须用 `cases/conv_s2/param_blk0.bin`（长度 73,984） |
| 输出大面积乱/明显错位 | 打包时 calbin 和程序不匹配 | 重新 `docc calbin from-calvx` + doctor，确保用的 `.cvx` 是刚编的 |
| conv 与 CPU 参考 cos 很低（<0.99） | 输入/权重布局或 padding 理解错 | 严格先做 §7.3 fast vs ref；若 ref vs CPU 也低，检查 `gen_conv_s2_case.py` 的布局与 `param_blk0.bin` 偏移（bias 在 73,728） |
| conv fast vs ref 有少量差异 | 疑似索引/奇偶 bug | 记录差异位置（哪个 (oy,ox,co)），对照 `examples/conv2d_s2_probe/README.md` 的 tap 映射表 |
| `docc calbin from-calvx` 报 `param_blk0.bin` 读不了 | 打包目录缺参数文件或长度不对 | resize 目录放 **0 字节** `param_blk0.bin`；conv 放 73,984 字节 |
| doctor 警告 `ping and pong the same address` | `double_buffer` 的 ping/pong 落到同址（仅警告） | 单次推理不受影响；若连续多轮推理出现覆盖，记录并避免复用同一输出 |
| `dort calbin run` 卡住/DeviceCrash | 板上状态异常或程序挂死 | 不要连发；记录日志；按 §5 恢复；必要时找组委会 |
| 数值差 1 ULP 级别 | rnu/RNE、SiLU 近似 | 属正常范围（见 §7.5），记录即可 |

---

## 10. 附录

### A. 文件/名字/尺寸速查

| 用例 | 子模型名 `-m` | in decl | out decl | 输入字节 | 输出字节 | 权重字节 |
| --- | --- | --- | --- | --- | --- | --- |
| resize | `resize_pkg` | `in_x` | `out_y` | 32,768 | 131,072 | 0 |
| resize_big | `resize_big_pkg` | `in_x` | `out_y` | 524,288 | 2,097,152 | 0 |
| conv fast | `conv_s2_pkg` | `in_x` | `out_y` | 3,442,688 | 903,168 | 73,984 |
| conv ref | `conv_s2_ref_pkg` | `in_x` | `out_y` | 3,442,688 | 903,168 | 73,984 |

本机平台侧产物（供核对，尺寸会随工具链小版本略有差异）：

| 用例 | `.cvx` | `.cvxo` | ccu0 ELF | ccu1 ELF | ELF load_size | 反汇编 |
| --- | --- | --- | --- | --- | --- | --- |
| resize | 35,160 B | 6,655 B | 19,560 B | 6,872 B | 14,608 | 2,849 条 / 0 unknown |
| resize_big | 40,369 B | 7,118 B | 86,568 B | 6,872 B | 81,612 | 15,997 条 / 0 unknown |
| conv fast | 14,496 B | 2,608 B | 22,168 B | 22,168 B | 17,216 | 2,699 条 / 0 unknown |
| conv ref | 15,241 B | — | 16,152 B | 16,152 B | 11,204 | 1,897 条 / 0 unknown |

### B. 命令速查

```bash
# 开发机：打包全部
cd onboard && CALBLAS=$HOME/calBLAS CASE_DIR=$PWD/cases OUT=$PWD/onboard_cal bash package_all.sh

# 板卡主机：一键验证
cd ~/onboard && DEVICE=0 CAL_ROOT=$PWD/onboard_cal CASE_DIR=$PWD/cases OUT=$PWD/board_out bash run_on_board.sh

# 单条跑 + 对比（resize 为例）
dort calbin run -i 0 -f onboard_cal/resize -m resize_pkg --raw-in cases/resize_small/input.bin --raw-out out_resize.bin --in-tensor in_x --out-tensor out_y
python3 compare_bf16.py out_resize.bin cases/resize_small/expected.bin --label resize
```

### C. 安全与规范

1. **真机资源有限**（赛题文档原话），先在本机把设备遍编通再来排队；上板前确认板卡健康。
2. **不要自行 reprobe 共享板**；不要跑 `dort` 的写命令（`reg write`、`jobq reinit` 等）。
3. **一次只动一个变量**：数值不对时，先确认"同一份 calbin/输入/参数块"重跑 3 次稳定，再换别的排查（对齐 `calBLAS/AGENTS.md:107`）。
4. **保留原始证据**：每次运行保留 `--raw-out` 文件 + `sha256sum` + `dort device` 输出 + doctor 日志。
5. `.cvx` 只用于调试；判对错以"真机输出 vs 参考"为准。

---

**文档结束**。遇到问题按 §9 先定位；无法解决时把 §8 的记录模板 + `board_out/logs/` 一起发回来。
