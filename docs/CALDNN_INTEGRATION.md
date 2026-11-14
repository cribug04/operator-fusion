# Resize 接入 calDNN 图 / 引擎表（2026-11-14）

**状态**：calDNN 与 calBLAS 主机侧编译通过 ✅ · 图 ⇒ 计划 ⇒ bind 演示通过 ✅ · 5 项主机单元测试通过 ✅ · 设备端 `.cvx` 逐字节不变 ✅ · **没有发射、没有上板 ❌**

赛题路线 A 的验收写了「能通过 calDNN 引擎表注册、由图调用」。之前 Resize 只是一段孤立的 CalVX 程序（`examples/resize_probe/src/step.rs` 直接调 kernel）。这一轮把它注册进官方图框架。

## 1. calDNN 的机制（读源码得到，不是猜的）

calDNN 共 1174 行，分四层（`calDNN/src/lib.rs`）：

| 层 | 内容 | 位置 |
| --- | --- | --- |
| 规格 | `TensorDesc`（类型、维、是否虚拟）、`Op`（形状描述 + 读写的张量号）、`Graph`（加运算的方法） | `graph.rs` |
| 引擎 | `trait Engine { fn step(&self, g, at) -> Option<Step> }`，`FirstFit` 按次序取第一个认得的 | `engine.rs` |
| 计划 | `Plan::new(&graph, engines, &heuristic)` 逐个运算挑引擎，给虚拟张量分 512 B 对齐的工作区；`Plan::bind(&VariantPack)` 把张量号换成地址 | `plan.rs` |
| 作业 | `Job { kernel, arg_kinds, elf, args }`，由 kernel 主机桩的 `npu::Launch` 转来 | `job.rs` |

calBLAS 侧（`calBLAS/src/engines.rs`）用一个 `engines!` 宏为每种运算生成查表引擎：持有 `&[(Desc, fn(...) -> Launch)]`，`step` 时拿图里该运算的 `Desc` 到表里找**完全相等**的一项。每个 kernel 模板用 `calblas::desc!` 生成同名模块里的 `名字::DESC`，`calblas::entry!(名字)` 展开成 `(名字::DESC, 名字)`。引擎表跟模型走，放在 `examples/<模型>/src/engines.rs`。

官方 README 明示：这条图路线「没有发射过，也没有上板」，现有全部视觉算子（包括已上板的 YOLO26）都「不进图」。所以本次接入证明的是**规格与编排层面的注册**，不是真机执行。

## 2. 改了什么

### 2.1 calDNN（[patches/caldnn_resize.patch](../patches/caldnn_resize.patch)，79 行）

`src/graph.rs`：

```rust
pub enum Interp { Nearest, Bilinear }                       // 插值方式
pub struct ResizeDesc { h, w, c, scale: u32, mode: Interp }  // 整数倍上采样，NHWC 紧排、无边框
pub enum Op { …, Resize(ResizeDesc, [Uid; 2]) }             // x [h][w][c]、y [scale·h][scale·w][c]
impl Graph {
    pub fn resize(&mut self, x: Uid, scale: u32, mode: Interp) -> Uid   // x 须三维，y 是虚拟张量
    pub fn upsample_bilinear2(&mut self, x: Uid) -> Uid                  // resize(x, 2, Bilinear)
}
```

`src/lib.rs`：导出 `Interp`、`ResizeDesc`。写法照 `ConvDesc` / `HwptnDesc` 与 `Graph::pointwise`。

### 2.2 calBLAS（[patches/onboard_kit.patch](../patches/onboard_kit.patch) 已重新生成）

- `src/engines.rs`：在 `engines!` 表里加一行
  `Resize(ResizeDesc, Resize, Notes::NONE) fn(Bf, Bf) = |io| (io[0], io[1]);`
- `src/kernels/vision.rs`：`resize_bilinear2!` 与 `resize_bilinear2_tiled!` 的主分支开头加
  `$crate::desc!($vis $name: ResizeDesc { h, w, c, scale: 2, mode: Interp::Bilinear });`
  分块版的 `cc` 是实现细节，不进描述：图只说「算什么」。
- `examples/resize_probe/src/engines.rs`：`ENGINES = &[&Resize(&[entry!(k::resize_bilinear2)])]`，`lib.rs` 用 `#[cfg(not(all(target_arch = "riscv32", target_os = "none")))]` 只在主机编（照 `examples/laya`）。
- `examples/resize_probe_big/src/engines.rs`：同上，注册 `resize_bilinear2_big`（32×32×256）。
- `examples/resize_probe/plan/`：主机侧 bin crate `resize-plan`。`main` 搭「输入 ⇒ 双线性 ×2 ⇒ 输出」的图、编排、打印作业、示意地址 bind；`#[cfg(test)]` 5 项测试。

本仓库 [kernels/](../kernels/)、[examples/](../examples/) 的源码副本已同步，并与 Linux 构建副本逐行比对一致。

## 3. 实测结果（Linux calc，calculet 工具链）

```
$ cd ~/bohack/calDNN && cargo build                 # Finished
$ cd ~/bohack/calBLAS && cargo build -p calblas -p laya   # 无 error
$ cd examples/resize_probe && cargo run -q -p resize-plan
2 个张量、1 个运算 ⇒ 1 个作业，工作区 DDR 0 字节、SRAM 0 字节
  resize_bilinear2(%0, %1)  arg_kinds = ["ptr:ddr", "ptr:ddr"]
bind ⇒ resize_bilinear2([Ptr(4294967296), Ptr(4295032832)])
$ cargo test -q -p resize-plan
test result: ok. 5 passed; 0 failed
```

5 项测试：图记录 `Op::Resize` 且输出维为 `[64, 64, 16]`；引擎表命中 32×32×16 并出 1 个作业、参数为张量号；16×16×16、32×32×32、64×64×16 报 `Unsupported`；bind 把张量号换成地址、缺地址报 `Unbound`；连用两次 ×2 时第二个在第 1 个运算处报 `Unsupported`。

设备端复编：`resize_probe.cvx` sha256 前缀 `536b406a6ec2512b`、`resize_probe_big.cvx` `5a22650a3f29335a`，与 `evidence/` 里的旧产物**逐字节相同**，所以之前的 docc 打包、doctor、反汇编结论不受影响。

## 4. 复现

```bash
cd ~/bohack/calDNN && git apply /path/to/patches/caldnn_resize.patch && cargo build
cd ~/bohack/calBLAS && git apply /path/to/patches/onboard_kit.patch && cargo build -p calblas
cd examples/resize_probe && cargo run -p resize-plan && cargo test -p resize-plan
```

## 5. 没做的 / 已知限制

1. **没有发射**：calDNN 自己的 `Plan` ⇒ dort jit 发射路径官方尚未打通（`calDNN/README.md`「现状」）。要在板上跑 Resize，仍走 `examples/resize_probe/step.rs` 的整程序 + `docc calbin` + `dort calbin run` 路线，见 [REAL_DEVICE_VERIFICATION.md](REAL_DEVICE_VERIFICATION.md)。
2. 引擎只认**完全相等**的 `ResizeDesc`，这是 calBLAS 的既定设计（kernel 尺寸是编译期常量）。要支持别的尺寸就多特化几行 `resize_bilinear2!(...)` 并加进表。
3. `Interp::Nearest` 只是枚举占位：现有 `upsample2!` 用带边框的视觉排布（`[H+4][W+4][P]`），与 `ResizeDesc` 的无边框紧排语义不同，没有注册。
4. `Graph::resize` 对非三维输入 `assert!` panic，与同文件其它方法处理规格错误的方式一致。
5. calDNN 的 `Op` 注释说运算集合「对齐 docc doccgraph 的 30 个图算子」。新增的 `Resize` 没有对应的 doccgraph 算子；若赛题方要求图算子与 docc 一一对应，这一项需要和组委会确认。
