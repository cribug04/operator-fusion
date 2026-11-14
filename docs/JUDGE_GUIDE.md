# 评委 / 复现者指南：10 分钟能在无板子的机器上核验什么

目标读者：想自己动手核验、而不只看文档的人。每一步给出命令和**我们实际得到的输出**，对不上就说明环境或版本不同。
真机结果（Resize 逐位相同、VAE PSNR 52.58 dB）需要 Calculet 板卡，本指南不覆盖；它们的来源与状态见 [README §0](../README.md)。

## A. 只需 Python 3 + numpy（约 2 分钟）

```bash
git clone https://gitee.com/nssdd/operator-fusion.git && cd operator-fusion
python3 -m unittest discover -s tests            # 43 项：验收工具、bf16 对比器、19 项 Resize 边界、3 项双核分配
python3 onboard/check_resize_cpu.py              # 两个已上板用例的 CPU 参考与数据复核
```

期望：

```
Ran 43 tests in …s
OK
resize_small: 按 kernel 次序计算; 65536 元素; 差异 0
resize_small: 独立坐标公式计算; 65536 元素; 差异 0
resize_big: 按 kernel 次序计算; 1048576 元素; 差异 0
resize_big: 独立坐标公式计算; 1048576 元素; 差异 0
PASS: 两个已提交 Resize 用例的 CPU 参考与数据一致；未执行 NPU。
```

这证明：板上拿来对比的 `expected.bin` 不是随手生成的，三种互相独立的算法（复刻 kernel 次序、一维地址仿真、独立坐标公式）给出同一份 bf16 位型。
测试内容说明见 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)。

## B. 加上普通 Rust 工具链（约 3 分钟）：calDNN 图注册与编排

需要 calBLAS / calDNN / npu 三个官方仓库并排放置（匿名可克隆）。

```bash
for r in calBLAS calDNN npu; do git clone http://115.159.82.89/git/calculet/$r.git; done
( cd calDNN  && git apply ../operator-fusion/patches/caldnn_resize.patch && cargo build )
( cd calBLAS && git apply ../operator-fusion/patches/onboard_kit.patch   && cargo build -p calblas )
cd calBLAS/examples/resize_probe && cargo run -q -p resize-plan && cargo test -q -p resize-plan
```

期望：

```
2 个张量、1 个运算 ⇒ 1 个作业，工作区 DDR 0 字节、SRAM 0 字节
  resize_bilinear2(%0, %1)  arg_kinds = ["ptr:ddr", "ptr:ddr"]
bind ⇒ resize_bilinear2([Ptr(4294967296), Ptr(4295032832)])
test result: ok. 5 passed; 0 failed
```

这证明：图里 `Op::Resize` 能被 calBLAS 引擎表命中、生成作业并绑定地址，即路线 A 的「注册进引擎表、由图调用」。
官方 calDNN 自己的图发射路径尚未打通（其 README「现状」），所以这一步止于编排层，上板走整程序路线。详见 [CALDNN_INTEGRATION.md](CALDNN_INTEGRATION.md)。

## C. 有 calculet 工具链时（Linux x86_64，约 5 分钟）：设备端编译与 docc 自检

```bash
cd operator-fusion/onboard
CALBLAS=/path/to/calBLAS bash package_all.sh      # 7 个程序：编译 → .cvx → docc calbin from-calvx → doctor
```

期望：每个程序打印 `拒收 0 条`（警告 2 条是 ping/pong 两个固定项），最后列出 7 个 calbin 目录。
可核对的哈希（`sha256sum examples/<名>/target/dopamine-unknown-none-cvx/npu/<名>.cvx`，前 16 位）：

| 程序 | `.cvx` sha256 前缀 | 说明 |
| --- | --- | --- |
| resize_probe | `536b406a6ec2512b` | 已上板，32×32×16 |
| resize_probe_big | `5a22650a3f29335a` | 已上板，32×32×256 分块 |
| resize_probe_big2 | `e13fba7920c6efd2` | 双核版，未上板 |
| resize_probe_huge / huge2 | `fc7f8dc099101e0d` / `172a4ee9bf89a02d` | 计时对，未上板 |

两次构建逐字节相同（`evidence/*/sha256*.txt`）。

VAE 5 段同理：把 `sd_vae/kernels/<段>/` 复制为 `calBLAS/examples/sd_vae_<段>/`，`cargo build --lib --features step --target dopamine-unknown-none-cvx`，
再 `docc calbin from-calvx` + `doctor`。期望 5 段全部 `拒收 0 条`，`.cvx` 与参数块字节数见 [README §2.5](../README.md)。打包参数块需要 SD1.5 fp32 权重与 torch，见 [SD_VAE_ON_NPU.md](SD_VAE_ON_NPU.md) §6。

## D. 读代码时建议的顺序

1. [kernels/resize_bilinear2.rs](../kernels/resize_bilinear2.rs)：101 行，整个算子。重点看三段展开与 `@h` 的行内加权。
2. [onboard/resize_reference.py](../onboard/resize_reference.py)：三路参考实现，`emulate_flat` 按 kernel 的地址算式在一维内存里仿真。
3. [patches/caldnn_resize.patch](../patches/caldnn_resize.patch)：79 行，图框架改动的全部。
4. [docs/REVIEW_SD_VAE_2026-11-14.md](REVIEW_SD_VAE_2026-11-14.md)：VAE 审查，两条阻断级发现各有可复跑的探针实验。
5. [sd_vae/gen/gen.py](../sd_vae/gen/gen.py)：层表到 Rust 的生成器。
