# Calculet NPU 黑客松 · 开发环境与仓库勘察报告

**文件**：`CALCULET_RECON.md`
**生成时间**：2026-11-13 17:58 (+08:00)
**工作目录**：`C:\Users\10463\Desktop\bohack`
**仓库位置**：`calculet-repos/{calBLAS, calDNN, npu}`（三仓库同级，见 §3）
**原始证据目录**：`_recon/`（install.sh、赛题页、快速开始、发行清单、各阶段命令输出全部留档）
**角色**：首席开发工程师（Lead，本报告由 Lead 汇总并对关键断言逐条复核）

**验证标记（全文统一）**

| 标记 | 含义 |
| --- | --- |
| ✅ 已验证（本机实测） | 在本机执行命令得到的真实输出 / 网络实测 |
| ✅ 已验证（源码阅读） | 直接读到源码并逐行核对（给 `文件:行`） |
| ⚠️ 文档口径（本机未复现） | 文档这么写，但本机无法执行验证；或文档之间互相矛盾 |
| ❌ 本机阻塞 | 因环境缺失无法执行 / 无法验证 |
| 🧠 推理 | 由已读证据推导，不是直接观测 |
| 🚫 阻塞 | 需要外部条件（Linux 主机、真机、工具链）才能推进 |

> **诚实声明**：本报告不编造任何命令输出、不做“假装编译成功”的描述。凡未在本机跑过的，一律标 ❌/⚠️；凡 API 引用，全部来自仓库真实源码（给出行号）。第 4 章的“修复建议”只给环境类检查项，没有、也无法针对真实编译日志做修复——因为本机拿不到真实编译日志。

---

## 0. 结论速览（TL;DR）

| 阶段 | 任务 | 状态 | 一句话 |
| --- | --- | --- | --- |
| 一 | 环境检查 | ⚠️ 部分满足 | `git` ✅、x86_64 ✅；**Linux ❌、rustup/rustc/cargo ❌**。本机是 **Windows 11 + MSYS2**，不是 Linux，也没有可用的 WSL |
| 二 | 安装工具链 | ✅ 完成（WSL2 路线） | 原生 Windows 装不上（发行包无 Windows host，已实测）；改用 **WSL2 + Ubuntu**：WSL 3.0.5 + rustup + calculet 工具链（rustc 1.98.1、docc 0.4.1、dort 0.4.1）全部就绪（§15–§16） |
| 三 | 获取源码 | ✅ 完成 | 三仓库全部克隆成功（`calBLAS@32d2bb8`、`calDNN@1e06c41`、`npu@1e06c41`，均为 2026-11-13 下发版） |
| 四 | 编译 tiny | ✅ 成功 | WSL2 内实跑成功：生成 **`tiny.cvxo` (14,792 B) + `tiny.cvx` (89,382 B)**；整步 `-cvx` 也成功（§16.2） |
| 五 | 仓库分析 | ✅ 完成 | 7 项问题全部有源码级答案（算子覆盖、SYNTAX、tiny、模板、Resize 核查、calDNN 图 API、真机测试） |
| 六 | 本报告 | ✅ 交付 | 含 3 个缺失算子建议、最终选题、48h 分工、未解决问题 |

**最重要的一句话**：**代码生态比“从零开始”好得多**——Nearest Resize 已经有了（`upsample2!`，固定 ×2）、2D 卷积/深度卷积/5×5 池化/GroupNorm/空间注意力模板都已存在，YOLO26n 已在真机上跑通并有数值记录；但 **Bilinear Resize 完全不存在**，CV 算子目前也全部“不进图”（无法通过 calDNN 引擎表注册）。因此 48 小时最稳的选题是**做一个 Bilinear Resize 算子并在真机验证**，而不是做完整 Stable Diffusion。

**当前 P0 阻塞**：Linux/工具链已解决（§15–§16），**只剩真机访问**（无板卡/驱动/pydort；本机也没有 `cvx` 模拟器）。见 §14。

---

## 1. 环境检查结果（第一阶段）

### 1.1 题目要求的 4 条命令与真实输出

证据原文：`_recon/phase1_evidence.txt`（2026-11-13 17:35 采集）。

```bash
$ uname -m
x86_64

$ uname -a
MSYS_NT-10.0-26200 cribug 3.6.10-5a1665c8.x86_64 2026-09-28 10:40 UTC x86_64 Msys

$ git --version
git version 2.56.0.windows.2

$ rustup --version
/dev/stdin: line 1: rustup: command not found        # exit=127

$ rustup toolchain list
/dev/stdin: line 1: rustup: command not found        # exit=127
```

补充实测（同样在证据文件里）：

```bash
$ rustc --version   → rustc: command not found      (exit 127)
$ cargo --version   → cargo: command not found      (exit 127)
$ docc  --version   → docc:  command not found      (exit 127)
$ dort  --version   → dort:  command not found      (exit 127)
$ echo $MSYSTEM → MSYS ; bash 5.3.15(2)-release
$ ls ~/.cargo ~/.rustup → No such file or directory
```

### 1.2 附加环境探测（为什么判断“不是 Linux”）

| 探测项 | 实测结果 | 结论 |
| --- | --- | --- |
| OS | Microsoft Windows 11 家庭中文版，Version 10.0.26200，Build 26200 | ✅ 已验证：Windows 11 |
| CPU / 内存 | 13th Gen Intel Core i9-13900H / 31.6 GB | ✅ 已验证：硬件足够（若装 Linux/WSL） |
| Shell | MSYS2（`MSYS_NT-10.0-26200 … Msys`，bash 5.3.15） | ✅ 已验证：POSIX 兼容层，**不是 Linux 内核** |
| WSL | `wsl.exe` 存在，但 `-l -v` / `--version` / `--status` / `-e echo` 均无输出（exit 127）；`AppData\Local\Packages` 下无任何 Ubuntu/Debian/WSL 发行版目录；`LxssManager` 服务查不到（未运行/不存在） | ✅ 已验证：**没有可用的 WSL 发行版** |
| 管理员权限 | `IsInRole(Administrator) = False`；查询 WSL 可选功能直接报“该操作需要提升” | ✅ 已验证：**无法启用/安装 WSL** |
| Docker | `docker: command not found` | ✅ 已验证：无容器环境 |
| SSH 远端 | `ssh_list` → `no hosts configured` | ✅ 已验证：没有可用的远端 Linux 主机 |
| 网络到赛题服务器 | `http://115.159.82.89/` → HTTP 200；`install.sh` → HTTP 200（875 B） | ✅ 已验证：网络可达 |

### 1.3 判定：不满足比赛要求

官方快速开始（`_recon/quickstart.md:5-8`，以及赛题页 §2）明确要求：

> 环境：**Linux x86_64**；已安装 **rustup**；**git**

对号入座：

| 要求 | 本机 | 结论 |
| --- | --- | --- |
| Linux x86_64 | Windows 11 / MSYS2；`uname -m` 虽然是 `x86_64`，但 `uname -s` 语义是 `MSYS_NT`，不是 Linux | ❌ **不满足** |
| rustup / rustc / cargo | 全部 `command not found` | ❌ **不满足** |
| git | git 2.56.0.windows.2 | ✅ 满足（Windows 版 git 可用，克隆/读仓库无障碍） |

> ⚠️ 容易误判的一点：在 MSYS2 里 `uname -m` 输出 `x86_64`，**不能**据此认为满足“Linux x86_64”。平台判定必须看 `uname -a` 的系统标识。

### 1.4 该结论的影响

- 第三阶段（克隆源码）不受影响 —— 已全部完成（§3）。
- 第二阶段（装工具链）、第四阶段（设备遍编译）、任何 docc/dort 调用、真机运行，全部**无法在本机执行**（§2、§4、§9）。
- 修正路径（按推荐顺序，均可解除阻塞）：
  1. **在另一台 Linux（x86_64）机器/云主机上执行**（最省事，组委会文档给出的标准路径）；
  2. 本机启用 **WSL2**（需要管理员权限 + 重启；先 `wsl --install -d Ubuntu`，再按 §2 安装工具链）；
  3. 申请组委会/平台提供的 Linux 开发机。

---

## 2. 工具链安装与验证（第二阶段）

> **更新（2026-11-13 18:57）**：已通过 **WSL2 + Ubuntu** 完成安装与验证，见 §16；本节保留当时的 Windows 原生失败记录（作为“为什么必须 Linux”的证据）。

### 2.1 获取与完整性记录（✅ 本机实测）

```text
URL      : http://115.159.82.89/install.sh
HTTP     : 200 OK
大小      : 875 bytes
sha256   : 3c175657a77a5f40afee8d96c79af785db59a8f87868381b1b2257a93bd92950
本地副本  : _recon/install.sh
同源页面  : http://115.159.82.89/（赛题页，HTTP 200，nginx/1.28.3）
```

### 2.2 install.sh 全文与逐项安全审查（✅ 已验证：内容已读）

```sh
#!/bin/sh
# Calculet 工具链安装：rustc 1.98.1 + 设备 target dopamine-unknown-none-elf / -cvx + docc + dort
# 用法：curl -sSf http://115.159.82.89/install.sh | sh
set -e
SERVER=http://115.159.82.89
CHANNEL=nightly-2026-11-13
command -v rustup >/dev/null 2>&1 || { echo "先安装 rustup：https://rustup.rs（国内可用 https://rsproxy.cn）"; exit 1; }
RUSTUP_DIST_SERVER=$SERVER rustup toolchain install "$CHANNEL" --profile minimal --no-self-update
HOST=$(rustc +"$CHANNEL" -vV | sed -n 's/^host: //p')
DIR=$(rustup run "$CHANNEL" rustc --print sysroot)
rustup toolchain uninstall calculet >/dev/null 2>&1 || true
rustup toolchain link calculet "$DIR"
BIN=${CARGO_HOME:-$HOME/.cargo}/bin
for t in docc dort; do ln -sf "$DIR/bin/$t" "$BIN/$t"; done
echo "已安装 calculet 工具链（$HOST）：$(rustc +calculet --version)"
echo "docc / dort 已链到 $BIN"
```

**逐项审查**

| 行为 | 风险评估 |
| --- | --- |
| `set -e` | 安全，出错即停 |
| 先检查 `rustup` 是否存在，否则退出 | 安全，有前置门 |
| 只从 `RUSTUP_DIST_SERVER=http://115.159.82.89` 装一个固定版本 `nightly-2026-11-13`（minimal profile） | 范围最小；但**这是由赛题方自建的 rustc 发行包** |
| `rustup toolchain uninstall calculet` | 只删名为 `calculet` 的链接，不动官方工具链，安全 |
| `rustup toolchain link calculet <sysroot>` | 把刚装好的发行包再链接成 `calculet`（仓库 `rust-toolchain.toml` 里 `channel = "calculet"` 依赖它） |
| `ln -sf "$DIR/bin/docc" "$DIR/bin/dort"` 到 `~/.cargo/bin` | 只建符号链接，安全 |
| 没有 `sudo`、没有 `rm -rf`、没有其它下载、没有向外部发送数据 | 未发现恶意行为 |

**信任判定（⚠️ 需注意）**

- ✅ 来源一致性：`install.sh`、赛题页、快速开始、三个仓库的 `rust-toolchain.toml` 注释互相印证，属于同一套赛事物料。
- ⚠️ **传输是明文 HTTP**：脚本内容本身可能被中间人替换（虽然本机拿到的 875 B 版本已审查无恶意行为，sha256 已记录，可请组委会核对）。
- ⚠️ **本质上是“运行赛题方自编译的 rustc”**：任何自定义工具链都等价于执行对方代码，这是此类 NPU 比赛的正常模式，但建议在**隔离环境/专用机器**里执行，不要在日常主力机上直接 `curl | sh`。
- ✅ 本机做法：**先下载到磁盘 → 读全文 → `sh -n` 语法检查 → 再执行**（没有 `curl | sh`）。

### 2.3 受控执行结果（✅ 本机实测）

```bash
$ sh -n install.sh
sh -n: OK

$ sh install.sh
先安装 rustup：https://rustup.rs（国内可用 https://rsproxy.cn）
# exit=1
```

证据：`_recon/install_run.log`。**脚本在自身前置检查处安全退出，没有任何安装动作发生。**

### 2.4 决定性验证：发行包没有 Windows 主机目标（✅ 本机实测）

仅仅“没装 rustup”还不足以说明“装了也没用”。为此做了两步硬验证：

**第 1 步：读发行清单**（`_recon/channel.toml`，来自 `http://115.159.82.89/dist/2026-11-13/channel-rust-nightly.toml`，HTTP 200）：

```toml
manifest-version = "2"
date = "2026-11-13"

[pkg.rustc]
version = "1.98.1-calculet (2026-11-13)"

[pkg.rustc.target."x86_64-unknown-linux-gnu"]   # ← 只有 Linux x86_64
available = true
xz_url = "https://calbin-…cos.ap-beijing.myqcloud.com/…/rustc-1.98.1-x86_64-unknown-linux-gnu.tar.xz"
xz_hash = "018c4dc6127518cc6c1a06e49213dd220f94474d9309ea59fc4c3c67a5ea54cc"
```

- rustc / cargo / rust-std / rustfmt / clippy 的 target **只有 `x86_64-unknown-linux-gnu`**；
- 另有 `rust-std.target."riscv32imafc-unknown-none-elf"`（设备侧标准库）与 `rust-src`（host 无关）；
- COS 制品实测存在：`rustc-1.98.1-x86_64-unknown-linux-gnu.tar.xz`，120,598,208 字节，HTTP 200；
- 直接探测 `rust-std-nightly-x86_64-pc-windows-msvc.tar.xz` → **HTTP 404**；苹果 aarch64 同样 404。

**第 2 步：隔离环境里的 rustup 实测**（不污染用户目录；`RUSTUP_HOME`/`CARGO_HOME` 都指向工作区，用完已删除，日志保留在 `_recon/toolchain_install_probe.log`）：

```text
$ rustup --version
rustup 1.29.3 (edbc831ce 2026-11-02)

$ RUSTUP_DIST_SERVER=http://115.159.82.89 rustup toolchain install nightly-2026-11-13 --profile minimal --no-self-update
info: syncing channel updates for nightly-2026-11-13-x86_64-pc-windows-msvc
info: latest update on 2026-11-13 for version 1.98.1-calculet (2026-11-13)
error: target 'x86_64-pc-windows-msvc' not found in channel.
       Perhaps check https://doc.rust-lang.org/nightly/rustc/platform-support.html for available targets
# exit=1
```

**结论（✅ 已验证）**：
1. 赛题服务器本身是好的（rustup 能连上，能读到 `1.98.1-calculet` 版本号）；
2. **但发行包里根本没有 Windows 构建**，所以在本机装 rustup 也装不上这个工具链；
3. 本机无法获得 `rustc 1.98.1-calculet`、`docc`、`dort`。

### 2.5 rustc / cargo / docc / dort 验证结果

| 组件 | 期望 | 本机实测 | 状态 |
| --- | --- | --- | --- |
| rustc | `1.98.1-calculet` + CalVX 后端 | `command not found` (127) | ❌ 不可用 |
| cargo | 与官方 1.98.1 相同 | `command not found` (127) | ❌ 不可用 |
| docc | calbin 打包/检查/比对 | `command not found` (127) | ❌ 不可用、仓库中也没有该二进制 |
| dort | 板卡管理与运行 | `command not found` (127) | ❌ 不可用、仓库中也没有该二进制 |

> 说明：`docc`/`dort` 在仓库里只以“外部路径”出现（如 `../docc/target/release/docc`、`PYTHONPATH=<pydort>`），三个 repo 内不含它们的源码或二进制；按赛题页说法，它们只随工具链提供二进制。

### 2.6 结论与解除前提

- **工具链：未安装、未验证**（❌）。原因不是操作失误，而是**平台不兼容 + 前置 rustup 缺失**，且已被实测证据钉死。
- 解除前提（任一）：
  1. 一台 **Linux x86_64** 机器（或云主机），装好 rustup 后执行 `install.sh`；
  2. 本机安装 **WSL2**（需要管理员），在 Ubuntu 里执行同样步骤（注意：WSL 里也同样需要 rustup，官方脚本 `https://sh.rustup.rs` / `rsproxy.cn` 均可）；
  3. 组委会提供开发机镜像。
- 建议向组委会确认：是否有 **HTTPS 版 install.sh** 或**安装包 sha256 官方公示**，以便在共享机器上安全安装。

---

## 3. 源码获取（第三阶段）✅

### 3.1 执行结果（✅ 本机实测）

原命令（题目要求三仓库同级）：

```bash
for r in calBLAS calDNN npu; do
    git clone http://115.159.82.89/git/calculet/$r.git
done
```

实际在本机以并行方式执行（结果等价，三仓库均同级）：

```bash
mkdir -p calculet-repos && cd calculet-repos
( git clone -q http://115.159.82.89/git/calculet/calBLAS.git & \
  git clone -q http://115.159.82.89/git/calculet/calDNN.git  & \
  git clone -q http://115.159.82.89/git/calculet/npu.git     & wait )
# => clone exit: 0；三个目录均生成
```

路径：`C:\Users\10463\Desktop\bohack\calculet-repos\{calBLAS,calDNN,npu}`
（等价于题目说的“同一目录下”，后续命令按 `cd calculet-repos/calBLAS` 执行即可。）

### 3.2 仓库 provenance（✅ 本机实测）

| 仓库 | commit | 作者日期 | 提交主题 | origin |
| --- | --- | --- | --- | --- |
| calBLAS | `32d2bb83d7160a593eaab67aa02e5dddbc8706cf` | 2026-11-13 14:42 +0800 | `calculet 黑客松下发版 2026-11-13` | `http://115.159.82.89/git/calculet/calBLAS.git` |
| calDNN | `1e06c41c557a469a9acc7566acb9f7e2c7f7ca62` | 2026-11-13 13:56 +0800 | `接入 calculet cargo 源：publish 到 calculet registry，路径依赖带版本` | `http://115.159.82.89/git/calculet/calDNN.git` |
| npu | `172aad2611c86ead4071356fbaca3cf6facd2917` | 2026-11-13 13:56 +0800 | 同上 | `http://115.159.82.89/git/calculet/npu.git` |

- 三仓库均**无 `.gitmodules`、无子模块**（`git submodule status` 为空）。
- 三仓库的 `.cargo/config.toml` 都指向同一个匿名可读的 calculet cargo 源：
  `index = "sparse+http://115.159.82.89/git/api/packages/calculet/cargo/"`（`npu/.cargo/config.toml:2-3`、`calDNN/.cargo/config.toml:2-3`、`calBLAS/.cargo/config.toml:2-3`）；实测该源 `config.json` 返回 HTTP 200（可匿名访问）。
- 三仓库的 `rust-toolchain.toml` 都是 `channel = "calculet"`（`npu/rust-toolchain.toml:3` 等）。

### 3.3 目录概览（规模感）

| 仓库 | 内容 | 关键位置 |
| --- | --- | --- |
| `npu` | kernel 编写基础库：类型、原语、`#[npu::kernel]`/`#[npu::func]`/`#[npu::program]`、proc-macro | `npu/src/lib.rs`（660 行）、`npu/macros/src/lib.rs`（309 行）、`npu/SYNTAX.md`、`npu/examples/{ops,prog}` |
| `calDNN` | 图 API / 引擎接口 / 计划（Graph / Engine / Plan / Job） | `calDNN/src/{graph,engine,plan,job}.rs`；`graph.rs` 914 行 |
| `calBLAS` | kernel 模板 + 引擎表 + 模型例子（最多） | `calBLAS/src/{engines.rs, frag.rs, shape.rs, kernels/*.rs}`、`calBLAS/examples/*`（约 20+ 个例子） |

---

## 4. 编译验证（第四阶段）✅ 已解决（WSL2，见 §15/§16）

> **更新**：以下“无法开始”是 Windows 原生环境的结论；在 WSL2 + calculet 工具链下 tiny 已编译成功（§16.2）。

### 4.1 按题目命令执行的真实输出（✅ 本机实测）

证据：`_recon/phase4_build_attempt.txt`。

```bash
$ cd calculet-repos/calBLAS
$ cargo build -p tiny --lib --target dopamine-unknown-none-elf
/dev/stdin: line 1: cargo: command not found          # exit=127

$ ls target/dopamine-unknown-none-elf/npu/
ls: cannot access 'target/dopamine-unknown-none-elf/npu/': No such file or directory   # exit=2

$ ls target/
ls: cannot access 'target/': No such file or directory   # exit=2
```

### 4.2 结论：tiny 未编译成功 —— 但原因是“没有编译器”，不是“编译报错”

- **是否生成 `tiny.cvxo` / `tiny.cvx`：❌ 未生成**（连 `target/` 目录都不存在）。
- **是否遇到 Rust 编译错误、可以分析日志并做最小修复：❌ 没有真实编译日志**。本机唯一的“真实日志”是 `cargo: command not found`。因此**不能、也不应该**对 `tiny` 或模板做任何“修复”——那会变成凭空猜测。任何“修复”都必须等工具链可用后、以真实编译器输出为准。
- 已知会造成真实编译失败的高风险点（⚠️ 待实机验证，仅列为检查单，不是结论）：
  1. `channel = "calculet"` 的工具链必须已安装（§2）；
  2. 三仓库必须同级（`calBLAS/Cargo.toml:14-17` 用 `path = "../calDNN"`、`path = "../npu"`）；
  3. cargo 源可访问（匿名 sparse registry，§3.2）；
  4. `target` 参数与工具链版本匹配：文档里 `dopamine-unknown-none-elf` 与 `dopamine-unknown-none-cvx` 两个 target 都有，且语义在文档间有冲突（见 §4.3）。

### 4.3 文档口径冲突：`-elf` 到底出什么产物？（⚠️ 必须实机复核）

三份文档对同一命令的描述互相矛盾，**不能凭文档断言产物名**：

| 来源 | 说法 |
| --- | --- |
| 赛题页 / `_recon/quickstart.md:29-33` | `cargo build -p tiny --lib --target dopamine-unknown-none-elf`；`ls target/dopamine-unknown-none-elf/npu/` ⇒ `tiny.cvxo`、`tiny.cvx` |
| `npu/SYNTAX.md:136` | `cargo build --target dopamine-unknown-none-cvx --lib` ⇒ `<target>/dopamine-unknown-none-cvx/npu/<crate>.cvxo + .cvx` |
| `npu/SYNTAX.md:140` | “`dopamine-unknown-none-elf` 现在同样只出 CalVX（警告一句）：ELF 要 docc 收 CalVX 1.0” |
| `calBLAS/README.md:84,90` | `-elf` 目标出 “CWX + 每个 kernel 的两核 ELF”（`<kernel>/ccu0_chip0.elf` 等）——**旧口径** |
| `calBLAS/STATUS.md:17-19` | 工具链换 CalVX 之后“**一个 ELF 都没出**：docc calt 读不进 `calvx.view.dyn_bind` / `calvx.sys.query`” |

**结论**：`tiny` 编译成功时应生成 `tiny.cvxo`（+ 可能 `tiny.cvx`），这是**文档口径，本机未验证**；`-elf` 目标的真实产物集合必须以实机 `ls` 为准。若在 Linux 上编译，第一件事就是确认这一点并记录。

### 4.4 上机后首查清单（环境类，不含任何 API 猜测）

1. `rustup toolchain list` 是否出现 `calculet`；`rustc +calculet --version` 是否 `1.98.1` 附近。
2. `docc --version` / `dort --version` 是否可执行。
3. 三仓库同级、`.cargo/config.toml` 与 `rust-toolchain.toml` 未被改坏。
4. 先跑文档给出的最小命令 `cargo build -p tiny --lib --target dopamine-unknown-none-elf`，**完整保存 stdout/stderr**；若失败，按真实错误逐项排查（工具链 → 依赖 → target 名 → 模板）。
5. 再跑整程序：`cargo build -p tiny --lib --features step --target dopamine-unknown-none-cvx`（`calBLAS/AGENTS.md:50`）。

---

## 5. 已有算子列表（第五阶段 1）

### 5.1 引擎/图算子（calDNN 可编排、可被图调用）

`calBLAS/src/engines.rs` 用宏 `engines!` 定义了 **44 个引擎项**（`calBLAS/src/engines.rs:19-41` 宏体，`:45-131` 的 44 条注册；`grep -c "Notes::" = 44`，✅ 已验证）。它们覆盖：

- **厂商 30 个图算子（README 自称全覆盖，decode 形态）**：embedding、rms_norm、l2_norm、layer_norm、matmul_add（bf16/bf16_hf/W8/W4/fp8/i8，前四种可带 bias）、multiply、add_relu、relu、silu_mul、mul_silu、mul_sigmoid、weighted_softplus、silu、sigmoid、gelu、rope、scatter_kv_cache、flash_attention、conv（depthwise 因果一维+SiLU）、gated_deltanet、moe_gate_pre、moe_fused_glu、moe_dense_group、moe_gate_post、moe_shared_gated_add、all_gather、all_gather_experts、broadcast、mat2_hwptn、hwptn2_mat。
- **厂商没有、自研的 13 项**：sparse_index、sparse_decode、hc（mHC 6 种）、compress、lightning_index、sink_attn（3 种）、DiT/多行 linear、attention、binary(silu_mul_hf)、dit_*、vit_* 等（见 §5.2）。

关键源码：`calBLAS/src/kernels.rs:6-24`（模板—Desc 一览表）、`calBLAS/src/engines.rs:45-131`、`calBLAS/src/lib.rs:29-45`（`entry!` = `(名字::DESC, 名字)`）。

**30 个厂商算子对照**（✅ 已验证：README 表 + 宏定义 + 引擎注册三处一致）：

| 算子 | 模板 | 引擎（`engines.rs` 行） | 备注 |
| --- | --- | --- | --- |
| embedding | `embedding!` | `Embedding` :45 | id→查表 |
| rms_norm | `rms_norm!` | `RmsNorm` :47 | 可带 offset |
| l2_norm | `l2_norm!` | `L2Norm` :49 | |
| layer_norm | `layer_norm!` | `LayerNorm` :51 / `LayerNormBias` :53 | |
| matmul_add | `linear!` | `MatmulBf16/W8/W4/Fp8` + Bias :55-67 | 多种量化 |
| multiply / add_relu / relu / silu_mul / mul_silu / mul_sigmoid / weighted_softplus | `binary!` | `Binary` :71 | |
| silu / sigmoid / gelu | `unary!` | `Unary` :69 | |
| rope | `rope!` | `Rope` :73 | |
| scatter_kv_cache | `scatter_kv!` | `ScatterKv` :75 / `ScatterKvPaged` :77 | |
| flash_attention | `flash_decode!` | `Sdpa` :79 / `SdpaPaged` :81 | 两遍 online softmax |
| conv（因果一维） | `causal_conv1d!` | `CausalConv1d` :109 | LLM 用，不是 2D |
| gated_deltanet | `gated_deltanet!` | `GatedDeltanet` :111 | |
| moe_*（5 个） | `moe_gate_pre!` 等 | :113-125 | |
| all_gather / all_gather_experts / broadcast | `all_gather!` 等 | `Gather`/`Broadcast` :127-129 | 跨芯粒 |
| mat2_hwptn / hwptn2_mat | `mat2_hwptn!` 等 | `Hwptn` :131 | |

### 5.2 “不进图”的模板（有 kernel、无 Desc/引擎）

`calBLAS/src/kernels.rs:21-24` 明确标注了多行“（不进图）”。其中与 CV 相关的（全部 ✅ 已验证存在于源码，但**没有一个例子实例化/上板**）：

| 类别 | 模板（`calBLAS/src/kernels/vision.rs` 行号） | 说明 |
| --- | --- | --- |
| 2D 卷积 | `conv2d!` :97 | k=1/3，s=1/2，act=silu/none，可加 bias/残差；NHWC、四周 2 像素边框；要求 CI/CO 为 16 的倍数（:119） |
| 深度卷积 | `dwconv3!` :288 | 3×3、s=1、padding 1 |
| 池化 | `maxpool5!` :354 | 5×5、s=1、padding 2（YOLO SPPF 用） |
| **最近邻上采样** | **`upsample2!` :392** | **固定 ×2**，NHWC 带边框，bf16，C≤256 |
| NCHW↔NHWC | `nchw_to_nhwc!` :434 / `nhwc_to_nchw!` :479 | 给框架（torch NCHW）接卷积模板用；零实例 |
| GroupNorm | `group_norm!` :524 | NCHW `[C][HW]`，两遍 f32 均值/方差，可带 silu；**零实例、零主机参考** |
| 转置 | `transpose2d!` :605 | 32 对齐的 2D 转置；零实例 |
| 填充 | `fill16k!` :638 | 边框清零/填 −MAX |
| 空间注意力 | `spatial_attn!` :664 | YOLO PSA 用 |

其他不进图模板：`dit.rs`（6 个，AdaLN/门控/残差流）、`prefill.rs`（8 个 `p_*`）、`encoder.rs`（`vit_gather!`/`vit_mean4!`）、`linear.rs` 的 `act_quant!/fp8_relayout!/bf16_relayout!`、`layout.rs` 的 10 个辅助宏。

### 5.3 覆盖结论 + README 自述未完成项（✅ 源码阅读）

- README 的“还没写”（`calBLAS/README.md:110-114`）原文：编码器多 token 注意力（laya）、prefill 其余算子（多 token 的 embedding/rope/scatter_kv/flash）、M-RoPE 多模态分路；分页 KV 只接 qwen38；动态 KV 没搬；融合引擎没做。
- 也就是说：**30 个图算子在 decode 形态下是全覆盖的**，缺的是 prefill/multi-token/多模态/融合。不要误以为“只剩 CV”。

### 5.4 明确缺失的算子（全仓库 grep 零命中，✅ 已验证）

在 `calBLAS` + `calDNN` + `npu` 三仓库（含 examples/部署脚本）中，以下关键字**零命中**：

`resize`、`interpolate`、`bilinear`、`bicubic`、`align_corners`、`antialias`、`scale_factor`、`conv_transpose`/`ConvTranspose`/`deconv`、`conv3d`、`dilated`/`dilation`、`avg_pool`/`adaptive_avg`、`grid_sample`。

（`letterbox` 的唯一命中是 YOLO 部署脚本里的主机 Python：`calBLAS/examples/yolo26/deploy/ref.py:8,18`；`NMS` 只有 "NMS-free 检测头" 的说明性文字（`examples/yolo26/deploy/DEPLOY.md:3` 等），并**没有 NMS 实现代码**；后处理在主机 Python 里，且该模型的头是 NMS-free，所以没有 NMS。）

### 5.5 关键核查：Nearest Resize 与 Bilinear Resize 是否已实现？

**这是本报告最重要的技术结论之一。**

#### Nearest Resize：**部分存在**（固定 ×2 的 NPU kernel）

- 定义：`calBLAS/src/kernels/vision.rs:390-425`，文档注释原文：
  > “最近邻上采样 ×2：输入 `[H + 4][W + 4][pin]` 的 c 个通道 ⇒ 输出 `[2H + 4][2W + 4][pout]`；每个输入行读一次、写四次（行 2y、2y+1 × 像素 2x、2x+1）。c ≤ 256。”
- 实现：四个交错写视图 + 一次 `vcache` 行加载 + 4 次 store（`:407-421`），按核分行（`halves!(H)`，`:412`）。
- 使用：**YOLO26n 里实用了两次**：`calBLAS/examples/yolo26/src/kernels.rs:43,46`；主机模拟侧同语义：`examples/yolo26/deploy/sim.py:119` 的 `xin.repeat(2, 0).repeat(2, 1)`。
- 限制（很重要）：
  - **只支持 ×2**（整数倍固定），h/w/C 都是编译期常量；
  - **只支持**“NHWC + 四周 2 像素边框”的视觉布局，bf16，C ≤ 256；
  - **不进图**：没有 `desc!`、没有引擎、没有 calDNN `Op`（§5.6）。
- 真机验证：YOLO26n 于 **npu2 / 2026-10-30** 上板通过（`examples/yolo26/deploy/DEPLOY.md:6-7`，见 §8.1），因此“Nearest ×2”是**已被真机验证过的算子**。

#### Bilinear Resize：**不存在**（❌ 零命中）

- 全仓库 grep：`bilinear`、`interpolate`、`resize`、`bicubic`、`align_corners` 全部 0 命中；
- 没有任何分数坐标 / 权重插值的 kernel；
- 现有 CV 例子中的图片缩放全部在**主机 Python** 完成：
  - YOLO26n：`cv2.imread` + Ultralytics `LetterBox((640, 640))` + `/255` + HWC→CHW + `F.pad` + 主机 im2col（`examples/yolo26/deploy/ref.py:17-32`）；
  - MiniCPM-V 4.7：`PIL.Image` + HF `AutoProcessor`（`examples/minicpm_v47/deploy/vit.py:240-285`）。

**结论表**

| 算子 | 是否存在 | 形态 | 进图/引擎 | 真机验证 |
| --- | --- | --- | --- | --- |
| Nearest Resize（×2） | ✅ 存在 | `upsample2!`，NHWC 带边框，bf16，C≤256 | ❌ 不进图 | ✅ 随 YOLO26n 上板验证 |
| Nearest Resize（任意倍率/尺寸） | ❌ 不存在 | — | — | — |
| **Bilinear Resize** | ❌ 不存在 | — | — | — |

### 5.6 结构性缺口：CV 算子目前无法“通过引擎表注册、被图调用”

赛题“路线 A”要求：“算子能通过 calDNN 的引擎表注册、被图调用”（`_recon/site.txt:126`）。但实测发现：

1. `calDNN/src/graph.rs:405-451` 的 `pub enum Op` 是**封闭枚举，22 个变体，没有任何 CV 算子**（无 conv2d/upsample/pool/group_norm）；
2. `Graph` 的构造函数 `push/tmp/op` 是私有（`calDNN/src/graph.rs:507,512,534`），下游 crate **无法**自己加一个 Op；也没有插件/注册 API；
3. 所有 `vision.rs` 模板**都不 emit `desc!`**（没有 Desc），因此无法成为引擎表的 key（`calBLAS/src/engines.rs:34` 的匹配依赖 Desc 相等）；
4. 因此 **yolo26 例子干脆不依赖 calDNN**：`examples/yolo26/Cargo.toml:10-12` 只有 `calblas` + `npu`，作业表由 `src/model.rs` 自己生成，再用 `deploy/gen_step.py` 拼成整步程序。

**要做到“CV 算子进图”，必须跨仓库改 calDNN**：加 Desc 结构体 + `Op` 变体 + `Graph` 方法 + `lib.rs` re-export；再在 calBLAS 加引擎项 + 模板 emit `desc!`；再在 example 的 `ENGINES` 注册。工作量不小，**这是 48h 内最容易被低估的部分**（详见 §7.4、§11）。

---

## 6. kernel 编写方法与 tiny 示例（第五阶段 2/3/4）

### 6.1 `npu/SYNTAX.md` 要点（✅ 源码阅读 + 宏实现核对）

**（a）两种“设备侧函数” + 一个程序入口**

| Rust 写法 | CalVX 产物 | 关键事实（`npu/macros/src/lib.rs`） |
| --- | --- | --- |
| `#[npu::kernel(cores = 1|2, numerics = "ks1")] pub fn …` | `calvx.func` + `npu.kernel` 属性 | 只收 `cores = 1|2`、`target = "…"`（未使用）、`numerics = "ks1"`；**kernel 不能泛型、不能有返回值、参数只能是 `Ptr`/标量**（:37-56、:73-84） |
| `#[npu::func] fn …` | `calvx.func`（调用处单态化） | 不收参数；可以泛型、可以写在 impl 里（:150-167） |
| `#[npu::program(chiplets = N, cores = 1|2)] pub fn …()` | `calvx.program` 入口 | **不能带参数**、不能带 `npu.kernel`（:265-294）；整个模型 = decls + 全部 kernel + 入口，`--target dopamine-unknown-none-cvx` 一次编出 |
| `npu::decls! { 名: 类别, bytes = N[, double_buffer]; … }` | `calvx.mem.decl` | 类别只收 `input/output/weights/workspace/kv/ram`（:183-185）；调用 `名()` 得到 `Ptr<Ddr|Ram, u8>` |

**（b）核心类型/原语**（`npu/src/lib.rs`，均 ✅ 已验证）

- 空间：`Ddr`（全局）/ `Ram`（片上共享）；访问：`Read`/`Write`/`ReadWrite`。
- 句柄：`Ptr<S,E>`（:130-133）、`View<S,E,A>`（:138-139）、`VCache<E>`（:142）、`MCache<E>`（:147）。
- `Ptr`: `bind(...)`（:157）、`bind_strided(...)`（:162）、`at(u64)`（:169，>4 GiB 权重用）、`add(usize)`（:173）、`cast::<E>()`（:178）、`load/store`（:183-193）。
- `vcache::<E,_>([..])` / `mcache::<E,_>([..])`（:208,:213）。
- `VCache`: `rows`（:337）、`load/store`（:342/:347）、`copy/fill/iota`（:352-362）、逐元素 `add/sub/mul/div/min/max/and/or/…`（:366-372）、`fma`（:383）、`cmp`（:456）、`select`（:391）、`reduce`（:396）、`scan`（:401）、`sort`（:406）、`transpose/reshape/broadcast/pack/unpack`（:416-436）、`cvt`（:441）、`exp/exp2/log/log2/sqrt/rsqrt/recip/sin/cos/tanh` + `*_approx`（:445-446）等。
- `MCache`: `tile`（:472）、`load/load_v/store/fill`（:477-494）、`mma/mma_scaled`（:500-505）。
- 通信/同步/调试：`all_gather/broadcast/all_reduce/…`（:562-595）、`sync_cores()`（:597）、`trap/assert/stamp`（:614-624）、`chip_rank/core_rank/cur_seq_len/cycle` 等（:641）。
- 元素类型 11 种计算元素 + 4 种仅存储元素（SYNTAX.md §2）。

**（c）硬约束**（`npu/SYNTAX.md:118-130`，⛔ 表示禁止）

- 控制流只收结构化的：`for …` / `if`；**⛔ `while` / `loop` / `break` / `continue` / `match` / 在 for、if 里 return**。
- kernel 体内 ⛔：堆分配、`static`、trait object、函数指针、递归、`panic!`（用 `npu::assert`）、调用 npu 原语/`#[npu::func]` 之外的函数。
- kernel ⛔ 泛型（泛型放 `#[npu::func]`）。
- 数值合同：kernel 只有 `numerics = "ks1"`；`m.mma` 自动补 `rnu + aligned16`，f32 逐元素补 `rnu + finite_only`；程序级固定 `rne`。

### 6.2 tiny 示例真实源码（✅ 源码阅读）

`examples/tiny/` 是一个**只有 4 个源文件**的极小例子（`src/kernels.rs` 18 行、`src/layout.rs` 119 行、`src/lib.rs` 8 行、`src/step.rs` 132 行；另有 `pack/` 两个 host 工具）：

- **`src/lib.rs:1`**：“小模型：和 qwen36 走**同一套 kernel 模板**，尺寸缩到权重约 10 MB —— 功能与数值的快速回归。”
- **`src/kernels.rs:9-18`**：10 个模板实例：

```rust
calblas::embedding!(pub tiny_embedding: vocab = 512, hidden = 512);
calblas::rms_norm!(pub tiny_rms: rows = 1, n = 512, eps = 1.0e-6);
calblas::linear!(pub tiny_linear: k = 512, n = 512, w = w4(32));
calblas::linear!(pub tiny_lm_head: k = 512, n = 256, w = w4(32));
calblas::moe_gate_pre!(pub tiny_gate_pre: k = 512, experts = 16, group = 32, topk = 2);
calblas::moe_fused_glu!(pub tiny_glu: k = 512, n = 128, group = 32, experts = 16, topk = 2, act = silu_mul);
calblas::moe_dense_group!(pub tiny_dense: k = 256, n = 512, group = 32, experts = 16, topk = 2, chips = 2);
calblas::moe_gate_post!(pub tiny_gate_post: topk = 2, n = 512);
calblas::binary!(pub tiny_add: add, n = 512);
calblas::all_gather!(pub tiny_gather: n = 256, chips = 2);
```

- **`src/step.rs`**：整个模型一个 `#[npu::program(chiplets = 2, cores = 2)] fn step()`（:104-105）；`npu::decls!`（:26-33）声明 `in_input_ids`/`in_position_ids`/`out_outputs_0`/`w`/`act`/`l2`；层体在 `layer!` 宏（:52-102）；`probe!` 宏（:40-49，`probe` feature 开启时把中间量拷进输出槽，供板子/模拟器逐步对比）。
- **`src/layout.rs`**：权重/激活的字节偏移**设备端与主机端共用一份**（:1-4 明确“⛔ 各写一份”）；这是很值得抄的设计。

### 6.3 tiny 的设备遍 / 主机遍（⚠️ 文档口径，本机未跑）

```bash
# 设备遍（只 kernel，不带 step）：题目给的命令
cargo build -p tiny --lib --target dopamine-unknown-none-elf

# 设备遍（整个模型：一份 .cvx / .cvxo）
cargo build -p tiny --lib --features step --target dopamine-unknown-none-cvx   # calBLAS/AGENTS.md:50

# 主机遍：造确定性权重 + CPU 参照
cargo run -q --release -p tiny-pack -- <目录>                                  # 写 param_blk{0,1}.bin
cargo run -q --release -p tiny-pack --bin tiny-ref -- <目录> <token> [out.bin] # CPU 参照（不是逐位金标准）

# 模拟器 + 板子（AGENTS.md:52-58）
cvx run <目录>/step.cvxo --cores 2 --prompt 7 --gen 2 --set "0:@w=…" --set "1:@w=…"
docc calbin from-calvx <目录> -o <cal> && dort calbin run …
```

### 6.4 最适合改造成 CV 算子的模板：`upsample2!`（推荐）

**理由**（✅ 源码阅读）：

- 它是唯一已经实现“空间重采样”的模板：NHWC-DDR 寻址、四个交错写视图、按行分核（`vision.rs:405-412`）；
- 结构最简单：一次行加载 + 4 次 store（`:413-422`），没有矩阵乘/权重加载，扩展成 bilinear 只需在同样的视图上做 f32 加权；
- 已有真机使用记录（YOLO26n），说明其数据搬运模式在板上是通的。

**备选**：
- 纯逐像素算子（如 leaky_relu/hardswish）应改 `binary!`/`unary!`（`calBLAS/src/kernels/elementwise.rs:44/:5`）——它是分块 `[R,C]` + `cores = 2` 骨架，而且**会 emit `desc!`**，因此可以进引擎表（vision 模板不行）。
- 窗口类（pool/conv）可参考 `maxpool5!`/`conv2d!`。

### 6.5 新增一个算子的最小步骤（基于源码事实）

1. **写模板**：在 `calBLAS/src/kernels/vision.rs` 里新增 `#[macro_export] macro_rules! resize_bilinear { … }`（放 `upsample2!` 之后最自然），内部用 `#[npu::kernel]`，参数风格照 `upsample2!`（`h/w/c/pin/pout` 全是编译期常量）。需要多档常量时按仓库规矩放 `src/shape.rs`（`src/shape.rs:1` “都是 const fn，各模板共用，⛔ 在模板里各写一份”）。
2. **不需要改 mod/导出**：`src/kernels.rs:42` 已经 `mod vision;`，`#[macro_export]` 会提升到 crate 根（`calblas::upsample2!` 就是这样的）。
3. **实例化**：在某个 example 的 `kernels.rs` 里一行，例如 `calblas::resize_bilinear!(pub y26_up_2x_…: …)`（照 `examples/yolo26/src/kernels.rs:43` 的写法）。
4. **调用**：目前 CV 算子不进图，所以要在 `#[npu::program]` 里**直接调用**生成的 kernel fn（照 `examples/tiny/src/step.rs:60` 调 `tiny_linear(...)` 的方式），或者走 yolo26 的“作业表 + 整步程序”路线。
5. **要进图/引擎（加分项）**：见 §7.4，需要跨仓库改 calDNN。
6. **验证**：CPU 参考（numpy/torch）→ `cvx run` 模拟器 → `docc calbin from-calvx` → `dort calbin run` 真机，逐元素比对（§9、§11）。

### 6.6 仓库规矩（来自 `calBLAS/AGENTS.md`，务必遵守）

- ⛔ 用脚本把各 kernel 的 MLIR 文本拼成整步程序（`gen_step.py` 是历史残留，新的写法是 `npu::decls!` + `#[npu::program]`，`AGENTS.md:15-27`）；yolo26 的 `gen_step.py` 路线属于旧路线，不要在新代码里扩散。
- **`cores = 2` 的 kernel 调完必须 `sync_cores()`** 才能读输出；同址既当输入又当输出是禁止的（`AGENTS.md:30-42`）。
- 先在 `examples/tiny` 上试功能，⛔ 拿 35B 的 21 GB 去试（`AGENTS.md:44-47`）。
- 一次只动一个变量（`AGENTS.md:107`）。
- 模拟器不是金标准：KS1 的 `rsqrt`/`exp` 是硬件近似（1 ULP），⛔ 拿 `cvx run` 当真机逐位参照（`AGENTS.md:77-87`）。
- 提交说明末尾写 `Feature: <名字>`（`AGENTS.md:1-...`、calDNN/npu 的 AGENTS.md 同样要求）。

---

## 7. calDNN 图 API 示例（第五阶段 6）

### 7.1 分层与对象（✅ 源码阅读）

```
npu：显式存储空间、tile、原语、#[npu::kernel]、Launch
 └ calDNN：TensorDesc / Op / Graph + frontend 复合运算
   │      Engine / Heuristic(FirstFit) / Notes
   │      Plan（编排 + 工作区）/ VariantPack（张量 ⇒ 地址）⇒ Job
   └ calBLAS：引擎库（kernel 模板 + 每种运算一个引擎）
       └ examples/<模型>/：kernel 特化、引擎表、模型的图
```

（`calDNN/README.md:6-13`、`:18-29`）

- 图只说“算什么”：`Graph` 里放 `TensorDesc` + `Op`（形状描述 Desc + 读写的张量号 `[Uid; N]`）；
- `Engine` 说“怎么算”：从第 `at` 个运算起认一段，出 `Job`（`covers > 1` 即融合）；
- `Plan::new(&graph, engines, &FirstFit)` 逐段挑引擎、给虚拟张量分工作区；
- `Plan::bind(&VariantPack)` 把 Uid 换成真实地址，得到可发射的 `Job` 列表。

### 7.2 `Engine` 接口（`calDNN/src/engine.rs`，✅ 原文）

```rust
pub struct Notes(pub u32);            // NONE / BF16_REDUCTION / TWO_PASS_SOFTMAX / CROSS_CHIP (:10-20)
pub struct Step<'a> { pub covers: usize, pub jobs: Vec<Job<'a>> }        // :31-35
impl Step { pub fn one(job: impl Into<Job<'a>>) -> Self }               // :39-41
pub trait Engine {
    fn name(&self) -> &str;                                            // :45
    fn notes(&self) -> Notes { Notes::NONE }                           // :46-48
    fn step(&self, g: &Graph, at: usize) -> Option<Step<'_>>;          // :50
}
pub trait Heuristic { fn pick<'e>(&self, g: &Graph, at: usize, engines: &[&'e dyn Engine]) -> Option<Step<'e>>; } // :54-56
pub struct FirstFit { pub exclude: Notes }  // 跳过带指定备注的引擎，先列的优先 (:59-67)
```

所有引擎的实现只有一处（宏生成）：`calBLAS/src/engines.rs:19-41`，核心 4 行：

```rust
impl Engine for $name<'_> {
    fn name(&self) -> &str { stringify!($name) }
    fn notes(&self) -> Notes { $notes }
    fn step(&self, g: &Graph, at: usize) -> Option<Step<'_>> {
        let Op::$variant(d, $io) = g.ops()[at] else { return None };   // :34
        let &(_, kernel) = self.0.iter().find(|(k, _)| *k == d)?;      // :35  Desc 相等 = 分派键
        Some(Step::one(kernel($($arg.into()),*)))                      // :36  Uid→Ptr、Launch→Job
    }
}
```

一个具体引擎项（`calBLAS/src/engines.rs:45/71`）：

```rust
Embedding(EmbeddingDesc, Embedding, Notes::NONE) fn(Ids, Bf, Bf) = |io| (io[0], io[1], io[2]);
Binary(PwDesc, Pointwise, Notes::NONE) fn(Bf, Bf, Bf) = |io| (io[0], io[1], io[2]);
```

### 7.3 图构建 / 计划编排 / 引擎表：真实代码

**图**（`calBLAS/examples/qwen3_8b/src/model.rs:38-58` 风格，✅ 源码阅读）：

```rust
let mut g = Graph::default();
let id  = g.input_sram(Elem::I32, &[16]);
let pos = g.input(Elem::I32, &[1]);
let table = g.weight_bf16(...);
let mut x = g.embedding(id, table);
for l in 0..self.layers { x = Self::layer(&mut g, x, ...); }
let logits = g.matmul(h, lm, None);
g.output(logits);
(g, Io { id, pos, logits }, names)
```

**计划**（`qwen3_8b/src/main.rs:8-12`、`:45`）：

```rust
let (g, io, names) = Qwen3 { layers }.graph();
let plan = Plan::new(&g, ENGINES, &FirstFit::default())?;
...
let jobs = plan.bind(&VariantPack { ptrs: &ptrs, workspace: next, sram: 0x0800_0000 })?;
```

**引擎表**（`qwen3_8b/src/engines.rs:7-16`，✅ 源码阅读）：

```rust
pub const ENGINES: &[&dyn Engine] = &[
    &Embedding(&[entry!(k::qwen3_embedding)]),
    &MatmulW4(&[entry!(k::qwen3_linear_4096x4096), ...]),
    // …
];
```

`entry!(path)` = `(path::DESC, path)`（`calBLAS/src/lib.rs:41-45`）；`desc!(...)` 由模板宏发出（如 `norm.rs:8`、`elementwise.rs:7`），**vision 模板一个都没有**。

### 7.4 新增一个“可进图”算子的确切改动清单（跨仓库）

| # | 仓库/文件 | 改动 |
| --- | --- | --- |
| 1 | `calDNN/src/graph.rs` | 新增 `#[derive(Clone, Copy, Debug, PartialEq)] pub struct XxxDesc { … }`（f32 字段用 `F32` 包裹以保证 `Eq`，:288-308）；在**封闭枚举** `pub enum Op` 加 `Xxx(XxxDesc, [Uid; N])`（:405-451）；加 `Graph::xxx(...)` 方法（构造 Desc → `self.op(...)` → 用 `self.tmp` 分配虚拟输出）；`calDNN/src/lib.rs` 里 re-export |
| 2 | `calBLAS/src/kernels/…` | 模板宏里 emit `desc!(名字: XxxDesc { … })`（字段必须与图构造值**完全相等**，引擎用 `==` 查表，`engines.rs:35`） |
| 3 | `calBLAS/src/engines.rs` | 在 `engines!` 里加一行 `Xxx(XxxDesc, Xxx, Notes::… ) fn(…) = |io| (…);` |
| 4 | example `src/engines.rs` | 在 `ENGINES` 常量里加 `&Xxx(&[entry!(k::your_kernel)])` |
| 5 | example `src/model.rs` | 用新的 `g.xxx(...)` 构图 |
| 注意 | — | kernel 参数 ≤ 8 个（`calDNN/src/job.rs:50-65` 的 `Launch→Job` 只生成到 8 元组）；`Graph::op` 是私有，下游无法绕开 |

🧠 工作量评估：这套改动对熟手是半天到一天，对“经验较少”的 3 人队是 1.5～2 天且容易卡在 Desc 字段相等/编译错误上。**建议把“图接入”作为 48h 的第二个里程碑，而不是第一个**（§11）。

---

## 8. 已有 NPU 真机运行测试（第五阶段 7）

### 8.1 有真机数值记录的例子（✅ 源码阅读：仓库内文档记录，非本机执行）

| 例子 | 板卡/日期 | 记录（原文要点） | 证据 |
| --- | --- | --- | --- |
| **YOLO26n**（唯一的 CV 端到端） | **npu2，2026-10-30** | bus.jpg 的 5 个检测与 torch f32 相同（bus 0.926、person 0.916/0.901/0.856/0.520，框差 ≤ 2 px）；三层原始输出对 torch 余弦 ≥ 0.99974；一次约 40 ms（未优化） | `calBLAS/examples/yolo26/deploy/DEPLOY.md:6-7` |
| Qwen3-8B-FP8（LLM） | npu2，2026-10-30 | 贪心 token 与 CPU 参照逐个相同；npu3 上另有版本与耗时 | `examples/qwen3_8b/deploy/DEPLOY.md:8,111-125` |
| Qwen3.6-35B-A3B | npu3 | 从 token 760 贪心 9 步，参照 `1156 369 18816 310 1791 264 4145 1957 421` | `calBLAS/AGENTS.md:117-119` |
| examples/tiny | npu3 | 两核屏障 bug 修复后 `cvx run` 三次一致、与 `tiny-ref` 512/512 逐位相同；npu3 prefill 挂死问题被二分定位 | `calBLAS/AGENTS.md:41-42,60-62,72-75` |
| DeepSeek-V4.1-Flash 1 层 4 芯粒 | npu2，2026-10-29 | docc 编、`dort calbin run` 跑完一步（6 个 all_gather 都过） | `examples/deepseek_v41_flash/model/README.md:10` |
| DeepSeek-V4.1-Flash 算子验收 | npu-03（16 芯粒卡） | round1 386/390 逐位、round2 4/4、round3 97 ✓、round4 150 ✓/10 ≈/7 ✗ | `examples/deepseek_v41_flash/deploy/BOARD-16CHIP.md:44-55` |
| vuprobe（近似指令标定） | **npu1 与 npu5 两块板** | `rsqrt` 精确命中 290/512、`exp` 429/512、`1/x` 512/512，均 ≤ 1 ULP，且两块板逐字节相同 ⇒ 硬件稳定性质 | `calBLAS/AGENTS.md:77-88`；程序 `examples/vuprobe/src/step.rs` |

> 标定表的意义：**带 `rsqrt`/`exp` 的模型，模拟器与真机不可能逐位相同**；而 `1/x`（除法）是精确的。做 Resize（只有乘/加）反而不受这个影响。

### 8.2 明确“未上板/未验证”的例子（✅ 源码阅读）

- `calBLAS/STATUS.md:26`：“2026-10-24，本机；数值都没有验证（未上板）”；`calBLAS/README.md:93` 同义。
- minimax_h3（DiT/文本/视频 VAE）：`examples/minimax_h3/README.md:56`“**全部没有上板、没有数值验证**：现在只到 calbin 编得出来。”
- d3_turbo：`examples/d3_turbo/README.md:56`（step 5“没有出过 calbin、没上板”）、`:58`（上板一栏空白）。
- MiniCPM-V 4.7：有板卡驱动脚本（`examples/minicpm_v47/deploy/vit_run.py`、`mcv_llm.py`）与 npu2 部署标题，但**没有数值结果记录**（⚠️）。

### 8.3 板卡健康检查与恢复（⚠️ 文档流程，本机无板）

`calBLAS/AGENTS.md:91-101` 原文要点：

```bash
dort device -i 0 | grep "Dev  :"      # 要 OS up   RT up
# 不是 RT up ⇒ dort device reprobe -i 0 --yes（SOFT_RESET，只有这一步重启板上 RT）
# ⇒ 等 15 秒 ⇒ sudo rmmod calculet_pci && sudo insmod ~/cal-pcie-driver/calculet_pci.ko
# （reprobe 后设备节点会变成 /dev/calculet1，这一步归位）⇒ 等 RT up
```

⛔ 只做 `rmmod/insmod` 想救崩掉的板子会一直停在 `OS up / RT DOWN`、所有件报 `configure 失败：NoDevice`（`AGENTS.md:100-101`）。
⚠️ 16 芯粒验收侧更严格：**不要自己 reprobe/复位**，交给人决定（`examples/deepseek_v41_flash/deploy/BOARD-16CHIP.md:103`）。

### 8.4 真机测试脚本资产（说明仓库里已有成熟流程）

- yolo26：`deploy/{gen_step.py,sim.py,ktest.py,cmp.py,run.py,ref.py,pack.py}`（整步生成、numpy 模拟、单 kernel 板测、探针对比、pydort 上板、torch 参照、权重打包）。
- qwen3_8b：`deploy/{gen_step.py,ref.py,pack.py,chat.py,chat.sh}`（含一键对话，KV 常驻多轮）。
- tiny：`pack/{main.rs,reference.rs}`（确定性权重 + CPU 参照）。
- deepseek_v41_flash：`deploy/board_*.py`、`gen/board_runner.py`、`deploy/bridge_client*.py` 等一整套上板/验收/桥接工具。
- `tools/bench/gen.py`：逐算子计时（但 `AGENTS.md:26` 说它是“脚本拼 MLIR”的残留，新代码不要学）。

### 8.5 结论

- ✅ 仓库确实有**成规模的真机测试资产和数值记录**，尤其是 YOLO26n（CV）和 Qwen 系列（LLM）。
- ❌ 本机没有板卡、没有 dort、没有驱动，无法进行任何真机验证；真机还涉及排队（赛题页：“真机资源有限，存在排队延迟”，`_recon/site.txt:36`）。
- 🧠 对 48h 选题的含义：**照抄 yolo26 的“整步程序 + docc calbin + dort run”路线，是风险最低的上板方式**；不要一上来就指望 calDNN Plan 发射（README 明确说“没有发射过”，`calDNN/README.md:35`）。

---

## 9. docc / dort 真实调用示例（第五阶段：命令取证）

> 以下命令**全部来自仓库内文档/脚本**（已逐条给 `文件:行`），**不是本机执行结果**（本机没有 docc/dort，❌）。路径里的 `../docc/…`、`../calculet-calvxus/…` 说明 docc 与 calvxus 的源码/二进制**不在**这三个仓库里，由工具链或另行提供。

### 9.1 分组命令表

**（a）设备遍构建 / 查看产物**

| 命令（原文） | 证据 |
| --- | --- |
| `cargo build -p tiny --lib --features step --target dopamine-unknown-none-cvx` | `calBLAS/AGENTS.md:50` |
| `cargo build -p yolo26 --lib --target dopamine-unknown-none-cvx` | `examples/yolo26/deploy/DEPLOY.md:28` |
| `cargo build -p qwen3-8b --lib --target dopamine-unknown-none-cvx` | `examples/qwen3_8b/deploy/DEPLOY.md:41` |
| `cargo build --workspace --lib --target dopamine-unknown-none-elf` | `calBLAS/README.md:84`（旧 CWX/ELF 口径，见 §4.3） |
| `../calculet-calvxus/target/release/cvx mlir target/dopamine-unknown-none-cvx/npu/yolo26.cvxo -o /tmp/y.mlir` | `examples/yolo26/deploy/DEPLOY.md:29` |
| `cvx dis <crate>.cvxo` / `cvx mlir <crate>.cvxo` / `cvx asm <mlir> -o <cvx>` / `cvx verify` | `npu/SYNTAX.md:137`、`examples/qwen3_8b/deploy/DEPLOY.md:43,52`、`examples/d3_turbo/README.md:56` |

**（b）calbin 打包 / 检查**

| 命令（原文） | 证据 |
| --- | --- |
| `docc calbin from-calvx <目录> -o <cal>`（目录里放 `step.cvx` + `param_blkN.bin`） | `calBLAS/AGENTS.md:54`、`examples/yolo26/deploy/DEPLOY.md:35` |
| `../docc/target/release/docc calbin from-calvx /tmp/q3/qwen3_8b_fp8_decode --no-params --max-seq-len 4096 -o /tmp/q3/cal` | `examples/qwen3_8b/deploy/DEPLOY.md:56` |
| `docc calbin from-calvx <目录>/<phase> -o <out> $(cat from-calvx.args)` | `examples/qwen36_35b_a3b/deploy/src/main.rs:14`（生成参数：`--network … --max-seq-len … --ddr … --sram …`） |
| `docc calbin doctor <calbin 目录>`（拒收 0 条才算过） | `examples/deepseek_v41_flash/deploy/README.md:75`、`deploy/board_sweep.py:303` |
| `docc calbin diff`、`docc calbin from-calvx`（赛题页提到的子命令） | `_recon/site.txt:76-77`（赛题页工具链表） |

**（c）板卡健康**

| 命令（原文） | 证据 |
| --- | --- |
| `dort device -i 0 \| grep "Dev  :"`（要 `OS up   RT up`） | `calBLAS/AGENTS.md:94` |
| `dort device reprobe -i 0 --yes`（SOFT_RESET，仅这一步重启 RT） | `calBLAS/AGENTS.md:97` |
| `dort device` / `dort processes`（多芯粒/16 芯粒场景） | `examples/deepseek_v41_flash/deploy/BOARD-16CHIP-PROMPT.md:24,27,43` |

**（d）上板执行（按模型类型）**

| 命令（原文） | 证据 |
| --- | --- |
| `dort calbin run -i 0 -f cal -m yolo26 --raw-in input.bin --raw-out out_y.bin --in-tensor in_x --out-tensor out_y` | `examples/yolo26/deploy/DEPLOY.md:44` |
| `PYTHONPATH=<pydort>:<numpy> python3 run.py cal input.bin out_y.bin 5`（pydort 路线） | `examples/yolo26/deploy/DEPLOY.md:43`、`deploy/run.py` |
| `dort calbin run -i 0 -f cal -m <模型> -d <模型> --tokens 760 -s 9 --in-tokens in_input_ids --in-positions in_position_ids --out-logits out_outputs_0` | `calBLAS/AGENTS.md:118-119` |
| `dort calbin run -i 0 -f . -m qwen3_8b_fp8_decode -d qwen3_8b_fp8_decode --in-tokens in_input_ids --in-positions in_position_ids --out-logits out_outputs_0 --tokens 785 -s 24` | `examples/qwen3_8b/deploy/DEPLOY.md:75-80` |
| `dort calbin run -m bench -d bench --tokens 1,2,3,4,5 -s 2 --in-tokens in_ids --in-positions in_pos --out-logits out`（逐算子计时） | `calBLAS/README.md:99` |

**（e）读回 / 模拟器**

| 命令（原文） | 证据 |
| --- | --- |
| `dort mem read -a 0x091c0000 -s 16384`（读 profbuf 用时；`-s` 是字节数） | `calBLAS/README.md:100` |
| `dort mem read -a <obuf 的 vaddr> -s <字节>`（vaddr 在 calbin 的 `submodel_memory_reserved_info.txt`） | `calBLAS/AGENTS.md:57-58` |
| `cvx run <目录>/step.cvxo --cores 2 --prompt 7 --gen 2 --set "0:@w=<目录>/param_blk0.bin" --set "1:@w=<目录>/param_blk1.bin"` | `calBLAS/AGENTS.md:52-53` |
| `cvx run … --dump-decl 0:@out_outputs_0=<文件>`（把中间量 dump 出来对比） | `calBLAS/AGENTS.md:57` |

### 9.2 端到端命令链（YOLO26n，仓库里的标杆 CV 路线）

来自 `examples/yolo26/deploy/DEPLOY.md:25-46`（原文顺序整理）：

```sh
# 1) 设备遍：编出 CalVX
cargo build -p yolo26 --lib --target dopamine-unknown-none-cvx
../calculet-calvxus/target/release/cvx mlir target/dopamine-unknown-none-cvx/npu/yolo26.cvxo -o /tmp/y.mlir

# 2) 主机遍：生成作业表 + 把各 kernel 拼成整步程序（旧路线，新代码别学）
cargo run -q -p yolo26 -- --jobs > /tmp/jobs.txt
python3 examples/yolo26/deploy/gen_step.py /tmp/jobs.txt /tmp/y.mlir /tmp/y26/yolo26
../calculet-calvxus/target/release/cvx asm /tmp/y26/yolo26/step.calvx.mlir -o examples/yolo26/yolo26n.cvx

# 3) 打包参数块 + calbin
python pack.py yolo26n.pt /tmp/y26/yolo26/w.map /tmp/y26/yolo26/param_blk0.bin
../docc/target/release/docc calbin from-calvx /tmp/y26/yolo26 -o /tmp/y26/cal    # ~1 分钟

# 4) 上板 + 对比
PYTHONPATH=<pydort>:<numpy> python3 run.py cal input.bin out_y.bin 5
# 或：dort calbin run -i 0 -f cal -m yolo26 --raw-in input.bin --raw-out out_y.bin --in-tensor in_x --out-tensor out_y
python ref.py yolo26n.pt <图片> <目录> out_y.bin io.map      # 对 torch、出检测
```

> 仓库里直接放了一份已核对过的整步程序 `examples/yolo26/yolo26n.cvx`（24,373 行文本，`mem.decl` + 84 种 kernel + `@step` + 111 个作业），DEPLOY.md 说它与上板那份 ELF 逐字节相同（`DEPLOY.md:38`）。

### 9.3 注意

- **docc/dort 不在三个仓库里**；`install.sh` 只负责把工具链里的二进制链到 `~/.cargo/bin`。
- 命令中的 `-f/-m/-d/-s/-i/-tokens/--in-tensor/--out-tensor/--raw-in/--raw-out` 语义以仓库文档为准，**本机未实测**。
- 不同模型/例子的参数不同（LLM 用 tokens/logits，CV 用 raw-in/raw-out，多芯粒用不同 calbin 目录），不要跨模型照搬到新算子；新算子建议走 `tiny` 风格的单程序（`npu::decls!` + `#[npu::program]`），再按 §9.1(e) 的 `cvx run --set/--dump-decl` 对比。
- `docc calbin doctor` 是“打包后自检”，建议每次 `from-calvx` 后都跑（拒收 0 条）。

---

## 10. 建议开发的 3 个缺失算子

> 判定标准：**缺失有源码证据** + **48h 可交付** + **能上真机**。以下排序即推荐优先级；第 3 个是“替补/加分”，不是必须。

### ① Bilinear Resize（任意输出尺寸）——首推

- **现状**：❌ 完全不存在（§5.5；`bilinear/interpolate/resize` 三仓库 0 命中）。Nearest 只有固定 ×2。
- **为什么需要**：Stable Diffusion 的 latent 上采样 / img2img 尺寸适配、SAM/CLIP 预处理、多尺寸输入，都需要双线性/双三次插值；赛题要求“varying 尺寸输入、不依赖 CPU 兜底”（`_recon/site.txt:20,128-132`）。
- **为什么适合新手**：纯“取两点 + 加权 + 舍入”，没有矩阵乘、没有权重加载、没有跨核复杂同步；数据搬运模式可以直接抄 `upsample2!`（`vision.rs:390-425`）。
- **实现要点**（全部基于真实 API 用法）：用 `Ptr::bind_strided`/`View::sub` 定位输入/输出（`vision.rs:405-410`）；`vcache::<bf16,_>` 行加载（`:411`）；f32 里算 `(1-t)·a + t·b` 再 `cvt` 回 bf16（`Cvt::default()`）；`halves!(H)` 分核（`:87-91`）。**注意**：整数坐标/权重是编译期还是运行时给出，要照 SYNTAX.md 的控制流约束（`for`/`if` 可以，`while` 不行）。
- **验收**：CPU 参考（numpy/torch，按 f32 算再 RNE 舍 bf16）逐元素对比；bf16 下目标 max|Δ| ≤ 2 ULP；模拟器与真机一致（Resize 不涉及 exp/rsqrt，理论上可逐位）。

### ② 通用 Nearest Resize（任意倍率 / 任意 H_out×W_out）——降级首选 / 第 2 个

- **现状**：⚠️ 部分存在：`upsample2!` 只支持 **×2**，h/w/C 全编译期常量，NHWC 带边框、bf16、C≤256，不进图。
- **价值**：实现难度最低（只有取整与复制），能在 48h 内稳稳交货；很多预处理（letterbox 到任意尺寸）需要它。
- **实现**：在 `upsample2!` 基础上把“固定 2”改成“编译期给定 `h_out/w_out`（或 `sx/sy`）”，索引用 `out_y * h / h_out` 这类整数运算（注意 kernel 里 ⛔ 浮点除法以外的复杂控制流；可先做“整数倍/整除”的受限版本，保证能编过）。
- **验收**：与 `np.repeat` 参考逐位相同；不同尺寸各测 2～3 组。

### ③ 首层 im2col / PatchEmbed 卷积上 NPU（k=s=14/16 的非重叠卷积）——加分项（可能来不及）

- **现状**：`conv2d!` 只支持 **k∈{1,3}、s∈{1,2}**（`vision.rs:93,97-119`），且要求通道数为 16 的倍数（`:119`）；于是 YOLO26n 的 RGB 首层被**在主机上做 im2col**（`examples/yolo26/deploy/ref.py:17-32`、`model.rs:4`），MiniCPM 的 ViT 预处理也在主机（`deploy/vit.py:240-285`）。
- **为什么算“缺失”**：SAM / CLIP / ViT 的 patch embedding 是 k=s=14/16 的非重叠卷积，目前没有对应 kernel，只能靠主机重排或拆成 matmul。
- **风险**：实现比 ①② 难，且需要与具体模型（SAM/ViT）绑定才能验收；48h 内建议只作为“时间富余时的 stretch”，不要作为主交付。

**备选（有证据但优先级低）**
- `GroupNorm` 的**真机验证与实例化**：模板已存在（`vision.rs:518-600`，注释里还点名“UNet 里最大的组 10×4096”），但**零实例、零主机参考**——它不是“缺失算子”，而是“缺失验证”；如果选题是 SD UNet，这是必做项（🧠）。
- `ConvTranspose2d / AvgPool2d`：仓库里 0 命中，但现有 CV 例子没有明确的需求证据（是否必要取决于目标模型架构，属外部知识，不是本仓库证据；SD 类 VAE 常见做法是 nearest×2 + conv，用已有模板即可拼）；48h 不建议做（🧠）。

---

## 11. 推荐的最终开发选题

### 11.1 选题一句话

**做 `Bilinear Resize`（先支持 ×2 上采样 + 任意 H_out×W_out 的通用版本之一），在 Calculet NPU 上完成“设备遍编译 → docc 打包 → dort 真机运行 → 与 CPU 参考逐元素比对”，并（时间允许时）把它注册进 calDNN 引擎表、被图调用。**

### 11.2 范围（做什么 / 不做什么）

**做（Must）**
1. `calBLAS/src/kernels/vision.rs` 新增 `resize_bilinear!`（或 `resize!`）模板，复用 `upsample2!` 的视图/分核骨架；
2. 一个最小的“整步程序”example（照 `examples/tiny` 或 `examples/vuprobe` 的结构：`npu::decls!` 输入/输出 + `#[npu::program]` 调 kernel），尺寸选小（如 32×32→64×64、16 通道）；
3. 主机侧：确定性输入生成 + CPU 参考（numpy/torch 双线性，f32 计算 + RNE 舍 bf16）；
4. 验证链：`cargo build … --target dopamine-unknown-none-cvx` → `cvx run`（模拟器，`--set/--dump-decl`）→ `docc calbin from-calvx` → `dort calbin run` → 逐元素误差报告。

**做（Should）**
5. 在 calDNN 加 `ResizeDesc`/`Op::Resize`/`Graph::resize`，calBLAS 加引擎项与 `desc!`，并在一个 example 的 `ENGINES` 注册（§7.4）——满足赛题“通过引擎表注册、被图调用”的要求。

**不做（Non-goals）**
- ❌ 完整 Stable Diffusion / 端到端模型（用户明确要求暂不做；且 SD 需要 CLIP 文本编码、UNet、VAE、调度器、ComfyUI 桥接，48h 不现实）；
- ❌ 不碰 LLM 算子的数值路径（风险高、价值与本选题无关）；
- ❌ 不重写 yolo26 的旧 `gen_step.py` 路线，新代码用 `npu::decls!` + `#[npu::program]`。

### 11.3 为什么是它（对比其他选项）

| 选项 | 缺失证据 | 48h 可行性 | 真机价值 | 结论 |
| --- | --- | --- | --- | --- |
| **Bilinear Resize** | ✅ 强（0 命中） | ✅ 高（纯 elementwise） | ✅ SD/SAM/预处理都需要 | **首选** |
| Nearest 任意倍率 | ⚠️ 只有 ×2 | ✅ 最高 | ✅ 预处理需要 | 降级/组合 |
| PatchEmbed/首层 im2col | ✅ 强（主机 im2col 证据） | ⚠️ 中低 | ✅ SAM/ViT 需要 | 加分项 |
| 完整 SD 移植 | — | ❌ 极低 | 极高 | 48h 不做 |

### 11.4 验收标准（可量化）

1. **编译**：`cargo build -p <example> --lib --features step --target dopamine-unknown-none-cvx` 成功，产物存在且 `.cvx` 可读；
2. **模拟器**：`cvx run` 输出与 CPU 参考逐元素对比，最近邻模式**逐位相同**；双线性模式 max|Δ| ≤ 2 ULP（以 bf16 计）；
3. **真机**：`docc calbin from-calvx` 后 `docc calbin doctor` 拒收 0 条；`dort calbin run` 成功；板读回与模拟器/参考一致（误差同 2）；
4. **（加分）图接入**：新算子能出现在 `Plan::new` 产出的 `Job` 列表里，图能被 `FirstFit` 成功编排；
5. **文档**：提交说明含 `Feature: Resize 算子`；README/模板表登记；给评委的复现实验命令一条龙。

### 11.5 风险与降级预案

| 风险 | 预案 |
| --- | --- |
| 环境（Linux/工具链）迟迟不可用 | Day 0 全员先解决环境（§14）；无法解决则立即联系组委会，同时把 CPU 参考/设计文档先做完 |
| `dopamine-unknown-none-cvx` 编译报错（文档冲突） | 以真实日志为准；先回到 `calBLAS/AGENTS.md:50` 的 tiny 全流程跑通，证明工具链健康，再编自己的 kernel |
| 真机排队/板子不可用 | 先交模拟器 + `cvx run` 证据；真机作为最后 12h 的必须项；提前预约板子 |
| bilinear 的坐标/权重控制流超出 SYNTAX 允许范围 | 降级到 Nearest 任意倍率（更简单，一定能编），把 bilinear 作为第二版 |
| 图接入（calDNN 跨仓库）改不动 | 保底：单算子整步程序真机验证（仍满足“可在真机运行的新算子”），图接入列为赛后 TODO |

---

## 12. 3 人团队下一阶段分工（48h 计划）

> 前提：先解决 §14 的 P0 环境问题（这是 Day 0 的第一件事，优先级高于写代码）。

### 12.1 角色

| 角色 | 职责 | 主要产出 |
| --- | --- | --- |
| **A：环境/构建/板卡 owner** | Linux 环境、rustup、`install.sh`、三仓库 clone、`tiny` 全流程编译、docc/dort、板卡健康检查与排队 | 可复现的环境脚本 + 编译日志 + 板卡可用性结论 |
| **B：kernel 实现 owner** | 读 `vision.rs`/`SYNTAX.md`，写 `resize_bilinear!` 模板与实例化、整步程序（`decls!`+`program`） | 可编译的 kernel 代码 + `.cvxo/.cvx` |
| **C：验证/精度 owner** | CPU 参考（numpy/torch）、确定性输入生成、逐元素对比脚本、模拟器/真机数据比对、结果记录 | 对比报告（误差表/图）+ 复现命令 |
| **Lead（队长）** | 需求拆解、接口/Desc 设计评审、代码 review、跨仓库改动协调、对组委会沟通、时间盒与风险升级 | 集成分支、最终提交、答辩材料 |

> 3 人经验较少时，**不要三个人同时改同一个文件**：A 管环境/编排，B 管 `vision.rs` 新模板，C 管 `examples/<新例子>` 与 Python 参考；Lead 做 integration 与 calDNN 改动（或指定 B 在 Day2 做）。

### 12.2 48h 时间盒（按小时）

| 时间 | 目标 | 负责 | 验收 |
| --- | --- | --- | --- |
| 0–6h | **环境打通**：Linux + rustup + install.sh + 三仓库 + `cargo build -p tiny --lib --target dopamine-unknown-none-elf` 成功；记录真实产物 | A 主，全员协助 | tiny 编译日志；产物 `ls` 记录 |
| 3–8h（并行） | B：读通 `vision.rs:390-425` + `SYNTAX.md`，写出 resize kernel 骨架；C：写 numpy 参考 + 输入生成器 | B/C | 代码骨架 + 参考脚本可跑 |
| 8–14h | 首次设备遍编译自己的 kernel；跑 `cvx run` 模拟器对比；修编译/语义问题 | B 主，C 验证 | `.cvxo` 生成；模拟器误差表 |
| 14–20h | **tiny 整程序模式**跑通（`--features step` + `npu::decls!`/`program`），把 resize 放进单程序 | A/B | 单程序 `.cvx` + 模拟器输出 |
| 20–30h | **首次上板**：`docc calbin from-calvx` → `doctor` → `dort calbin run`；对比板子输出 | A 主（板子），C 比对 | 真机误差报告 |
| 30–38h | 精度/边界修复（不同尺寸、通道 16 对齐、尾块）；性能可选 | B/C | 误差 ≤ 目标 |
| 38–44h | **图接入**（加分）：calDNN Desc/Op/Graph + calBLAS 引擎 + example ENGINES | Lead/B | `Plan::new` 能出 Job |
| 44–48h | 文档、提交（`Feature: Resize 算子`）、备份、演示排练 | 全员 | 提交 + 5 分钟 demo |

### 12.3 协作纪律（直接照抄仓库规矩）

1. 一次只改一个变量（`calBLAS/AGENTS.md:107`）；
2. `cores = 2` 的 kernel 之后必须 `sync_cores()`（`AGENTS.md:30-42`）；
3. ⛔ 用脚本拼 MLIR 文本当整步程序（`AGENTS.md:15-27`）；
4. ⛔ 拿 `cvx run` 当真机逐位金标准（`AGENTS.md:77-87`），但可用来抓结构性错误（挂死/越界/整段为零）；
5. 提交说明末尾写 `Feature: Resize 算子`（三个仓库统一）。

### 12.4 48h 结束时必须拿出的东西

- [ ] 一份可复现的环境/构建脚本（或在 README 写清步骤）；
- [ ] `resize_bilinear!`（或 `resize!`）模板 + 一个 example；
- [ ] 设备遍编译成功的 `.cvxo`（+ `.cvx`）；
- [ ] `cvx run` 与（必须）`dort calbin run` 的输出与 CPU 参考的逐元素对比结果；
- [ ] 如果做了图接入：`Plan::new` 输出 Job 的证据；
- [ ] 一份 3 分钟 demo 脚本 + 提交说明。

---

## 13. 当前尚未解决的问题

### P0：阻塞一切（必须先解决）

| # | 问题 | 证据 | 需要什么 |
| --- | --- | --- | --- |
| P0-1 | **本机不是 Linux**（Windows 11 + MSYS2） | §1.2 | ✅ **已解决**：WSL2 3.0.5 + Ubuntu 24.04.3 已装好并验证（§15/§16.1） |
| P0-2 | **calculet 工具链装不上** | §2.4 | ✅ **已解决**：calculet 工具链已安装（rustc 1.98.1 / docc 0.4.1 / dort 0.4.1），tiny 已编译（§16） |
| P0-3 | **无真机访问**：没有 dort、没有驱动、没有板卡分配机制/队列说明；`~/cal-pcie-driver/calculet_pci.ko` 之类只在文档里 | §2.5、§8.3 | 组委会提供板卡/排队方式、驱动安装说明、pydort |
| P0-4 | **三个仓库里没有 docc/dort/calvxus 的源码或二进制** | §9.3 | 工具链安装脚本会提供 docc/dort 二进制；calvxus 需另行获取（README 引用 `../calculet-calvxus`） |

### P1：影响交付质量

| # | 问题 | 证据 | 影响 |
| --- | --- | --- | --- |
| P1-1 | **文档口径冲突**：`-elf` 产物（README 说 CWX+ELF，STATUS 说 CalVX 后 ELF 全出不来，SYNTAX 说 -elf 也只出 CalVX）；`channel = "calculet"` vs README 说 `dopamine`；tiny 产物名 | §4.3、§3.2 | 上机首日必须实测确认，否则会误判“编译失败” |
| P1-2 | **CV 算子没有图/引擎接入路径**：calDNN `Op` 是封闭枚举、`Graph::op` 私有，vision 模板不 emit `desc!` | §5.6 | 赛题“路线 A”要求引擎注册；需跨仓库改 calDNN（1.5～2 天工作量） |
| P1-3 | `group_norm!`、`nchw_to_nhwc!`、`nhwc_to_nchw!`、`transpose2d!` **零实例、零验证** | §5.2 | SD UNet 若用 GroupNorm，等于从零验证 |
| P1-4 | **SD 端到端资料缺失**：仓库里没有 SD 模型例子（只有 minimax_h3 的 DiT/VAE）、没有 ComfyUI 桥接、没有 SD 权重；赛题说“待组委会发布提交与 FAQ” | `_recon/site.txt:134-135` | 若最终升级路线 B，需要组委会补料 |
| P1-5 | 真机资源有限、需要排队；板卡固件/驱动/健康状态流程只在 AGENTS.md 里 | §8.3、`_recon/site.txt:36` | 48h 计划必须给板子留 buffer 时间 |
| P1-6 | 备份站点 `https://docs.calculet.t12e.cc/` 从本机 TLS 连接被重置（HTTP 301 → HTTPS 失败），本机读不到官方文档 | §2.2 采集时的探测 | 只能以仓库内文档 + 赛题页为准 |

### P2：低优先级 / 长期

| # | 问题 |
| --- | --- |
| P2-1 | `Dograph.td`（厂商 30 算子定义）不在三仓库中，“30 个”只能靠 README 自述交叉核对，无法对照 TD |
| P2-2 | `npu/macros/src/lib.rs:15` 的文档仍写 `Ptr<Shared | Cache | Remote | Const, _>`（这些类型不存在，实际是 `Ddr`/`Ram`）；`target = "…"` 属性在宏里接受但 SYNTAX.md 未记录、全仓库无人使用 —— 写代码时以 `src/lib.rs` 和实际用法为准 |
| P2-3 | `calDNN/README.md:35` 说“没有发射过，也没有上板”；`calBLAS/STATUS.md` 与 README/AGENTS 的时间线互相矛盾（README 说六个例子未上板，但 10-30/11-02 有 yolo26/qwen36 上板记录）——引用结论时以带日期的记录为准 |
| P2-4 | 本报告的“建议 3 个缺失算子”中，PatchEmbed/im2col 与 GroupNorm 验证属于 🧠 推理优先级，最终以队伍/组委会目标模型为准 |

### 需要用户/组委会确认的问题（可直接转发）

1. 能否提供 **Linux x86_64 开发机**（或 WSL2 所需管理员权限）？本机是 Windows，官方工具链实测无 Windows 版本（附报错）。
2. 是否有 **HTTPS 版 install.sh / 官方 sha256 公示**？当前脚本走 HTTP（sha256 已记录：`3c17565…`）。
3. **docc / dort / calvxus** 的确切获取方式（工具链自带？还是另有下载）？
4. **真机如何申请/排队**？驱动（`calculet_pci.ko`）与 pydort 如何安装？是否有多芯粒板可用？
5. 赛题“路线 A”的引擎注册要求，是否接受“单算子整步程序 + 真机验证”作为阶段性交付？
6. SD/ComfyUI 的提交与评测材料何时发布（`_recon/site.txt:134-135` 写着“待组委会发布”）？

---

## 14. 附录

### A. 本机原始证据文件（全部在 `_recon/`）

| 文件 | 内容 |
| --- | --- |
| `_recon/phase1_evidence.txt` | 第一阶段的 4 条命令 + 附加探测的原始输出（含 OS/CPU/RAM） |
| `_recon/phase4_build_attempt.txt` | 第四阶段 `cargo build -p tiny …` 的真实输出（`cargo: command not found`、无 `target/`） |
| `_recon/install.sh` | 下载的安装脚本（875 B，sha256 3c17565…） |
| `_recon/install_run.log` | 受控执行 `sh install.sh` 的输出（`先安装 rustup`，exit 1） |
| `_recon/toolchain_install_probe.log` | 隔离 rustup 实测：`target 'x86_64-pc-windows-msvc' not found in channel` |
| `_recon/channel.toml` | 官方发行清单（仅 linux-gnu target） |
| `_recon/index.html` / `_recon/site.txt` | 赛题页原文/纯文本 |
| `_recon/quickstart.md` | 官方快速开始 |

### B. 三仓库版本（本报告全部源码引用的版本）

- calBLAS `32d2bb83d7160a593eaab67aa02e5dddbc8706cf`（2026-11-13 “黑客松下发版”）
- calDNN `1e06c41c557a469a9acc7566acb9f7e2c7f7ca62`（2026-11-13）
- npu `172aad2611c86ead4071356fbaca3cf6facd2917`（2026-11-13）

### C. 关键 file:line 索引（抽查用）

| 断言 | 证据 |
| --- | --- |
| Nearest Resize（×2）存在且限制 | `calBLAS/src/kernels/vision.rs:390-425`（doc :390，宏 :392，写视图 :407-410，分核 :412） |
| Bilinear/任意 resize 不存在 | 三仓库 case-insensitive grep 全 0 命中（§5.4/§5.5） |
| YOLO 用 upsample2 | `calBLAS/examples/yolo26/src/kernels.rs:43,46`；`examples/yolo26/deploy/sim.py:119` |
| 2D 卷积存在 + 约束 | `vision.rs:93-119`（k=1/3、s=1/2；CI/CO %16==0 于 :119） |
| GroupNorm 模板存在但没实例化 | `vision.rs:518-600`；“`calblas::group_norm!`” 除定义外 0 命中（本机 grep 验证） |
| CV 算子不进图 / calDNN Op 封闭 | `calBLAS/src/kernels.rs:21-24`（“不进图”）；`calDNN/src/graph.rs:405-451`（22 变体）；`calBLAS/src/engines.rs:45-131`（44 项，无 vision） |
| yolo26 不依赖 calDNN | `calBLAS/examples/yolo26/Cargo.toml:10-12` |
| YOLO26n 真机数值 | `examples/yolo26/deploy/DEPLOY.md:6-7` |
| tiny 的 10 个模板 | `examples/tiny/src/kernels.rs:9-18` |
| tiny 全程序 | `examples/tiny/src/step.rs:26-33,40-49,104-132` |
| SYNTAX kernel 约束 | `npu/SYNTAX.md:19-142`；宏实现 `npu/macros/src/lib.rs:37-56,73-84,265-294` |
| Engine 接口 | `calDNN/src/engine.rs:44-51`；`calBLAS/src/engines.rs:19-41` |
| docc/dort 命令 | `calBLAS/AGENTS.md:49-58,91-101,116-120`；`examples/yolo26/deploy/DEPLOY.md:25-46`；`examples/qwen3_8b/deploy/DEPLOY.md:41-56,75-80`；`calBLAS/README.md:97-100` |
| 工具链仅 Linux | `_recon/channel.toml` + `_recon/toolchain_install_probe.log` |
| 赛题环境要求 | `_recon/quickstart.md:5-8` |
| 赛题路线 A 要求引擎注册 | `_recon/site.txt:124-127` |

### D. 验证状态总结（本报告用到的每类信息）

| 信息类别 | 状态 |
| --- | --- |
| 本机环境（OS/shell/权限/WSL/网络/命令可用性） | ✅ 本机实测 |
| 工具链可安装性（发行清单 + rustup 实测报错） | ✅ 本机实测 |
| 源码事实（算子、模板、API、行号） | ✅ 源码阅读（Lead 亲自复核关键条目，3 名分析员交叉核对） |
| 编译结果 / 真机结果 | ❌ 本机无法验证；仓库文档记录标注为 ⚠️ |
| 48h 计划与选题建议 | 🧠 基于上述证据的推理，非既成事实 |

---

**报告结束**。下一步最小动作：**先拿到 Linux x86_64 环境（或管理员权限装 WSL2），按 §2 装工具链，按 §4 跑 tiny，把真实编译日志贴回来**；在环境恢复前，本报告 §5–§12 的代码结论可以直接用于写 kernel 草稿与准备验证脚本。

---

## 15. 环境修复进展：WSL2 安装（2026-11-13 18:26 更新，✅ 本机实测）

> 本节是对 §13 P0-1/P0-2 的进展更新。用户已授权以管理员权限安装 WSL2。

### 15.1 已完成

| 项 | 状态 | 证据 |
| --- | --- | --- |
| UAC 提权 | ✅ 成功（脚本以管理员身份运行） | `_recon/wsl_install_log.txt`：`IsAdmin: True` |
| `Microsoft-Windows-Subsystem-Linux` | ✅ **Enabled**（DISM exit 3010 = 成功、待重启生效） | 同上：`FeatureNow ... Enabled` |
| `VirtualMachinePlatform` | ✅ **Enabled**（同上） | 同上 |
| `wsl --install --no-distribution` | ❌ 失败：`REGDB_E_CLASSNOTREG`（WSL Store 包未注册；不阻塞“手动导入 rootfs”路线） | 同上 |
| WSL2 内核 MSI | ✅ 已下载 17,104,896 B，sha256 `4d09c776c8d45f70a202281d18e19be1118f53159b0c217a5274a31ce18525fe` | `_recon/wsl/wsl_update_x64.msi` |
| Ubuntu 24.04.3 rootfs | ✅ 已下载 29,983,772 B，**官方 sha256 校验通过**（`6bc2cd…`） | `_recon/wsl/ubuntu-base-24.04.3-base-amd64.tar.gz`、`SHA256SUMS` |
| Linux rustup-init | ✅ 已下载（ELF x86_64，21,347,088 B） | `_recon/wsl/rustup-init-linux-x86_64` |
| 一键脚本 | ✅ 已写好 | `_recon/wsl/env_setup.sh`（WSL 内装工具链+编译 tiny）、`_recon/wsl/post_reboot_setup_elevated.ps1`（提权装内核+导入 Ubuntu）、`_recon/wsl/trigger_post_reboot.ps1` |
| 续传说明 | ✅ | `_recon/wsl/WSL_RESUME_PLAN.md` |

### 15.2 当前唯一待办：重启 Windows

- WSL 的两个可选功能已启用，但**重启前服务不会激活**（`LxssManager`/`WslService`/`vmcompute` 均不存在；`wsl` 命令无输出）。
- 重启后按 `WSL_RESUME_PLAN.md` 的流程：提权脚本装内核 + `wsl --import` 导入 Ubuntu → `env_setup.sh` 装 rustup + calculet 工具链 + 编译 tiny → 真实日志 → 开始 Bilinear Resize 算子。
- WSL2 起不来时自动降级 WSL1（脚本内置），仍可编译 Linux ELF 的 rustc/docc。

### 15.3 状态变化

- §13 P0-1（无 Linux）：🚫 → **✅ 已解决（WSL2 3.0.5 + Ubuntu 24.04.3，§16.1）**。
- §13 P0-2（工具链装不上）：🚫 → **✅ 已解决（calculet 工具链 + tiny 编译成功，§16.2）**。
- §13 P0-3（无真机访问）、P0-4（docc/dort/calvxus）：**仍然阻塞**，需组委会提供板卡/驱动/pydort。

---

## 16. 最终结果：环境修复 + tiny 编译 + 新算子交付（2026-11-13 19:15，✅ 本机实测）

### 16.1 环境（全部实测）

| 项 | 结果 |
| --- | --- |
| WSL | `wsl --version` → **WSL 3.0.5.0**、内核 **6.18.58.1-1**、WSLg 1.0.79、Windows 10.0.26200.9974；`WslService: Running` |
| 发行版 | **Ubuntu 24.04.3 LTS**（官方 ubuntu-base rootfs，SHA256 校验后 `wsl --import` 导入） |
| rustup | 1.29.3 |
| calculet 工具链 | **rustc 1.98.1 (48a229cea 2026-09-01)**、cargo 1.98.1、**docc 0.4.1**、**dort 0.4.1** |
| 仓库 | WSL 内 `/root/{calBLAS,calDNN,npu}`，commit 与 §3.2 一致 |
| 工具链位置 | `/root/.rustup/toolchains/nightly-2026-11-13-x86_64-unknown-linux-gnu/bin/`；`docc`/`dort` 链到 `/root/.cargo/bin` |
| 发行版磁盘位置 | `C:\Users\10463\AppData\Local\WSL\Ubuntu`（`ext4.vhdx`，WSL2 导入式发行版）；2026-11-14 已从工作区 `_recon/wsl/UbuntuRoot` 迁出 |

修复路径（可复现，脚本与日志在 `_recon/wsl/`）：启用 WSL 功能（DISM 3010）→ 重启 → `winget` 安装 **`Microsoft.WSL 3.0.5`**（Store 的 2.7.10 只是启动器壳，MSI 才是完整 WSL）→ 导入官方 Ubuntu rootfs → `env_setup.sh` 装 rustup/工具链/仓库并编译。

迁移记录（2026-11-14，发行版实例不再放在工作区）：`wsl --shutdown` → `wsl --export Ubuntu <备份.tar>` → `wsl --unregister Ubuntu` → `wsl --import Ubuntu C:\Users\10463\AppData\Local\WSL\Ubuntu <备份.tar> --version 2` → `wsl --set-default Ubuntu`。迁移后实测：`rustc +calculet 1.98.1` / `docc 0.4.1` / `dort 0.4.1` 正常，`/root/{calBLAS,calDNN,npu}` 与已编译产物完整；`tiny`（elf + cvx）、`resize_probe`、`conv2d_s2_probe` 重新编译全部 exit 0，产物尺寸与迁移前一致（`resize_probe.cvx 35,160 B`、`conv2d_s2_probe.cvx 14,496 B`）。工作区 `_recon/wsl/` 仅保留安装脚本、日志与 rootfs 等证据资产（约 120 MB），不再有 2 GB+ 的 `ext4.vhdx`。

### 16.2 tiny 编译（第四阶段最终答案）

```bash
$ cargo build -p tiny --lib --target dopamine-unknown-none-elf
warning: npu：ELF 要 docc 收 CalVX 1.0（还没接上），这次只出了 CalVX（…/tiny.cvxo）
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.56s
BUILD_TINY_ELF_EXIT=0

$ ls -la target/dopamine-unknown-none-elf/npu/
tiny.cvx   89382
tiny.cvxo  14792
```

- ✅ **生成 `tiny.cvxo` + `tiny.cvx`**，与赛题页/`quickstart.md:31` 的期望一致。
- `-elf` 目标确实只出 CalVX（警告原文），印证 §4.3：README 的「CWX + 两核 ELF」是旧口径，SYNTAX/STATUS 的 CalVX 口径是对的。
- 整步程序也编译成功：`cargo build -p tiny --lib --features step --target dopamine-unknown-none-cvx` ⇒ `tiny.cvx 111863`、`tiny.cvxo 18491`。
- 日志：`_recon/wsl/build_logs/`（`env_setup.log`、`build_tiny_elf.log`、`build_tiny_cvx.log`、`env_summary.txt`）。

### 16.3 新算子 `resize_bilinear2`（交付）

**Bilinear Resize ×2（NHWC bf16，align_corners=False）** 已从零实现并完成平台侧验证：

| 验证 | 结果 |
| --- | --- |
| 设备遍编译 | ✅ `resize_probe.cvx` 35,160 B / `.cvxo` 6,655 B |
| `docc calbin from-calvx` | ✅ exit 0，产出 CCU ELF（ccu0 19,560 B / ccu1 6,872 B） |
| `docc calbin doctor` | ✅ **拒收 0 条**（2 条 ping/pong 警告） |
| 构建确定性 | ✅ 两次 calbin 逐字节相同（`docc calbin diff`：一致） |
| CCU 反汇编 | ✅ 2849 条指令全部可识别、0 unknown、每字节覆盖；`load_size 14608` 与节大小之和吻合 |
| CPU 参考 | ✅ kernel 次序 f32 vs 独立 float64：65,536 个输出 **bf16 舍入 100% 一致**（≤0.996 ULP） |
| 真机数值 | ❌ **未验证**（本机无板卡/无 `cvx` 模拟器） |

产物与说明：`RESIZE_OPERATOR.md`；代码已同步到工作区仓库 `calculet-repos/calBLAS`（`src/kernels/vision.rs` + `examples/resize_probe/`），补丁 `_recon/work/resize/artifacts/resize_bilinear2.patch`。

### 16.4 仍未解决

- **P0-3 真机访问**：无 Calculet 板卡/驱动/pydort/排队方式；`dort overview` 报「未发现 Calculet 设备」。
- **无 `cvx` 模拟器**：calvxus（`cvx run`）不在三个下发仓库，工具链也不带 ⇒ 本机只能做平台侧静态验证，不能跑数值。
- **P1-2 CV 算子进图**：`calDNN::Op` 是封闭枚举、`Graph::op` 私有、vision 模板不 emit `desc!`；要满足路线 A 需跨仓库改 calDNN。
- **docc 限制**（新算子开发中实测到）：① `calvx.for` 体内带结果的 `if` 不支持（已用无分支写法绕开）；② `docc elf kabi` 对该 kernel 报 `calt-bind` 错误（docc 自诊断「多半是 docc 自己的缺陷」）；③ `double_buffer` 的 ping/pong 地址警告。
- 其余同 §13 P1/P2。
