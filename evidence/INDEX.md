# Resize 算子 · 证据索引

目标算子：`resize_bilinear2`（2× 双线性上采样，NHWC bf16，`align_corners=False`）。
本文件只收 resize 的证据；其它线（conv 原型 / VAE / FFN 融合）不在本交付范围内。

---

## 0. 环境（复跑前对齐）

| 项 | 值 |
|---|---|
| 板机 | **calculet-MS-A2**，x86_64，`/dev/calculet0`（PCI 0000:07:00.0，ID 20f5:c001） |
| 驱动 / 固件 | Driver **0.9.3**；DRT commit **74832e51**（0x03010101，rt 3.1 / ABI 1.1，线协议 rev 2） |
| 链路 / 芯粒 | PCIe **16.0 GT/s x4**；`Dev: OS up  RT up  chips: 2` |
| 工具链 | rustc **1.98.1-calculet** · cargo 1.98.1 · **docc 0.4.1** · **dort 0.4.1** |
| 板锁 | 本机 **无** `/opt/data/locks` ⇒ 所有 `dort calbin run` 必须带 `--board-lock external` |

---

## 1. 三向一致性（核心证据）

| 方向 | 元素数 | 结果 |
|---|---|---|
| **独立参考**（`_bundle/resize_ref.py`，`--order=f32`）→ 仓库 `expected.bin` | 65 536 / 1 048 576 | **100% 逐字节相同**（两个尺寸都是） |
| 仓库 `expected.bin` → **板子输出** | 65 536 / 1 048 576 | **EXACT**，max\|diff\| = 0，cosine = 1.000000000 |
| 纯数学 f64 参考 → 仓库 `expected.bin`（交叉口径） | 同上 | 96.90% 命中，其余 ≤1 ULP 级（= 只差舍入次序） |

**结论**：算子与"按 kernel 运算次序的 f32 参考"逐位相同；与纯数学参考的差异全部落在 bf16 1 ULP 内。

### 1.b 跨分支复验（2026-11-14，第二次上板）

| 项 | 结果 |
|---|---|
| calbin 来源 | `jax/validation-fixes` 分支（含 Resize 图注册补丁）重新编译打包；其 `.cvx` 与 master 版 **逐字节相同**（`536b406a6ec2…7118f`） |
| `resize` 输出 | 三次 `725b91e13d946710c48eb194a11f7a0e25fcccde5df2d563cb2a585b90ba2cff`（与第一次上板**完全同一哈希**） |
| `resize_big` 输出 | 三次 `6fb683176318acd3b0736307bec0490dc210e5a8696062cb974c22852f9e40ff`（同上） |
| 附带校验 | 输出字节数（131 072 / 2 097 152）由脚本强制校验；三次 `cmp` 必须相同才继续；比对用官方 `compare_bf16.py`（numpy） |
| 机器 | `Dev: OS up  RT up  chips: 2` · DRT `74832e51` (rt 3.1 / ABI 1.1) · `Link 16.0 GT/s x4` · dort 0.4.1 |

⇒ 构建侧（换分支、换补丁重编）与运行侧（三次重跑）都得到**同一哈希**，可复现性成立。

## 2. 复跑命令

```bash
# ① 参考/用例（开发机或板机，纯 python3，不需要 numpy）
python3 resize_ref.py gen 32 32 16  /tmp/case --order=f32      # 造 input.bin + expected.bin
python3 resize_ref.py check out.bin /tmp/case/expected.bin     # 只报 EXACT/DIFF
#    复现仓库自带期望（证明参考实现与作者那份等价）：
python3 resize_ref.py gen 32 32 256 /tmp/case_big --order=f32   # 与 cases/resize_big/expected.bin 逐字节比

# ② 板上跑
dort device -i 0 | grep "Dev  :"                                # 要 OS up  RT up
dort calbin run -i 0 --board-lock external -f ~/of_cal/resize -m resize_pkg \
  --raw-in operator-fusion/onboard/cases/resize_small/input.bin \
  --raw-out out_resize.bin --in-tensor in_x --out-tensor out_y
python3 compare_bf16_raw.py out_resize.bin \
  operator-fusion/onboard/cases/resize_small/expected.bin --label resize
```

## 3. 哈希与产物

| 项 | 值 |
|---|---|
| `resize_probe.cvx`（现场重编） | `536b406a6ec2512b17e936e6847a138354be8466ee52fb3751b4c49b0947118f` |
| `evidence/resize/resize_probe.cvx`（仓库） | **同上**（逐字节相同 ⇒ 构建可复现） |
| 板子 `resize` 输出（3 次） | `725b91e13d946710c48eb194a11f7a0e25fcccde5df2d563cb2a585b90ba2cff` ×3 |
| 板子 `resize_big` 输出（3 次） | `6fb683176318acd3b0736307bec0490dc210e5a8696062cb974c22852f9e40ff` ×3 |
| 用例数据 | `resize_small`: input 32 768 B / expected 131 072 B；`resize_big`: input 524 288 B / expected 2 097 152 B；**无权重**（`param_blk0.bin` = 0 B） |

## 4. 缺口清单（按 ROI）

| # | 缺口 | 现状 | 下一步 |
|---|---|---|---|
| 1 | ~~参考生成器未交付~~ | **已补**：`_bundle/resize_ref.py`，复现仓库 expected 100% | — |
| 2 | **形状覆盖不足**：只有 32×32×16 / 32×32×256 | 缺 `C=1`、`C=128`（上限）、`C=129`（单块→分块切换点）、`H=W=3`（最小边界，压第 0 行/最后一行展开分支）、奇数尺寸 `17×33×8` | 每形状要一条编译期特化 + 一个 calbin；用 `resize_ref.py gen` 造用例 |
| 3 | **第三方参考**：目前只有"自家 numpy 参考" | 缺 **PyTorch** `F.interpolate(...)` 对拍 | 开发机有 torch 时生成一份 expected 比一次 |
| 4 | **板上重复性**：只有"同进程连跑 3 次同哈希" | 缺跨进程/重启、换设备号、换另一块板（不同固件） | 3 条命令即可 |
| 5 | **边界数值输入**：全零、±bf16 max、`-0`、denormal | 未测 clamp 与 RNE 边界 | `resize_ref.py gen` 加 `--mode extreme`（待加） |
| 6 | **性能**：单 kernel 周期戳 / 等效带宽 | 未测 | `tools/bench` 的 `debug.stamp` |
| 7 | **集成验证**：串进一段真实前后处理整段跑 | 未测 | 用检测模型的 upsample 段 |
| 8 | **错误路径**：尺寸与特化不符时的行为 | 未测 | 造一个不匹配输入看是否明确报错 |

## 5. 已知环境坑（会挡住复跑）

| 现象 | 绕过 |
|---|---|
| `error: 打不开板锁 /opt/data/locks/board.lock` | 加 `--board-lock external`；正解 `sudo install -d -o <user> -m 0775 /opt/data/locks` |
| `docc 0.4.1` 无 `calbin doctor` | `package_all.sh` 的 doctor 两步要注释掉 |
| `--dump-out` 不自动建目录 | 先 `mkdir -p` |
| 补丁只在 LF 检出上能打 | Windows CRLF 检出会报 `different line endings`（假失败） |
