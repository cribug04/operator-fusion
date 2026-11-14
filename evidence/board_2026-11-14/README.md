# 真机原始件归档 · 2026-11-14（calculet-MS-A2）

本目录存放**板上跑出来的原始文件**，与两份报告配对：

| 报告（`evidence/` 根） | 原始件（本目录） |
|---|---|
| `vae_stage_report.md` | `vae/timing.txt`、`vae/compare.json`、`vae/image.png` |
| `resize_dualcore_board_report.md` | `resize_dualcore/timing.tsv`、`resize_dualcore/logs/` |
| `INDEX.md`（Resize 交付证据索引） | 见上两行 |

## 环境（两次运行同一台机器）

```text
板机   calculet-MS-A2, x86_64, /dev/calculet0  PCI 0000:07:00.0  ID 20f5:c001
驱动   Driver 0.9.3
固件   DRT commit 74832e51 · 0x03010101（rt 3.1 / ABI 1.1，线协议 rev 2）
链路   PCIe 16.0 GT/s x4     芯粒  Dev: OS up  RT up  chips: 2
工具链 rustc 1.98.1-calculet · docc 0.4.1 · dort 0.4.1
```

⚠ 本机缺 `/opt/data/locks`：① dort 板锁用 `--board-lock external` 绕过；② CalRT band 锁亦缺 ⇒ 日志固定出现
`启动流带：关`（CSR 写不合并，性能降级）。两者都不影响数值。

## VAE（5 段全 NPU，SD1.5 真实权重）

`vae/timing.txt`（段级端到端墙钟，含部署与裸字节 IO）：

```text
pkg_a 0.9s · pkg_b1 1.0s · pkg_b2a 3.9s · pkg_b2b 0.3s · pkg_c 28.4s   total 34.6s
```

`vae/compare.json`（板上 `image.npy` vs torch fp32 参考，同一 `latent.bin`）：

```json
{ psnr: 52.5801,  max_abs: 0.0535825,  mean_abs: 0.00347418,  rmse: 0.00469921 }
```

判据（文档）：`PSNR ≥ 40 dB` 且 `max|Δ| ≤ 0.08` ⇒ **通过**。参照：黄金仿真器同口径 52.83 dB / 0.0473。
`vae/image.png` 为板上解出的 512×512 图（参考图见 `sd_vae/evidence/reference_image.png`）。

## Resize 双核 vs 单核（`resize_bilinear2_tiled2`）

`resize_dualcore/timing.tsv`：21 行（表头 + 4 例 × 5 次），列 `name / run / seconds / sha256(前16)`。

| 用例 | 核数 | 中位数 | 输出 sha256（前 16） | 判据 |
|---|---|---|---|---|
| `resize_big` | 单核 | 0.006 s | `6fb683176318acd3` | EXACT 1 048 576/1 048 576 |
| `resize_big2` | 双核 | 0.006 s | `6fb683176318acd3` | EXACT（与单核同哈希） |
| `resize_huge` | 单核 | 0.052 s | `5c1d4c42450b96ce` | EXACT 16 777 216/16 777 216 |
| `resize_huge2` | 双核 | 0.047 s | `5c1d4c42450b96ce` | EXACT（与单核同哈希） |

- 中位数比：`big` 1.04（噪声级）、`huge` **1.10**；双核输出与单核**逐字节相同**。
- `logs/cmp-*.txt`：四份对比器输出（`criterion: exact`、`VERDICT: EXACT`）。
- `logs/run-<name>-<r>.log`：20 份逐次运行日志（含部署与 `启动流带：关`）。
- `logs/device.txt`：健康检查快照（驱动/固件/链路/芯粒/进程表）。

**计时口径（必须与数字同报）**：`dort calbin run` 整条命令墙钟，含进程启动、模型部署、输入输出搬运
（`huge` 输入 8 MiB + 输出 32 MiB），**不是 kernel 净时间**，只能作同形状单核/双核比值。
按 `T = 固定开销 + kernel` 两点粗估：`huge` 段 kernel ≈ 10 ms → 5 ms（约 2 倍），固定开销 ≈ 42 ms（**待 stamp 插桩确认**）。

## 复跑

```bash
# VAE
tar -xzf sd_vae_onboard.tar.gz -C ~ && cd ~/sdvae_pkg
cp /tmp/reference_image.npy ref_image.npy && bash run_on_board.sh . 0
python3 host.py compare image.npy ref_image.npy compare.json

# Resize 双核 vs 单核
tar -xzf resize_dualcore_onboard.tar.gz -C ~ && cd ~/rz2_pkg
for m in resize_big resize_big2 resize_huge resize_huge2; do : > $m/param_blk0.bin; done
CAL_ROOT=$PWD N=5 bash run_resize_perf.sh
```

（`CAL_ROOT=$PWD` 必须带：脚本默认找 `$HERE/onboard_cal`；`param_blk0.bin` 必须存在，无权重程序也要 0 字节文件。）
