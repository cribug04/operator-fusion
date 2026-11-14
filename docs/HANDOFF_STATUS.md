# 接续验收记录（2026-11-14）

基线：`ef2c845`，分支 `jax/resize-dualcore`。Fable 已完成 README、评委指南与讲稿初稿；本轮补齐复现路径、证据边界和路径清理。

## 本轮实测

- macOS arm64，Python 3.14.7 + numpy 2.5.4，临时虚拟环境。
- `python3 -m unittest discover -s tests`：43 项通过，8.779 秒。
- `python3 onboard/check_resize_cpu.py`：65,536 与 1,048,576 个元素，两种复核实现均 0 差异；未执行 NPU。
- 修改文档的本地 Markdown 链接检查及 `git diff --check` 通过。
- 本轮未重新运行 Linux Rust / calculet 编译；指南 B/C 的预期输出来自既有记录，不能算本轮实测。

## 已修正

- 评委指南固定分支和上游提交，使用绝对补丁路径，补 numpy 环境步骤，修复 A→B→C 连续执行的目录问题。
- 讲稿纠正 VAE 最近邻上采样的表述、单函数分支范围限制，正文时间改为 3:50，加 10 秒换页余量。
- 真机数字保留为队友转述；仿真 `sd_vae/evidence/compare.json` 不能替代真机报告。
- 清理文档个人 Windows / Mac 路径及打包校验清单绝对路径；历史 Git 提交未重写。
- 更新过时的“等待板卡登录”表述，标明历史部署记录；删除仅凭提交时间无法证明的赛前代码断言。

## 尚未完成的外部核验

Gitee HTTPS 的 git / curl 访问失败，网页抓取不可用，浏览器连接不可用。因此**公开状态、远端 master 最新提交、队友报告是否入库均未确认**。本地缓存的远端分支信息不能替代实时核验。

网络恢复后，在仓库根目录执行：

```bash
git fetch origin
git log --oneline HEAD..origin/master
git ls-tree -r --name-only origin/master | rg 'board|timing|compare|report'
git push -u origin jax/resize-dualcore
```

推送到该分支不会自动更新 master。向 master 合并前应检查上述新增内容及冲突；不要强推。公开可见性要用未登录窗口打开仓库确认，登录后能看到仓库不等于公开。

真机报告入库后，需核对运行提交号、设备版本、输入/输出哈希、对比判据和原始计时口径，再更新 README 与讲稿。双核 Resize 和 stride-2 卷积的上板验证仍未完成。
