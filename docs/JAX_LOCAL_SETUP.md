# Jax 的本地部署进度

## 历史部署记录（首轮四探针）

- Gitee 仓库已克隆到 Mac：`<本仓库绝对路径>`。
- 本次读取的提交为 `35d1f4b`。
- 根据此前终端截图，OrbStack `calc` 已安装 Linux x86_64 calculet 工具链；三个官方仓库在 Linux 的 `~/bohack` 下。
- 根据此前截图，官方 tiny 已编译、CPU 参考已运行、docc 已打包。这些结果不代表新算子已在本机或真机验证。
- 本仓库包含 Resize、通道分块 Resize、stride-2 卷积原型和慢路对照，以及已有验证记录。

## 本地构建（已完成，可按需重跑）

在 Mac 终端进入本仓库根目录后执行：

```bash
/Applications/OrbStack.app/Contents/MacOS/bin/orb -m calc bash "$PWD/onboard/setup_local.sh"
```

脚本会检查环境、检查并应用 `onboard_kit.patch` 到 Linux `~/bohack/calBLAS`，然后编译四个探针、docc 打包、doctor 检查。发生冲突时停止；不会重置已有源码。输出写入新的 `~/bohack/operator-fusion-builds/run.*` 目录。成功标志是最后出现“本地编译、打包及 doctor 检查通过”。

Jax 随后在 Mac 终端运行脚本，提供的截图显示四个探针的本地编译、打包及 doctor 检查通过，输出位于 Linux `~/bohack/operator-fusion-builds/run.*`。这项确认依据用户终端截图；尚无 NPU 运行证据。

## 后续顺序

1. 已完成本地四个探针的编译、打包检查；CPU 数据复核与验收脚本修复见 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)。
2. 获取主办方提供的 NPU 真机访问方式与测试安排。
3. 在 NPU 主机运行 Resize，取回输出并与 CPU 参考比较。
4. 卷积 fast/ref 在同一设备上比较输出、测量耗时；正确性通过后再讨论加速收益。
5. 补齐 calDNN 图与引擎接入、整理测试证据和展示。编译或 doctor 通过均不能替代真机正确性与性能验证。

当前对比脚本已按 exact / resize / conv-cpu 分离判据，失败返回非零退出码，且已通过本地回归测试。容差来自仓库原有操作手册，仍需实际硬件结果与赛事要求确认。

## 当前状态入口

上文保留首轮部署记录，不作为当前待办。最新七探针及双核状态见 [RESIZE_DUALCORE.md](RESIZE_DUALCORE.md)，图注册已完成见 [CALDNN_INTEGRATION.md](CALDNN_INTEGRATION.md)，真机转述及证据状态见 [README](../README.md)。
