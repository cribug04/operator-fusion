# 题五 · 用 NPU 找出振动过强的时间段

合成信号切成 16 段，每段 64 个 FP32 采样点。NPU 在每段内将数值平方并求平均。
CPU 标记均方值超过 0.5 的段。预期第 3、10、15 段被标记，从 0 开始编号。
采样数据和阈值用于演示。

在本目录运行 `python3 detect.py` 可执行 CPU 参考。运行 `python3 detect.py --device 0`
可在比赛设备上执行，0 换成设备编号。设备模式逐项检查 16 个均方值，并验证异常段号。
设备模式耗时包含工具启动、加载和输入输出。当前已完成本地 CPU 样例和 CalBin 静态解析，
NPU 的数值和用时待上板验证。

源码在压缩包中，解压到 calBLAS 的 examples/signal_energy 目录后，在该目录执行
`cargo build --locked --lib --target dopamine-unknown-none-cvx` 可重新生成 CalVX。
