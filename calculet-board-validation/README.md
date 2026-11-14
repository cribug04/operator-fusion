# Calculet NPU 五题上板验收

在已安装 Calculet 驱动、dort 和 Python 3 的 Linux 比赛机器上，执行 dort device 确认设备编号。下列命令从本目录运行，将 0 改为分配的编号。题三使用两芯粒，其余包使用单芯粒。

| 题 | 运行命令 | 校验内容 |
| --- | --- | --- |
| 一 · 残差加法与 RMSNorm 融合 | cd fused_norm_board && bash run_board.sh 0 | 比较融合、两步结果及 CPU 参考 |
| 二 · Top-K | cd topk_board && bash run_board.sh 0 | 256 选 16，对照 CPU 检查分数和编号 |
| 三 · Tiny 模型 RMSNorm | cd tiny_rms_board && bash run_board.sh 0 | 核对最终 RMSNorm 和整模型 logits |
| 四 · 本地检索 | cd retrieval_board && python3 retrieval.py --device 0 --evaluate | 核对排序、查询命中和端到端耗时 |
| 五 · 振动监测 | cd signal_energy_board && python3 detect.py --device 0 | 核对 16 段均方值和异常段号 |

各目录 README 说明输入输出、源码位置和支持范围。提交包含设备程序包、本地测试输入及参考数据。题三权重由脚本生成，为运行和 CPU 校验一并保留。

题一已通过本地主机测试，题二已通过编译及 CPU 校验器自测，题三已生成 CPU 参考，题四及题五已运行 CPU 样例。五题均待队友运行真机并保存日志。题一的搬运量是静态估算，设备数值和性能以队友记录为准。题一到题四的官方接口和验收数据仍需核对。
