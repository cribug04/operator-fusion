# 题四 · 合成资料本地检索与 NPU 重排

样例包含 8 条合成英文资料和 10 条标注查询。CPU 将英文问题拆成词，
按词重合数给每条资料打分。NPU 复用题二的 FP32 Top-K CalBin 对分数排序，
返回前 16 项和原始编号。CPU 过滤补位和零分结果，然后根据编号取回原文。
词匹配覆盖英文单词。语义相似、中文分词和外部资料不在本版支持范围。

```bash
python3 retrieval.py --query "fusion residual ddr"
python3 retrieval.py --evaluate
python3 retrieval.py --device 0 --evaluate
```

前两条命令使用 CPU 参考。第三条使用 NPU 排序，0 换成分配到的设备号。
每个查询都把 NPU 的结果与 CPU 参考逐项对比，并检查第一条结果是否命中标注，
或无相关资料时是否返回空。输出中的端到端时间从查询拆词到返回结果，NPU 模式
包含启动 dort、加载 CalBin 和输入输出，计时也包含 CPU 参考校验。
这些数据可用来比较当前调用流程。板端纯执行时间需从设备侧另行测量。

当前先用自建合成语料验证流程，正式验收还需核对主办方指定资料和核心计算要求。
