# 题三 · Tiny 小模型的 RMSNorm 适配与校验

固定 calBLAS 自带的 Tiny 两层模型，隐藏宽度 512，使用确定生成的合成权重。
三个仓库的上游版本在 upstream.txt。tiny-rms.patch 适用于其中的 calBLAS 版本。
复用现有 BF16 RMSNorm 实现，增加 rms-check 构建功能，在完整模型推理中
导出最后一层 RMSNorm 的输入和输出。两芯粒各自完成模型计算，汇合出 512 个 logits。

进入本目录，运行 `bash run_board.sh 0`。把 0 换成分配到的设备编号。
样例使用 token 7。脚本保存一轮推理的日志和输出，然后分别检查以下两项。

- 从设备输出缓冲区读取最终 RMSNorm 的输入和输出。CPU 读取同一份模型权重，
  对这一层独立求均方根并计算参考值，用于定位归一化本身的问题。
- 取输出缓冲区头 1024 字节中的 512 个 logits，与既有 tiny-ref 整模型 CPU 输出逐项比较。

校验阈值暂定为 RMSNorm 3.5% 相对误差加 0.002 绝对误差，整模型 5% 加 0.05，
待设备输出验证。这里用同一份合成权重和单个 token 验收固定样例。
本地已编译及打包，CPU 参考 logits 已生成。物理设备上的加载、推理和数值校验待执行。

应用补丁后，在 calBLAS 根目录可重新构建。输出路径需替换为已有的空目录。
```bash
cargo build --locked -p tiny --lib --features rms-check --target dopamine-unknown-none-cvx
cp target/dopamine-unknown-none-cvx/npu/tiny.cvxo "$output/step.cvxo"
cargo run --locked -q -p tiny-pack --bin tiny-pack -- "$output"
cargo run --locked -q -p tiny-pack --bin tiny-ref -- "$output" 7 "$output/reference_logits.bin"
docc calbin from-calvx "$output" --network tiny -o "$output/cal"
```
