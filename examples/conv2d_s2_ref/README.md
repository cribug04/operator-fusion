# conv2d_s2_ref：stride-2 慢路对照（真机验证用）

与 `examples/conv2d_s2_probe`（`conv2d_s2_fast!` 原型）**同形状、同 decl 布局**的仓库现成慢路实现：

- `calblas::conv2d!(h=80, w=80, cin=64, cout=64, k=3, s=2, pin=(64,0), pout=(64,0), act=silu)`
- `S=2` 时 `conv2d!` 走慢路（`vision.rs:122` 的 FAST 只覆盖 `S==1`）

**用途**：在真机上用**同一份 `input.bin` + 同一份 `param_blk0.bin`** 分别跑 fast 与 ref 两个 calbin，比较 `@out_y`：

- 两者应当**逐位相同**（同样的 9 个 tap、同样的 mma 累加次序、同样的 bias/SiLU 宏；唯一区别是 lhs 来自 VB 还是 DDR）；
- 若不同：说明 fast 原型的索引/奇偶映射有 bug，或 docc 对两条路径的数值合同不同——按 `REAL_DEVICE_VERIFICATION.md` 的排错表处理。

## 构建/打包（WSL/Linux，同 probe）

```bash
cd ~/calBLAS
cargo build --manifest-path examples/conv2d_s2_ref/Cargo.toml --lib --features step \
  --target dopamine-unknown-none-cvx
# 打包时 param_blk0.bin 用开发机 gen_conv_s2_case.py 生成的真实参数块（与 probe 完全相同）
```

本机平台侧验证：`.cvx` 15,241 B，docc calbin + doctor 通过（ccu0/ccu1 各 16,152 B；权重块 73,984 B）。
