//! 慢路对照：calBLAS 仓库现成的 `conv2d!`（S=2 时走慢路），其它参数与 conv2d_s2_probe 完全一致。
calblas::conv2d!(
    pub conv_s2_ref: h = 80, w = 80, cin = 64, cout = 64, k = 3, s = 2,
    pin = (64, 0), pout = (64, 0), act = silu
);
