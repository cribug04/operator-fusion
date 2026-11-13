//! stride-2 FAST 卷积探针：h=80, w=80, cin=64, cout=64, k=3, s=2, act=silu。
calblas::conv2d_s2_fast!(pub conv_s2_probe: h = 80, w = 80, cin = 64, cout = 64, act = silu);
