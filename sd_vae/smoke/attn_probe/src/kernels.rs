//! 同一个 kernel 模板，两个头维：缺省 D = 128（预期可用），`d512` feature ⇒ D = 512（预期被 docc 拒）。
//! 形状：单头、T = 4096 行 q、无真掩码（`mask = keys` + 全 0 的 mask 缓冲 = 恒等）。
#[cfg(not(feature = "d512"))]
calblas::attention!(pub probe_attn_d128: rows = 4096, heads = 1, head = 128, mask = keys);

#[cfg(feature = "d512")]
calblas::attention!(pub probe_attn_d512: rows = 4096, heads = 1, head = 512, mask = keys);

#[cfg(feature = "d256")]
calblas::attention!(pub probe_attn_d256: rows = 4096, heads = 1, head = 256, mask = keys);
