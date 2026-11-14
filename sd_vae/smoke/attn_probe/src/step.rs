//! 整个程序：q / k / v / o 紧排 `[T, D]` bf16，mask = keys（全 0）。
use crate::kernels::*;

#[cfg(not(feature = "d512"))]
npu::decls! {
    in_q: input, bytes = 1048576;    // [4096, 128] bf16
    in_k: input, bytes = 1048576;
    in_v: input, bytes = 1048576;
    in_mask: input, bytes = 8192;    // [32][128] bf16 = T 个 key 的掩码
    out_o: output, bytes = 1048576;
}

#[cfg(feature = "d512")]
npu::decls! {
    in_q: input, bytes = 4194304;    // [4096, 512] bf16
    in_k: input, bytes = 4194304;
    in_v: input, bytes = 4194304;
    in_mask: input, bytes = 8192;
    out_o: output, bytes = 4194304;
}

#[cfg(not(any(feature = "d512", feature = "d256")))]
#[npu::program(chiplets = 1, cores = 2)]
pub fn step() {
    probe_attn_d128(in_q().cast(), in_k().cast(), in_v().cast(), in_mask().cast(), out_o().cast());
}

#[cfg(all(feature = "d512", not(feature = "d256")))]
#[npu::program(chiplets = 1, cores = 2)]
pub fn step() {
    probe_attn_d512(in_q().cast(), in_k().cast(), in_v().cast(), in_mask().cast(), out_o().cast());
}

#[cfg(feature = "d256")]
npu::decls! {
    in_q256: input, bytes = 2097152;    // [4096, 256] bf16
    in_k256: input, bytes = 2097152;
    in_v256: input, bytes = 2097152;
    in_mask256: input, bytes = 8192;
    out_o256: output, bytes = 2097152;
}

#[cfg(feature = "d256")]
#[npu::program(chiplets = 1, cores = 2)]
pub fn step256() {
    probe_attn_d256(in_q256().cast(), in_k256().cast(), in_v256().cast(), in_mask256().cast(), out_o256().cast());
}
