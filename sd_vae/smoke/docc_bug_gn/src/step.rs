//! 最小复现：单个 128² group_norm 调用（x / y 各 512·16384 bf16 = 16 MiB；w/b 各 524,288 B）。
use crate::kernels::*;

#[cfg(not(feature = "sram"))]
npu::decls! {
    x: input, bytes = 16777216;
    y: output, bytes = 16777216;
    w: weights, bytes = 1048576;   // w + b 各 524,288 B
}

#[cfg(feature = "sram")]
npu::decls! {
    x: input, bytes = 16777216;
    y: output, bytes = 16777216;
    w: weights, bytes = 1048576;
    l2: ram, bytes = 64;           // 主机把 NH（本形状 = 16）写在这里
}

#[cfg(not(feature = "sram"))]
#[npu::program(chiplets = 1, cores = 2)]
pub fn step() {
    k_gn_16384_512_32_silu(x().cast(), w().at(0).cast(), w().at(524288).cast(), y().cast());
}

#[cfg(feature = "sram")]
#[npu::program(chiplets = 1, cores = 2)]
pub fn step() {
    k_gn_sram_16384_512_32_silu(x().cast(), w().at(0).cast(), w().at(524288).cast(), y().cast(), l2().at(0).cast());
}
