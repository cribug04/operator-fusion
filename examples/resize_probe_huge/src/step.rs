//! 整个程序：静态存储 + 入口（`cargo build --features step --target dopamine-unknown-none-cvx` 出一份 `.cvx`）。
use crate::kernels::*;

npu::decls! {
    in_x: input, bytes = 8388608, double_buffer;      // 128*128*256*2
    out_y: output, bytes = 33554432, double_buffer;   // 256*256*256*2
}

#[npu::program(chiplets = 1, cores = 1)]
pub fn step() {
    resize_bilinear2_huge(in_x().cast(), out_y().cast());
}
