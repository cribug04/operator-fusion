//! 整个程序：静态存储 + 入口（`cargo build --features step --target dopamine-unknown-none-cvx` 出一份 `.cvx`）。
use crate::kernels::*;

npu::decls! {
    in_x: input, bytes = 32768, double_buffer;      // 32*32*16*2
    out_y: output, bytes = 131072, double_buffer;   // 64*64*16*2
}

#[npu::program(chiplets = 1, cores = 1)]
pub fn step() {
    resize_bilinear2(in_x().cast(), out_y().cast());
}
