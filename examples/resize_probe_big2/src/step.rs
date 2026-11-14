//! 整个程序：静态存储 + 入口（`cargo build --features step --target dopamine-unknown-none-cvx` 出一份 `.cvx`）。
use crate::kernels::*;

npu::decls! {
    in_x: input, bytes = 524288, double_buffer;      // 32*32*256*2
    out_y: output, bytes = 2097152, double_buffer;   // 64*64*256*2
}

#[npu::program(chiplets = 1, cores = 2)]
pub fn step() {
    resize_bilinear2_big2(in_x().cast(), out_y().cast());
}
