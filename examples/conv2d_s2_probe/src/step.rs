//! 整个程序：输入 [164][164][64] bf16（padded NHWC 紧排）、权重 [9][64][64] bf16、bias f32 [64]、输出 [84][84][64] bf16。
use crate::kernels::*;

npu::decls! {
    in_x: input, bytes = 3442688, double_buffer;    // 164*164*64*2 = 3,442,688 B
    out_y: output, bytes = 903168, double_buffer;   // 84*84*64*2
    w: weights, bytes = 73984;                      // 9*64*64*2 + 64*4
}

#[npu::program(chiplets = 1, cores = 2)]
pub fn step() {
    conv_s2_probe(
        in_x().cast(),
        w().cast(),
        w().at(73728).cast(),
        out_y().cast(),
    );
}
