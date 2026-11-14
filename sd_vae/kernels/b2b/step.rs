//! SD1.5 VAE decoder 整步程序（gen.py 生成，勿手改）。
use crate::kernels::*;
use npu::prelude::*;

npu::decls! {
    in_x: input, bytes = 17842176;
    out_y: output, bytes = 69222400;
    w: weights, bytes = 4722688;
    act: workspace, bytes = 69222400;
}

#[npu::program(chiplets = 1, cores = 2)]
pub fn step() {
    // 所有图像 slot 一次清零：边框之后不再被写
    if core_rank() == 0 {
        k_fill0(act().at(0).cast(), 4225);
        k_fill0(out_y().cast(), 4225);
    }
    sync_cores();
    if core_rank() == 0 { k_up_128_256_p512_512(in_x().add(0).cast(), act().at(0).add(0).cast()); }
    sync_cores();
    if core_rank() == 0 { k_up_128_256_p512_512(in_x().add(512).cast(), act().at(0).add(512).cast()); }
    sync_cores();
    k_conv_256_512to512_k3_p512_512(act().at(0).cast(), w().at(0).cast(), w().at(4718592).cast(), out_y().cast());
    sync_cores();
}
