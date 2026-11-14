//! SD1.5 VAE decoder 整步程序（gen.py 生成，勿手改）。
use crate::kernels::*;
use npu::prelude::*;

npu::decls! {
    in_x: input, bytes = 163840;
    out_y: output, bytes = 4734976;
    w: weights, bytes = 21483520;
    act: workspace, bytes = 39436288;
}

#[npu::program(chiplets = 1, cores = 2)]
pub fn step() {
    // 所有图像 slot 一次清零：边框之后不再被写
    if core_rank() == 0 {
        k_fill0(act().at(65536).cast(), 289);
        k_fill0(act().at(4800512).cast(), 289);
        k_fill0(act().at(9535488).cast(), 289);
        k_fill0(out_y().cast(), 289);
    }
    sync_cores();
    k_conv_64_16to512_k3_p16_512(in_x().cast(), w().at(0).cast(), w().at(147456).cast(), act().at(65536).cast());
    sync_cores();
    k_nhwc2nchw_64_512_p512(act().at(65536).cast(), act().at(14270464).cast());
    sync_cores();
    k_gns_4096_512_32(act().at(14270464).cast(), act().at(0).cast(), 16);
    sync_cores();
    k_gna_4096_512_silu(act().at(14270464).cast(), w().at(151552).cast(), w().at(184320).cast(), act().at(0).cast(), act().at(14270464).cast(), 16);
    sync_cores();
    k_nchw2nhwc_64_512_p512(act().at(14270464).cast(), act().at(4800512).cast());
    sync_cores();
    k_conv_64_512to512_k3_p512_512(act().at(4800512).cast(), w().at(217088).cast(), w().at(4935680).cast(), act().at(9535488).cast());
    sync_cores();
    k_nhwc2nchw_64_512_p512(act().at(9535488).cast(), act().at(14270464).cast());
    sync_cores();
    k_gns_4096_512_32(act().at(14270464).cast(), act().at(0).cast(), 16);
    sync_cores();
    k_gna_4096_512_silu(act().at(14270464).cast(), w().at(4939776).cast(), w().at(4972544).cast(), act().at(0).cast(), act().at(14270464).cast(), 16);
    sync_cores();
    k_nchw2nhwc_64_512_p512(act().at(14270464).cast(), act().at(4800512).cast());
    sync_cores();
    k_conv_64_512to512_k3_p512_512_r(act().at(4800512).cast(), w().at(5005312).cast(), w().at(9723904).cast(), act().at(65536).cast(), act().at(9535488).cast());
    sync_cores();
    k_nhwc2nchw_64_512_p512(act().at(9535488).cast(), act().at(14270464).cast());
    sync_cores();
    k_gns_4096_512_32(act().at(14270464).cast(), act().at(0).cast(), 16);
    sync_cores();
    k_gna_4096_512_none(act().at(14270464).cast(), w().at(9728000).cast(), w().at(9760768).cast(), act().at(0).cast(), act().at(14270464).cast(), 16);
    sync_cores();
    k_nchw2nhwc_64_512_p512(act().at(14270464).cast(), act().at(65536).cast());
    sync_cores();
    k_conv_64_512to512_k1_p512_512(act().at(65536).cast(), w().at(9793536).cast(), w().at(10317824).cast(), act().at(4800512).cast());
    sync_cores();
    if core_rank() == 0 { k_crop_64_p512_512(act().at(4800512).cast(), act().at(18464768).cast()); }
    sync_cores();
    k_conv_64_512to512_k1_p512_512(act().at(65536).cast(), w().at(10321920).cast(), w().at(10846208).cast(), act().at(4800512).cast());
    sync_cores();
    if core_rank() == 0 { k_crop_64_p512_512(act().at(4800512).cast(), act().at(22659072).cast()); }
    sync_cores();
    k_conv_64_512to512_k1_p512_512(act().at(65536).cast(), w().at(10850304).cast(), w().at(11374592).cast(), act().at(4800512).cast());
    sync_cores();
    if core_rank() == 0 { k_crop_64_p512_512(act().at(4800512).cast(), act().at(26853376).cast()); }
    sync_cores();
    k_t2d_4096_512(act().at(22659072).cast(), act().at(31047680).cast());
    sync_cores();
    sd_vae_attn(act().at(18464768).cast(), act().at(31047680).cast(), act().at(26853376).cast(), act().at(35241984).cast());
    sync_cores();
    if core_rank() == 0 { k_pad_64_p512_512(act().at(35241984).cast(), act().at(65536).cast()); }
    sync_cores();
    k_conv_64_512to512_k1_p512_512_r(act().at(65536).cast(), w().at(11378688).cast(), w().at(11902976).cast(), act().at(9535488).cast(), act().at(4800512).cast());
    sync_cores();
    k_nhwc2nchw_64_512_p512(act().at(4800512).cast(), act().at(14270464).cast());
    sync_cores();
    k_gns_4096_512_32(act().at(14270464).cast(), act().at(0).cast(), 16);
    sync_cores();
    k_gna_4096_512_silu(act().at(14270464).cast(), w().at(11907072).cast(), w().at(11939840).cast(), act().at(0).cast(), act().at(14270464).cast(), 16);
    sync_cores();
    k_nchw2nhwc_64_512_p512(act().at(14270464).cast(), act().at(65536).cast());
    sync_cores();
    k_conv_64_512to512_k3_p512_512(act().at(65536).cast(), w().at(11972608).cast(), w().at(16691200).cast(), act().at(9535488).cast());
    sync_cores();
    k_nhwc2nchw_64_512_p512(act().at(9535488).cast(), act().at(14270464).cast());
    sync_cores();
    k_gns_4096_512_32(act().at(14270464).cast(), act().at(0).cast(), 16);
    sync_cores();
    k_gna_4096_512_silu(act().at(14270464).cast(), w().at(16695296).cast(), w().at(16728064).cast(), act().at(0).cast(), act().at(14270464).cast(), 16);
    sync_cores();
    k_nchw2nhwc_64_512_p512(act().at(14270464).cast(), act().at(65536).cast());
    sync_cores();
    k_conv_64_512to512_k3_p512_512_r(act().at(65536).cast(), w().at(16760832).cast(), w().at(21479424).cast(), act().at(4800512).cast(), out_y().cast());
    sync_cores();
}
