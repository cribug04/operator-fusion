//! SD1.5 VAE decoder 整步程序（gen.py 生成，勿手改）。
use crate::kernels::*;
use npu::prelude::*;

npu::decls! {
    in_x: input, bytes = 4734976;
    out_y: output, bytes = 17842176;
    w: weights, bytes = 33452032;
    act: workspace, bytes = 32112640;
}

#[npu::program(chiplets = 1, cores = 2)]
pub fn step() {
    // 所有图像 slot 一次清零：边框之后不再被写
    if core_rank() == 0 {
        k_fill0(act().at(65536).cast(), 1089);
        k_fill0(act().at(17907712).cast(), 289);
        k_fill0(act().at(22642688).cast(), 289);
        k_fill0(act().at(27377664).cast(), 289);
    }
    sync_cores();
    if core_rank() == 0 { k_nhwc2nchw_64_512_p512(in_x().cast(), act().at(32112640).cast()); }
    sync_cores();
    k_gns_4096_512_32(act().at(32112640).cast(), act().at(0).cast(), 16);
    sync_cores();
    k_gna_4096_512_silu(act().at(32112640).cast(), w().at(0).cast(), w().at(32768).cast(), act().at(0).cast(), act().at(32112640).cast(), 16);
    sync_cores();
    if core_rank() == 0 { k_nchw2nhwc_64_512_p512(act().at(32112640).cast(), act().at(17907712).cast()); }
    sync_cores();
    k_conv_64_512to512_k3_p512_512(act().at(17907712).cast(), w().at(65536).cast(), w().at(4784128).cast(), act().at(22642688).cast());
    sync_cores();
    if core_rank() == 0 { k_nhwc2nchw_64_512_p512(act().at(22642688).cast(), act().at(32112640).cast()); }
    sync_cores();
    k_gns_4096_512_32(act().at(32112640).cast(), act().at(0).cast(), 16);
    sync_cores();
    k_gna_4096_512_silu(act().at(32112640).cast(), w().at(4788224).cast(), w().at(4820992).cast(), act().at(0).cast(), act().at(32112640).cast(), 16);
    sync_cores();
    if core_rank() == 0 { k_nchw2nhwc_64_512_p512(act().at(32112640).cast(), act().at(17907712).cast()); }
    sync_cores();
    k_conv_64_512to512_k3_p512_512_r(act().at(17907712).cast(), w().at(4853760).cast(), w().at(9572352).cast(), in_x().cast(), act().at(22642688).cast());
    sync_cores();
    if core_rank() == 0 { k_nhwc2nchw_64_512_p512(act().at(22642688).cast(), act().at(32112640).cast()); }
    sync_cores();
    k_gns_4096_512_32(act().at(32112640).cast(), act().at(0).cast(), 16);
    sync_cores();
    k_gna_4096_512_silu(act().at(32112640).cast(), w().at(9576448).cast(), w().at(9609216).cast(), act().at(0).cast(), act().at(32112640).cast(), 16);
    sync_cores();
    if core_rank() == 0 { k_nchw2nhwc_64_512_p512(act().at(32112640).cast(), act().at(17907712).cast()); }
    sync_cores();
    k_conv_64_512to512_k3_p512_512(act().at(17907712).cast(), w().at(9641984).cast(), w().at(14360576).cast(), act().at(27377664).cast());
    sync_cores();
    if core_rank() == 0 { k_nhwc2nchw_64_512_p512(act().at(27377664).cast(), act().at(32112640).cast()); }
    sync_cores();
    k_gns_4096_512_32(act().at(32112640).cast(), act().at(0).cast(), 16);
    sync_cores();
    k_gna_4096_512_silu(act().at(32112640).cast(), w().at(14364672).cast(), w().at(14397440).cast(), act().at(0).cast(), act().at(32112640).cast(), 16);
    sync_cores();
    if core_rank() == 0 { k_nchw2nhwc_64_512_p512(act().at(32112640).cast(), act().at(17907712).cast()); }
    sync_cores();
    k_conv_64_512to512_k3_p512_512_r(act().at(17907712).cast(), w().at(14430208).cast(), w().at(19148800).cast(), act().at(22642688).cast(), act().at(27377664).cast());
    sync_cores();
    if core_rank() == 0 { k_nhwc2nchw_64_512_p512(act().at(27377664).cast(), act().at(32112640).cast()); }
    sync_cores();
    k_gns_4096_512_32(act().at(32112640).cast(), act().at(0).cast(), 16);
    sync_cores();
    k_gna_4096_512_silu(act().at(32112640).cast(), w().at(19152896).cast(), w().at(19185664).cast(), act().at(0).cast(), act().at(32112640).cast(), 16);
    sync_cores();
    if core_rank() == 0 { k_nchw2nhwc_64_512_p512(act().at(32112640).cast(), act().at(17907712).cast()); }
    sync_cores();
    k_conv_64_512to512_k3_p512_512(act().at(17907712).cast(), w().at(19218432).cast(), w().at(23937024).cast(), act().at(22642688).cast());
    sync_cores();
    if core_rank() == 0 { k_nhwc2nchw_64_512_p512(act().at(22642688).cast(), act().at(32112640).cast()); }
    sync_cores();
    k_gns_4096_512_32(act().at(32112640).cast(), act().at(0).cast(), 16);
    sync_cores();
    k_gna_4096_512_silu(act().at(32112640).cast(), w().at(23941120).cast(), w().at(23973888).cast(), act().at(0).cast(), act().at(32112640).cast(), 16);
    sync_cores();
    if core_rank() == 0 { k_nchw2nhwc_64_512_p512(act().at(32112640).cast(), act().at(17907712).cast()); }
    sync_cores();
    k_conv_64_512to512_k3_p512_512_r(act().at(17907712).cast(), w().at(24006656).cast(), w().at(28725248).cast(), act().at(27377664).cast(), act().at(22642688).cast());
    sync_cores();
    if core_rank() == 0 { k_up_64_256_p512_512(act().at(22642688).add(0).cast(), act().at(65536).add(0).cast()); }
    sync_cores();
    if core_rank() == 0 { k_up_64_256_p512_512(act().at(22642688).add(512).cast(), act().at(65536).add(512).cast()); }
    sync_cores();
    k_conv_128_512to512_k3_p512_512(act().at(65536).cast(), w().at(28729344).cast(), w().at(33447936).cast(), out_y().cast());
    sync_cores();
}
