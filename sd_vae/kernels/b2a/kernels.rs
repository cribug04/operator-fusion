//! SD1.5 VAE decoder kernel 特化（gen.py 生成，勿手改）

calblas::nhwc_to_nchw!(pub k_nhwc2nchw_128_512_p512: h = 128, w = 128, c = 512, p = 512);

/// GroupNorm 统计：x [C][HW] bf16 ⇒ stats [2C][16] f32（前 C 行 16 份 mean，后 C 行 16 份 rstd）。
/// 均值/方差在「组」上求：一组 CG = C/G 个通道 × HW 个像素（同组各通道共享同一 mean/rstd）。
#[npu::kernel(cores = 2, numerics = "ks1")]
pub fn k_gns_16384_512_32(x: npu::Ptr<npu::Ddr, npu::bf16>, stats: npu::Ptr<npu::Ddr, f32>,
                          nchunk: usize) {
    use npu::prelude::*;
    const C: usize = 512;
    const HW: usize = 16384;
    const CH: usize = 256;
    const G: usize = 32;
    const CG: usize = C / G;
    const INV_N: f32 = 3.814697265625e-06f32;
    let gx = x.bind_strided::<npu::Read, 2>([C, HW], [HW, 1]);
    let gs = stats.bind_strided::<npu::Write, 2>([2 * C, 16], [16, 1]);
    let (bb, f) = (vcache::<bf16, _>([1, CH]), vcache::<f32, _>([1, CH]));
    let (sm, sq) = (vcache::<f32, _>([1, 1]), vcache::<f32, _>([1, 1]));
    let t = vcache::<f32, _>([1, CH]);
    let (s1, m2, v16) = (vcache::<f32, _>([1, 1]), vcache::<f32, _>([1, 1]),
                         vcache::<f32, _>([1, 16]));
    let (g0, gn) = calblas::halves!(G);
    for gi in 0..gn {
        let g = g0 + gi;
        sm.fill(0.0f32);
        sq.fill(0.0f32);
        for ci in 0..CG {
            let c = g * CG + ci;
            for j in 0..nchunk {
                bb.load(gx.sub([c, j * CH], [1, CH]));
                f.cvt(bb, Cvt::default());
                s1.reduce(Combine::Sum, f, 1);
                sm.add(sm, s1);
                t.mul(f, f);
                s1.reduce(Combine::Sum, t, 1);
                sq.add(sq, s1);
            }
        }
        sm.mul(sm, INV_N);
        sq.mul(sq, INV_N);
        m2.mul(sm, sm);
        sq.sub(sq, m2);
        sq.add(sq, 1.0e-6f32);
        sq.rsqrt_approx(sq);
        for ci in 0..CG {
            let c = g * CG + ci;
            v16.fill(0.0f32);
            v16.add(v16, bcast(sm, &[0, 1]));
            v16.store(gs.sub([c, 0], [1, 16]));
            v16.fill(0.0f32);
            v16.add(v16, bcast(sq, &[0, 1]));
            v16.store(gs.sub([C + c, 0], [1, 16]));
        }
    }
}


/// GroupNorm 应用（原地）：y = [(x − mean) · rstd · gamma + beta]（可选 silu），x/y [C][HW] bf16。
#[npu::kernel(cores = 2, numerics = "ks1")]
pub fn k_gna_16384_512_silu(x: npu::Ptr<npu::Ddr, npu::bf16>, gamma: npu::Ptr<npu::Ddr, f32>,
                            beta: npu::Ptr<npu::Ddr, f32>, stats: npu::Ptr<npu::Ddr, f32>,
                            y: npu::Ptr<npu::Ddr, npu::bf16>, nchunk: usize) {
    use npu::prelude::*;
    const C: usize = 512;
    const HW: usize = 16384;
    const CH: usize = 256;
    const ACT: bool = true;
    let gx = x.bind_strided::<npu::Read, 2>([C, HW], [HW, 1]);
    let gy = y.bind_strided::<npu::Write, 2>([C, HW], [HW, 1]);
    let gs = stats.bind_strided::<npu::Read, 2>([2 * C, 16], [16, 1]);
    let (gg, gb2) = (gamma.bind_strided::<npu::Read, 2>([C, 16], [16, 1]),
                     beta.bind_strided::<npu::Read, 2>([C, 16], [16, 1]));
    let (bb, f) = (vcache::<bf16, _>([1, CH]), vcache::<f32, _>([1, CH]));
    let (m16, mm) = (vcache::<f32, _>([1, 16]), vcache::<f32, _>([1, 1]));
    let (rstd, ga, be) = (vcache::<f32, _>([1, 1]), vcache::<f32, _>([1, 1]),
                          vcache::<f32, _>([1, 1]));
    let tmp = vcache::<bf16, _>([1, CH]);
    const CORE: usize = C / 2;
    let mine = core_rank() * CORE;
    for i in 0..CORE {
        let c = mine + i;
        m16.load(gs.sub([c, 0], [1, 16]));
        mm.reduce(Combine::Max, m16, 1);
        m16.load(gs.sub([C + c, 0], [1, 16]));
        rstd.reduce(Combine::Max, m16, 1);
        m16.load(gg.sub([c, 0], [1, 16]));
        ga.reduce(Combine::Max, m16, 1);
        m16.load(gb2.sub([c, 0], [1, 16]));
        be.reduce(Combine::Max, m16, 1);
        for j in 0..nchunk {
            bb.load(gx.sub([c, j * CH], [1, CH]));
            f.cvt(bb, Cvt::default());
            f.sub(f, bcast(mm, &[0, 1]));
            f.mul(f, bcast(rstd, &[0, 1]));
            f.mul(f, bcast(ga, &[0, 1]));
            f.add(f, bcast(be, &[0, 1]));
            bb.cvt(f, Cvt::default());
            if ACT {
                calblas::logistic!(silu, bb, bb, tmp);
            }
            bb.store(gy.sub([c, j * CH], [1, CH]));
        }
    }
}

calblas::nchw_to_nhwc!(pub k_nchw2nhwc_128_512_p512: h = 128, w = 128, c = 512, p = 512);
calblas::conv2d!(pub k_conv_128_512to512_k3_p512_512: h = 128, w = 128, cin = 512, cout = 512, k = 3, s = 1, pin = (512, 0), pout = (512, 0), act = none);
calblas::conv2d!(pub k_conv_128_512to512_k3_p512_512_r: h = 128, w = 128, cin = 512, cout = 512, k = 3, s = 1, pin = (512, 0), pout = (512, 0), act = none, res = (512, 0));

// 图像 slot 边框清零：n 是运行期标量（常量循环体会被展开成 1 MiB 代码，链接器分支够不着）
calblas::fill16k!(pub k_fill0: value = 0.0f32);

// ---------------------------------------------------------------- 手写搬运 kernel
//
// border NHWC 图（[H+4][H+4][PIN]，内部 [H][H] 像素）与紧排 [H·H][POUT] 互搬。
// 一次搬「一行图像 = H 个像素 × 256 个通道」：视图 [像素, 通道]（行距 PIN/POUT），
// 每个像素的 256 个通道连续（512 B = 一行 VB）。


// ---------------------------------------------------------------- 手写：单头 flash 注意力（VAE mid attention）
//
// q / k / v / o 都是紧排 [T, D] bf16；kT 是 [D, T]（transpose2d 转好）。
// 两遍：第一遍逐 key 块算分数取行最大；第二遍重算分数、exp、行和，V 也分块累加。
// 单头、无 mask、scale = 1/√D；query 块两核各一半。
