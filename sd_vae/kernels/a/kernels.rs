//! SD1.5 VAE decoder kernel 特化（gen.py 生成，勿手改）

calblas::conv2d!(pub k_conv_64_16to512_k3_p16_512: h = 64, w = 64, cin = 16, cout = 512, k = 3, s = 1, pin = (16, 0), pout = (512, 0), act = none);
calblas::nhwc_to_nchw!(pub k_nhwc2nchw_64_512_p512: h = 64, w = 64, c = 512, p = 512);

/// GroupNorm 统计：x [C][HW] bf16 ⇒ stats [2C][16] f32（前 C 行 16 份 mean，后 C 行 16 份 rstd）。
/// 均值/方差在「组」上求：一组 CG = C/G 个通道 × HW 个像素（同组各通道共享同一 mean/rstd）。
#[npu::kernel(cores = 2, numerics = "ks1")]
pub fn k_gns_4096_512_32(x: npu::Ptr<npu::Ddr, npu::bf16>, stats: npu::Ptr<npu::Ddr, f32>,
                          nchunk: usize) {
    use npu::prelude::*;
    const C: usize = 512;
    const HW: usize = 4096;
    const CH: usize = 256;
    const G: usize = 32;
    const CG: usize = C / G;
    const INV_N: f32 = 1.525878906250e-05f32;
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
pub fn k_gna_4096_512_silu(x: npu::Ptr<npu::Ddr, npu::bf16>, gamma: npu::Ptr<npu::Ddr, f32>,
                            beta: npu::Ptr<npu::Ddr, f32>, stats: npu::Ptr<npu::Ddr, f32>,
                            y: npu::Ptr<npu::Ddr, npu::bf16>, nchunk: usize) {
    use npu::prelude::*;
    const C: usize = 512;
    const HW: usize = 4096;
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

calblas::nchw_to_nhwc!(pub k_nchw2nhwc_64_512_p512: h = 64, w = 64, c = 512, p = 512);
calblas::conv2d!(pub k_conv_64_512to512_k3_p512_512: h = 64, w = 64, cin = 512, cout = 512, k = 3, s = 1, pin = (512, 0), pout = (512, 0), act = none);
calblas::conv2d!(pub k_conv_64_512to512_k3_p512_512_r: h = 64, w = 64, cin = 512, cout = 512, k = 3, s = 1, pin = (512, 0), pout = (512, 0), act = none, res = (512, 0));

/// GroupNorm 应用（原地）：y = [(x − mean) · rstd · gamma + beta]（可选 silu），x/y [C][HW] bf16。
#[npu::kernel(cores = 2, numerics = "ks1")]
pub fn k_gna_4096_512_none(x: npu::Ptr<npu::Ddr, npu::bf16>, gamma: npu::Ptr<npu::Ddr, f32>,
                            beta: npu::Ptr<npu::Ddr, f32>, stats: npu::Ptr<npu::Ddr, f32>,
                            y: npu::Ptr<npu::Ddr, npu::bf16>, nchunk: usize) {
    use npu::prelude::*;
    const C: usize = 512;
    const HW: usize = 4096;
    const CH: usize = 256;
    const ACT: bool = false;
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

calblas::conv2d!(pub k_conv_64_512to512_k1_p512_512: h = 64, w = 64, cin = 512, cout = 512, k = 1, s = 1, pin = (512, 0), pout = (512, 0), act = none);
calblas::transpose2d!(pub k_t2d_4096_512: rows = 4096, cols = 512);
calblas::conv2d!(pub k_conv_64_512to512_k1_p512_512_r: h = 64, w = 64, cin = 512, cout = 512, k = 1, s = 1, pin = (512, 0), pout = (512, 0), act = none, res = (512, 0));

// 图像 slot 边框清零：n 是运行期标量（常量循环体会被展开成 1 MiB 代码，链接器分支够不着）
calblas::fill16k!(pub k_fill0: value = 0.0f32);

// ---------------------------------------------------------------- 手写搬运 kernel
//
// border NHWC 图（[H+4][H+4][PIN]，内部 [H][H] 像素）与紧排 [H·H][POUT] 互搬。
// 一次搬「一行图像 = H 个像素 × 256 个通道」：视图 [像素, 通道]（行距 PIN/POUT），
// 每个像素的 256 个通道连续（512 B = 一行 VB）。


/// crop：h=64, pin=512, pout=512（border ⇔ 紧排）
#[npu::kernel]
pub fn k_crop_64_p512_512(x: npu::Ptr<npu::Ddr, npu::bf16>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
    use npu::prelude::*;
    const H: usize = 64;
    const PIN: usize = 512;
    const POUT: usize = 512;
    const WI: usize = H + 4;
    const CH: usize = 256;
    const NC: usize = POUT / CH;
    let gx_border = x.add((2 * WI + 2) * PIN * 2).bind_strided::<npu::Read, 2>([H * WI, PIN], [PIN, 1]);
    let gy_border = y.add((2 * WI + 2) * PIN * 2).bind_strided::<npu::Write, 2>([H * WI, PIN], [PIN, 1]);
    let gx_tight = x.bind_strided::<npu::Read, 2>([H * H, POUT], [POUT, 1]);
    let gy_tight = y.bind_strided::<npu::Write, 2>([H * H, POUT], [POUT, 1]);
    let t = vcache::<bf16, _>([H, CH]);
    for r in 0..H {
        for c in 0..NC {
            if true {
                t.load(gx_border.sub([r * WI, c * CH], [H, CH]));
                t.store(gy_tight.sub([r * H, c * CH], [H, CH]));
            } else {
                t.load(gx_tight.sub([r * H, c * CH], [H, CH]));
                t.store(gy_border.sub([r * WI, c * CH], [H, CH]));
            }
        }
    }
}


/// pad：h=64, pin=512, pout=512（border ⇔ 紧排）
#[npu::kernel]
pub fn k_pad_64_p512_512(x: npu::Ptr<npu::Ddr, npu::bf16>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
    use npu::prelude::*;
    const H: usize = 64;
    const PIN: usize = 512;
    const POUT: usize = 512;
    const WI: usize = H + 4;
    const CH: usize = 256;
    const NC: usize = POUT / CH;
    let gx_border = x.add((2 * WI + 2) * PIN * 2).bind_strided::<npu::Read, 2>([H * WI, PIN], [PIN, 1]);
    let gy_border = y.add((2 * WI + 2) * PIN * 2).bind_strided::<npu::Write, 2>([H * WI, PIN], [PIN, 1]);
    let gx_tight = x.bind_strided::<npu::Read, 2>([H * H, POUT], [POUT, 1]);
    let gy_tight = y.bind_strided::<npu::Write, 2>([H * H, POUT], [POUT, 1]);
    let t = vcache::<bf16, _>([H, CH]);
    for r in 0..H {
        for c in 0..NC {
            if false {
                t.load(gx_border.sub([r * WI, c * CH], [H, CH]));
                t.store(gy_tight.sub([r * H, c * CH], [H, CH]));
            } else {
                t.load(gx_tight.sub([r * H, c * CH], [H, CH]));
                t.store(gy_border.sub([r * WI, c * CH], [H, CH]));
            }
        }
    }
}


// ---------------------------------------------------------------- 手写：单头 flash 注意力（VAE mid attention）
//
// q / k / v / o 都是紧排 [T, D] bf16；kT 是 [D, T]（transpose2d 转好）。
// 两遍：第一遍逐 key 块算分数取行最大；第二遍重算分数、exp、行和，V 也分块累加。
// 单头、无 mask、scale = 1/√D；query 块两核各一半。


/// VAE mid attention：T=4096, D=512, 单头，无 mask，无残差（残差在图外面）。
///
/// mma 累加链要避免「分数链 + 4 个 PV 累加器链」同时在飞（docc calt-fate 会拒）：
/// 第一遍只算分数取行最大；第二遍先算 p 全存进 VB（32 块 × 16 行 × 128），再单独一段 4 个
/// PV 累加器按块累加（次序固定，和 conv FAST 路的四个累加器同构）。
#[npu::kernel(cores = 2, numerics = "ks1")]
pub fn sd_vae_attn(q: npu::Ptr<npu::Ddr, npu::bf16>, kt: npu::Ptr<npu::Ddr, npu::bf16>,
                   v: npu::Ptr<npu::Ddr, npu::bf16>, o: npu::Ptr<npu::Ddr, npu::bf16>) {
    use npu::prelude::*;
    const T: usize = 4096;
    const D: usize = 512;
    const KB: usize = 128;
    const NB: usize = T / KB;
    const DC: usize = 128;
    const ND: usize = D / DC;
    const SCALE: f32 = 0.044194174f32;
    let gq = q.bind_strided::<npu::Read, 2>([T, D], [D, 1]);
    let gkt = kt.bind_strided::<npu::Read, 2>([D, T], [T, 1]);
    let gv = v.bind_strided::<npu::Read, 2>([T, D], [D, 1]);
    let go = o.bind_strided::<npu::Write, 2>([T, D], [D, 1]);
    let (lq, rk, sacc) = (mcache::<bf16, _>([16, DC]), mcache::<bf16, _>([DC, KB]),
                          mcache::<f32, _>([16, KB]));
    let (lp, rv) = (mcache::<bf16, _>([16, KB]), mcache::<bf16, _>([KB, DC]));
    let (a0, a1) = (mcache::<f32, _>([16, DC]), mcache::<f32, _>([16, DC]));
    let (a2, a3) = (mcache::<f32, _>([16, DC]), mcache::<f32, _>([16, DC]));
    let (sf, sb, tf) = (vcache::<f32, _>([16, KB]), vcache::<bf16, _>([16, KB]),
                        vcache::<f32, _>([16, KB]));
    let (m, top, mb) = (vcache::<f32, _>([16, 1]), vcache::<f32, _>([16, 1]),
                        vcache::<bf16, _>([16, 1]));
    let (l, rs, lb) = (vcache::<f32, _>([16, 1]), vcache::<f32, _>([16, 1]),
                       vcache::<bf16, _>([16, 1]));
    let (out, outb) = (vcache::<f32, _>([16, DC]), vcache::<bf16, _>([16, DC]));
    let pall = vcache::<bf16, _>([NB * 16, KB]);
    const QB: usize = T / 16;
    let mine = core_rank() * (QB / num_cores());
    let cnt = if core_rank() + 1 < num_cores() { QB / num_cores() } else { QB - mine };
    for i in 0..cnt {
        let qb = mine + i;
        // ── 第一遍：行最大
        m.fill(-3.0e38f32);
        for kb in 0..NB {
            sacc.fill(0.0f32);
            for dk in 0..ND {
                lq.load(gq.sub([qb * 16, dk * DC], [16, DC]));
                rk.load(gkt.sub([dk * DC, kb * KB], [DC, KB]));
                sacc.mma(lq, rk);
            }
            sf.from_m(sacc);
            sb.cvt(sf, Cvt::default());
            sb.mul(sb, SCALE);
            tf.cvt(sb, Cvt::default());
            top.reduce(Combine::Max, tf, 1);
            m.max(m, top);
        }
        mb.cvt(m, Cvt::default());
        // ── 第二遍 a：p = exp(s − max)，存进 VB，同时累计行和
        l.fill(0.0f32);
        for kb in 0..NB {
            sacc.fill(0.0f32);
            for dk in 0..ND {
                lq.load(gq.sub([qb * 16, dk * DC], [16, DC]));
                rk.load(gkt.sub([dk * DC, kb * KB], [DC, KB]));
                sacc.mma(lq, rk);
            }
            sf.from_m(sacc);
            sb.cvt(sf, Cvt::default());
            sb.mul(sb, SCALE);
            sb.sub(sb, bcast(mb, &[0, 1]));
            sb.exp_approx(sb);
            tf.cvt(sb, Cvt::default());
            rs.reduce(Combine::Sum, tf, 1);
            l.add(l, rs);
            pall.rows(kb * 16, 16).copy(sb);
        }
        lb.cvt(l, Cvt::default());
        // ── 第二遍 b：out = Σ p·v（4 个头维块，累加器次序固定）
        a0.fill(0.0f32);
        a1.fill(0.0f32);
        a2.fill(0.0f32);
        a3.fill(0.0f32);
        for kb in 0..NB {
            lp.load_v(pall.rows(kb * 16, 16));
            rv.load(gv.sub([kb * KB, 0 * DC], [KB, DC]));
            a0.mma(lp, rv);
            rv.load(gv.sub([kb * KB, 1 * DC], [KB, DC]));
            a1.mma(lp, rv);
            rv.load(gv.sub([kb * KB, 2 * DC], [KB, DC]));
            a2.mma(lp, rv);
            rv.load(gv.sub([kb * KB, 3 * DC], [KB, DC]));
            a3.mma(lp, rv);
        }
        out.from_m(a0);
        outb.cvt(out, Cvt::default());
        outb.div(outb, bcast(lb, &[0, 1]));
        outb.store(go.sub([qb * 16, 0 * DC], [16, DC]));
        out.from_m(a1);
        outb.cvt(out, Cvt::default());
        outb.div(outb, bcast(lb, &[0, 1]));
        outb.store(go.sub([qb * 16, 1 * DC], [16, DC]));
        out.from_m(a2);
        outb.cvt(out, Cvt::default());
        outb.div(outb, bcast(lb, &[0, 1]));
        outb.store(go.sub([qb * 16, 2 * DC], [16, DC]));
        out.from_m(a3);
        outb.cvt(out, Cvt::default());
        outb.div(outb, bcast(lb, &[0, 1]));
        outb.store(go.sub([qb * 16, 3 * DC], [16, DC]));
    }
}
