#![allow(unused_macros)]

// ── 复现侧：原样调用 calBLAS 的 group_norm!（128²，512 通道，32 组）──
#[cfg(not(feature = "sram"))]
calblas::group_norm!(pub k_gn_16384_512_32_silu: hw = 16384, c = 512, groups = 32, eps = 1.0e-06f32, act = silu);

// ── 绕法侧：同一份代码，只把「一组切几段 NH」改成从 SRAM 读（同 rope! 的 pos.load() 写法）──
#[cfg(feature = "sram")]
macro_rules! gn_rt_sram {
    ($vis:vis $name:ident: hw = $hw:expr, c = $c:expr, groups = $g:expr, eps = $eps:expr, act = $act:ident) => {
        #[npu::kernel(cores = 2, numerics = "ks1")]
        $vis fn $name(
            x: npu::Ptr<npu::Ddr, npu::bf16>,
            w: npu::Ptr<npu::Ddr, f32>,
            b: npu::Ptr<npu::Ddr, f32>,
            y: npu::Ptr<npu::Ddr, npu::bf16>,
            nh: npu::Ptr<npu::Ram, i32>,
        ) {
            use npu::prelude::*;
            const HW: usize = $hw;
            const C: usize = $c;
            const G: usize = $g;
            const CG: usize = C / G;
            const CC: usize = calblas::kernels::Cols::norm_cols(CG * HW, HW);
            const R: usize = CG * HW / CC;
            const NH: usize = calblas::kernels::Cols::norm_parts(R, CC);
            const RP: usize = R / NH;
            const INV_N: f32 = 1.0 / (CG * HW) as f32;
            const _: () = assert!(C % G == 0 && HW % CC == 0 && RP % 8 == 0, "gn_rt_sram");
            // ⚠ 用 i32（rope! 的 pos.load() 就是这么写的）；Ptr<Ram, u32> 会报
            //   "failed to legalize unresolved materialization"
            let nh = (nh.load() as usize);
            let gx = x.bind_strided::<npu::Read, 2>([C * HW / CC, CC], [CC, 1]);
            let gy = y.bind_strided::<npu::Write, 2>([C * HW / CC, CC], [CC, 1]);
            let (gw, gb) = (calblas::rd!(w, [C * HW / CC, 8]), calblas::rd!(b, [C * HW / CC, 8]));
            let (h, f, t) = (vcache::<bf16, _>([RP, CC]), vcache::<f32, _>([RP, CC]), vcache::<f32, _>([RP, CC]));
            let (s1, sp, s) = (vcache::<f32, _>([RP, 1]), vcache::<f32, _>([1, 1]), vcache::<f32, _>([1, 1]));
            let (w8, wv, bv) = (vcache::<f32, _>([RP, 8]), vcache::<f32, _>([RP, 1]), vcache::<f32, _>([RP, 1]));
            let (g0, gn) = calblas::halves!(G);
            for i in 0..gn {
                let r0 = (g0 + i) * R;
                s.fill(0.0f32);
                for part in 0..nh {
                    h.load(gx.sub([r0 + part * RP, 0], [RP, CC]));
                    f.cvt(h, Cvt::default());
                    s1.reduce(Combine::Sum, f, 1);
                    sp.reduce(Combine::Sum, s1, 0);
                    s.add(s, sp);
                }
                s.mul(s, INV_N);
                let m = vcache::<f32, _>([1, 1]);
                m.copy(s);
                s.fill(0.0f32);
                for part in 0..nh {
                    h.load(gx.sub([r0 + part * RP, 0], [RP, CC]));
                    f.cvt(h, Cvt::default());
                    f.sub(f, bcast(m, &[0, 1]));
                    t.mul(f, f);
                    s1.reduce(Combine::Sum, t, 1);
                    sp.reduce(Combine::Sum, s1, 0);
                    s.add(s, sp);
                }
                s.mul(s, INV_N);
                s.add(s, $eps);
                s.rsqrt_approx(s);
                for part in 0..nh {
                    let rp = r0 + part * RP;
                    h.load(gx.sub([rp, 0], [RP, CC]));
                    f.cvt(h, Cvt::default());
                    f.sub(f, bcast(m, &[0, 1]));
                    f.mul(f, bcast(s, &[0, 1]));
                    w8.load(gw.sub([rp, 0], [RP, 8]));
                    wv.reduce(Combine::Max, w8, 1);
                    w8.load(gb.sub([rp, 0], [RP, 8]));
                    bv.reduce(Combine::Max, w8, 1);
                    f.mul(f, bcast(wv, &[0, 1]));
                    f.add(f, bcast(bv, &[0, 1]));
                    h.cvt(f, Cvt::default());
                    gn_rt_sram!(@$act h);
                    h.store(gy.sub([rp, 0], [RP, CC]));
                }
            }
        }
    };
    (@silu $h:ident) => {{
        let tmp = vcache::<bf16, _>([RP, CC]);
        calblas::logistic!(silu, $h, $h, tmp);
    }};
    (@none $h:ident) => {};
}

#[cfg(feature = "sram")]
gn_rt_sram!(pub k_gn_sram_16384_512_32_silu: hw = 16384, c = 512, groups = 32, eps = 1.0e-06f32, act = silu);
