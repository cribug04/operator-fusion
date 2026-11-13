
/// 双线性上采样 ×2（NHWC、无边框，bf16）—— **通道分块版**：输入 `[H][W][C]` ⇒ 输出 `[2H][2W][C]`。
///
/// 语义与 `resize_bilinear2!` 完全相同（`align_corners=False`、边缘 clamp、f32 中间计算、最后 RNE 舍 bf16），
/// 区别只在把 C 切成 `NC = C/CC` 块顺序处理：每块的中间 cache 只占 `W × CC` 个元素，因此 **支持更大的 C**
/// （原版 f32 中间 cache 一行 `C*4 <= 512 B` ⇒ C ≤ 128；分块版按 CC 算）。
///
/// 约束：`CC >= 32 && CC % 32 == 0 && C % CC == 0`（块偏移 `cc*CC*2 B` 是 64 B 整数倍，docc 可证对齐）；
/// `H, W >= 3`；`W*CC*56 B <= 384 KiB`（12 份 f32 + 4 份 bf16 中间 cache 的容量上限）。
/// 结构：`for cc in 0..NC`（**常数轮数**，docc/calt 允许与内层运行时循环组合，同 `conv2d!` 的 nb 外层）内嵌
/// “第 0 行 / `for iy in 1..H-1` / 最后一行”三段，全程无运行时条件分支。
#[macro_export]
macro_rules! resize_bilinear2_tiled {
    ($vis:vis $name:ident: h = $h:expr, w = $w:expr, c = $c:expr, cc = $cc:expr) => {
        #[npu::kernel(numerics = "ks1")]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
            const H: usize = $h;
            const W: usize = $w;
            const C: usize = $c;
            const CC: usize = $cc;
            const NC: usize = C / CC;
            const _: () = assert!(H >= 3 && W >= 3, "resize_bilinear2_tiled：H、W 至少 3");
            const _: () = assert!(CC >= 32 && CC % 32 == 0 && C % CC == 0, "resize_bilinear2_tiled：CC 是 >=32 的 32 倍数且整除 C");
            const _: () = assert!(W * CC * 56 <= 384 * 1024, "resize_bilinear2_tiled：W*CC 太大，中间 cache 放不进 VB");
            for cci in 0..NC {
                let xoff = cci * CC;
                $crate::resize_bilinear2_tiled!(@pair x, y, $h, $w, $c, $cc, xoff, 0, 0, 1);
                for iy in 1..H - 1 {
                    $crate::resize_bilinear2_tiled!(@pair x, y, $h, $w, $c, $cc, xoff, iy - 1, iy, iy + 1);
                }
                $crate::resize_bilinear2_tiled!(@pair x, y, $h, $w, $c, $cc, xoff, $h - 2, $h - 1, $h - 1);
            }
        }
    };

    // 处理输入行 iy 对应的两条输出行 2iy / 2iy+1；ia/ic 是上/下邻行（边界处已 clamp 成常量/线性表达式）。
    (@pair $x:ident, $y:ident, $h:expr, $w:expr, $c:expr, $cc:expr, $xoff:expr, $ia:expr, $iy:expr, $ic:expr) => {{
        use npu::prelude::*;
        const H: usize = $h;
        const W: usize = $w;
        const C: usize = $c;
        const CC: usize = $cc;
        const WO: usize = 2 * W;
        let xoff: usize = $xoff;
        // 输入按“像素行距 C、通道块偏移 xoff”绑定；输出四个相位按“像素行距 2*C、通道块偏移 xoff”绑定
        let gx = $x.add(xoff * 2).bind_strided::<npu::Read, 3>([H, W, CC], [W * C, C, 1]);
        let g00 = $y.add(((0 * WO + 0) * C + xoff) * 2).bind_strided::<npu::Write, 2>([H * WO, CC], [2 * C, 1]);
        let g01 = $y.add(((0 * WO + 1) * C + xoff) * 2).bind_strided::<npu::Write, 2>([H * WO, CC], [2 * C, 1]);
        let g10 = $y.add(((1 * WO + 0) * C + xoff) * 2).bind_strided::<npu::Write, 2>([H * WO, CC], [2 * C, 1]);
        let g11 = $y.add(((1 * WO + 1) * C + xoff) * 2).bind_strided::<npu::Write, 2>([H * WO, CC], [2 * C, 1]);
        let (ab, bb, cb) = (vcache::<bf16, _>([W, CC]), vcache::<bf16, _>([W, CC]), vcache::<bf16, _>([W, CC]));
        let (af, bf, cf) = (vcache::<f32, _>([W, CC]), vcache::<f32, _>([W, CC]), vcache::<f32, _>([W, CC]));
        let (he_a, ho_a) = (vcache::<f32, _>([W, CC]), vcache::<f32, _>([W, CC]));
        let (he_b, ho_b) = (vcache::<f32, _>([W, CC]), vcache::<f32, _>([W, CC]));
        let (he_c, ho_c) = (vcache::<f32, _>([W, CC]), vcache::<f32, _>([W, CC]));
        let (oe, oo) = (vcache::<f32, _>([W, CC]), vcache::<f32, _>([W, CC]));
        let (ob, t) = (vcache::<bf16, _>([W, CC]), vcache::<f32, _>([W, CC]));
        ab.load(gx.sub([$ia, 0, 0], [1, W, CC]));
        bb.load(gx.sub([$iy, 0, 0], [1, W, CC]));
        cb.load(gx.sub([$ic, 0, 0], [1, W, CC]));
        af.cvt(ab, Cvt::default());
        bf.cvt(bb, Cvt::default());
        cf.cvt(cb, Cvt::default());
        $crate::resize_bilinear2_tiled!(@h af, he_a, ho_a, t, W);
        $crate::resize_bilinear2_tiled!(@h bf, he_b, ho_b, t, W);
        $crate::resize_bilinear2_tiled!(@h cf, he_c, ho_c, t, W);
        oe.mul(he_a, 0.25f32);
        t.mul(he_b, 0.75f32);
        oe.add(oe, t);
        ob.cvt(oe, Cvt::default());
        ob.store(g00.sub([$iy * WO, 0], [W, CC]));
        oo.mul(ho_a, 0.25f32);
        t.mul(ho_b, 0.75f32);
        oo.add(oo, t);
        ob.cvt(oo, Cvt::default());
        ob.store(g01.sub([$iy * WO, 0], [W, CC]));
        oe.mul(he_b, 0.75f32);
        t.mul(he_c, 0.25f32);
        oe.add(oe, t);
        ob.cvt(oe, Cvt::default());
        ob.store(g10.sub([$iy * WO, 0], [W, CC]));
        oo.mul(ho_b, 0.75f32);
        t.mul(ho_c, 0.25f32);
        oo.add(oo, t);
        ob.cvt(oo, Cvt::default());
        ob.store(g11.sub([$iy * WO, 0], [W, CC]));
    }};

    // 行内双线性（与 resize_bilinear2! 完全一致）：he[x]=0.75f[x]+0.25f[x-1]，ho[x]=0.75f[x]+0.25f[x+1]，边缘复制。
    (@h $f:ident, $he:ident, $ho:ident, $t:ident, $w:expr) => {{
        $he.mul($f, 0.75f32);
        $t.rows(0, $w - 1).mul($f.rows(0, $w - 1), 0.25f32);
        $he.rows(1, $w - 1).add($he.rows(1, $w - 1), $t.rows(0, $w - 1));
        $he.rows(0, 1).copy($f.rows(0, 1));
        $ho.mul($f, 0.75f32);
        $t.rows(0, $w - 1).mul($f.rows(1, $w - 1), 0.25f32);
        $ho.rows(0, $w - 1).add($ho.rows(0, $w - 1), $t.rows(0, $w - 1));
        $ho.rows($w - 1, 1).copy($f.rows($w - 1, 1));
    }};
}
