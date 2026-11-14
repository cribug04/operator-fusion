
/// 双线性上采样 ×2（NHWC、无边框，bf16）：输入 x `[H][W][C]` ⇒ 输出 y `[2H][2W][C]`。
///
/// 语义 = `torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)`
/// 的 2× 情形：输出像素 (oy, ox) 的源坐标 `u = (ox + 0.5) / 2 - 0.5`，边缘 clamp 到 `[0, W-1] / [0, H-1]`。
/// 中间按 f32 计算（每一步 mul/add 各舍一次），最后一次性 RNE 舍回 bf16；权重只有 0.25 / 0.75（编译期常数）。
/// 约束：`C <= 128`（f32 中间 cache 一行 `C*4 B <= 512 B`），`H, W >= 3`，尺寸均为编译期常量。
///
/// 实现要点：**无运行时条件分支**（docc 的 calvx-to-calt 目前不收「for 体里带结果的 if」）——
/// 第 0 行、中间行（`for iy in 1..H-1`，边界都是常量）、最后一行三段展开，行号 clamp 直接写成常量/线性表达式。
#[macro_export]
macro_rules! resize_bilinear2 {
    ($vis:vis $name:ident: h = $h:expr, w = $w:expr, c = $c:expr) => {
        $crate::desc!($vis $name: ResizeDesc { h: $h as u32, w: $w as u32, c: $c as u32, scale: 2, mode: $crate::caldnn::Interp::Bilinear });
        #[npu::kernel(numerics = "ks1")]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
            const H: usize = $h;
            const W: usize = $w;
            const C: usize = $c;
            const _: () = assert!(H >= 3 && W >= 3, "resize_bilinear2：H、W 至少 3（第 0 行/最后一行单独展开）");
            const _: () = assert!(C > 0 && C <= 128, "resize_bilinear2：C 必须 1..=128（f32 中间 cache 一行 ≤ 512 B）");

            // 第 0 行：输出行 0/1 用输入行 (0, 0, 1)
            $crate::resize_bilinear2!(@pair x, y, $h, $w, $c, 0, 0, 1);
            // 中间行：输入行 (iy-1, iy, iy+1)
            for iy in 1..H - 1 {
                $crate::resize_bilinear2!(@pair x, y, $h, $w, $c, iy - 1, iy, iy + 1);
            }
            // 最后一行：输出行 2H-2/2H-1 用输入行 (H-2, H-1, H-1)
            $crate::resize_bilinear2!(@pair x, y, $h, $w, $c, $h - 2, $h - 1, $h - 1);
        }
    };

    // 处理一个「输入行 iy」对应的两条输出行（2iy、2iy+1），ia/ic 是上下邻行（已按边界 clamp 好的常量/线性表达式）。
    (@pair $x:ident, $y:ident, $h:expr, $w:expr, $c:expr, $ia:expr, $iy:expr, $ic:expr) => {{
        use npu::prelude::*;
        const H: usize = $h;
        const W: usize = $w;
        const C: usize = $c;
        const WO: usize = 2 * W;
        let gx = $x.bind_strided::<npu::Read, 3>([H, W, C], [W * C, C, 1]);
        // 输出四个相位（行 2iy+a、列 2j+b）：视图行距 2C，行 r 对应输出像素 2r；起点偏 a*WO*C + b*C
        let g00 = $y.add((0 * WO + 0) * C * 2).bind_strided::<npu::Write, 2>([H * WO, C], [2 * C, 1]);
        let g01 = $y.add((0 * WO + 1) * C * 2).bind_strided::<npu::Write, 2>([H * WO, C], [2 * C, 1]);
        let g10 = $y.add((1 * WO + 0) * C * 2).bind_strided::<npu::Write, 2>([H * WO, C], [2 * C, 1]);
        let g11 = $y.add((1 * WO + 1) * C * 2).bind_strided::<npu::Write, 2>([H * WO, C], [2 * C, 1]);
        // 三行输入（bf16 与 f32 各一份）+ 六条行内插值结果 + 两条输出 + 临时
        let (ab, bb, cb) = (vcache::<bf16, _>([W, C]), vcache::<bf16, _>([W, C]), vcache::<bf16, _>([W, C]));
        let (af, bf, cf) = (vcache::<f32, _>([W, C]), vcache::<f32, _>([W, C]), vcache::<f32, _>([W, C]));
        let (he_a, ho_a) = (vcache::<f32, _>([W, C]), vcache::<f32, _>([W, C]));
        let (he_b, ho_b) = (vcache::<f32, _>([W, C]), vcache::<f32, _>([W, C]));
        let (he_c, ho_c) = (vcache::<f32, _>([W, C]), vcache::<f32, _>([W, C]));
        let (oe, oo) = (vcache::<f32, _>([W, C]), vcache::<f32, _>([W, C]));
        let (ob, t) = (vcache::<bf16, _>([W, C]), vcache::<f32, _>([W, C]));
        ab.load(gx.sub([$ia, 0, 0], [1, W, C]));
        bb.load(gx.sub([$iy, 0, 0], [1, W, C]));
        cb.load(gx.sub([$ic, 0, 0], [1, W, C]));
        af.cvt(ab, Cvt::default());
        bf.cvt(bb, Cvt::default());
        cf.cvt(cb, Cvt::default());
        $crate::resize_bilinear2!(@h af, he_a, ho_a, t, W);
        $crate::resize_bilinear2!(@h bf, he_b, ho_b, t, W);
        $crate::resize_bilinear2!(@h cf, he_c, ho_c, t, W);

        // 输出行 2iy = 0.25*row(iy-1) + 0.75*row(iy)
        oe.mul(he_a, 0.25f32);
        t.mul(he_b, 0.75f32);
        oe.add(oe, t);
        ob.cvt(oe, Cvt::default());
        ob.store(g00.sub([$iy * WO, 0], [W, C]));
        oo.mul(ho_a, 0.25f32);
        t.mul(ho_b, 0.75f32);
        oo.add(oo, t);
        ob.cvt(oo, Cvt::default());
        ob.store(g01.sub([$iy * WO, 0], [W, C]));

        // 输出行 2iy+1 = 0.75*row(iy) + 0.25*row(iy+1)
        oe.mul(he_b, 0.75f32);
        t.mul(he_c, 0.25f32);
        oe.add(oe, t);
        ob.cvt(oe, Cvt::default());
        ob.store(g10.sub([$iy * WO, 0], [W, C]));
        oo.mul(ho_b, 0.75f32);
        t.mul(ho_c, 0.25f32);
        oo.add(oo, t);
        ob.cvt(oo, Cvt::default());
        ob.store(g11.sub([$iy * WO, 0], [W, C]));
    }};

    // 行内双线性：he[x] = 0.75*f[x] + 0.25*f[x-1]（x=0 用 f[0]）；ho[x] = 0.75*f[x] + 0.25*f[x+1]（x=W-1 用 f[W-1]）
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
