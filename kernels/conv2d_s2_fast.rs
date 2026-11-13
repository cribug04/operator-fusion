
/// **实验性** 3×3 / stride=2 卷积的 FAST 版（NHWC、四周 2 像素边框、bf16；`y = silu(conv(x)+b)`）。
///
/// 背景：`conv2d!` 的 FAST 路只覆盖 `S == 1`（`vision.rs:122`），stride-2 卷积（YOLO/UNet 的下采样层）只能走慢路。
/// 本宏的做法：对每个输出行 `oy` 把 3 个输入行（padded 行 `oy*2+ky+1`）各自拆成 **偶/奇像素两份 VB cache**
/// （odd 从 padded 列 `B-1` 起、even 从 `B` 起，两份都按像素行距 `2*CI` 直接绑视图加载），于是 tap `(ky, kx)` 需要的 lhs 永远是某一份 cache 里
/// **连续 16 行**：`kx=0 → odd[ky] 行 mb*16`；`kx=1 → even[ky] 行 mb*16`；`kx=2 → odd[ky] 行 mb*16+1`。
/// 这样 stride-2 也能享受“输入行只从 DDR 读一次、9 个 tap 从 VB 取 lhs”快路。
///
/// 约束/简化（探针版）：3×3、s=2、`W % 16 == 0`、`CI/CO % 16 == 0`、输入 `[H*2+4][W*2+4][CI]` 紧排（无通道窗口）、
/// 权重 `[9][CI][CO]` bf16（BN 已折入）、bias f32 `[CO]`、`act = silu | none`、`cores = 2`（按输出行分）。
#[macro_export]
macro_rules! conv2d_s2_fast {
    ($vis:vis $name:ident: h = $h:expr, w = $w:expr, cin = $ci:expr, cout = $co:expr, act = $act:ident) => {
        #[npu::kernel(numerics = "ks1", cores = 2)]
        $vis fn $name(
            x: npu::Ptr<npu::Ddr, npu::bf16>,
            wt: npu::Ptr<npu::Ddr, npu::bf16>,
            bias: npu::Ptr<npu::Ddr, f32>,
            y: npu::Ptr<npu::Ddr, npu::bf16>,
        ) {
            use npu::prelude::*;
            const H: usize = $h;
            const W: usize = $w;
            const CI: usize = $ci;
            const CO: usize = $co;
            const B: usize = $crate::kernels::BORDER; // 2
            const WI: usize = W * 2 + 2 * B;          // 输入 padded 宽（像素）
            const WO: usize = W + 2 * B;              // 输出 padded 宽（像素）
            const HI: usize = H * 2 + 2 * B;          // 输入 padded 行数
            const EW: usize = W + 1;                  // 偶/奇 cache 需要的像素列数（0..=W）
            const _: () = assert!(B == 2 && W % 16 == 0, "conv2d_s2_fast：B=2、W 是 16 的倍数");
            const _: () = assert!(CI % 16 == 0 && CO % 16 == 0, "conv2d_s2_fast：CI/CO 是 16 的倍数");
            const NC: usize = $crate::kernels::Cols(CI, CO).nc();
            const NBLK: usize = CO / NC;

            // 输入紧排 [HI][WO][CI]：偶像素（padded x = B, B+2, …）/ 奇像素（padded x = B+1, B+3, …）
            let g_even = x.add(B * CI * 2).bind_strided::<npu::Read, 3>([HI, EW, CI], [WI * CI, 2 * CI, 1]);
            let g_odd = x.add((B - 1) * CI * 2).bind_strided::<npu::Read, 3>([HI, EW, CI], [WI * CI, 2 * CI, 1]);
            // 输出 [H+4][W+4][CO]（从内部左上角起）
            let gy = y.add(((B * WO + B) * CO) * 2).bind_strided::<npu::Write, 4>([H, W / 16, 16, CO], [WO * CO, 16 * CO, CO, 1]);

            let (ev0, od0) = (vcache::<bf16, _>([EW, CI]), vcache::<bf16, _>([EW, CI]));
            let (ev1, od1) = (vcache::<bf16, _>([EW, CI]), vcache::<bf16, _>([EW, CI]));
            let (ev2, od2) = (vcache::<bf16, _>([EW, CI]), vcache::<bf16, _>([EW, CI]));
            let gw = $crate::rd!(wt, [9 * CI, CO]);
            // 每个 nb 的 bias 一行（[NBLK][NC] 的一行）
            let (lhs, rhs) = (mcache::<bf16, _>([16, CI]), mcache::<bf16, _>([CI, NC]));
            let (acc, vo) = (mcache::<f32, _>([16, NC]), vcache::<f32, _>([16, NC]));
            let (vb, yh, tmp) = (vcache::<f32, _>([1, NC]), vcache::<bf16, _>([16, NC]), vcache::<bf16, _>([16, NC]));

            let (h0, hn) = $crate::halves!(H);
            for i in 0..hn {
                let oy = h0 + i;
                // padded 输入行 = oy*2 + ky + 1（padding = 1）
                ev0.load(g_even.sub([oy * 2 + 1, 0, 0], [1, EW, CI]));
                od0.load(g_odd.sub([oy * 2 + 1, 0, 0], [1, EW, CI]));
                ev1.load(g_even.sub([oy * 2 + 2, 0, 0], [1, EW, CI]));
                od1.load(g_odd.sub([oy * 2 + 2, 0, 0], [1, EW, CI]));
                ev2.load(g_even.sub([oy * 2 + 3, 0, 0], [1, EW, CI]));
                od2.load(g_odd.sub([oy * 2 + 3, 0, 0], [1, EW, CI]));
                for nb in 0..NBLK {
                    let gb = $crate::rd!(bias, [CO / NC, NC]);
                    vb.load(gb.sub([nb, 0], [1, NC]));
                    for mb in 0..W / 16 {
                        acc.fill(0.0f32);
                        // tap (0,0)：even[0] 行 mb*16
                        lhs.load_v(od0.rows(mb * 16, 16));
                        rhs.load(gw.sub([0 * CI, nb * NC], [CI, NC]));
                        acc.mma(lhs, rhs);
                        // tap (0,1)：odd[0]
                        lhs.load_v(ev0.rows(mb * 16, 16));
                        rhs.load(gw.sub([1 * CI, nb * NC], [CI, NC]));
                        acc.mma(lhs, rhs);
                        // tap (0,2)：even[0] 行 mb*16+1
                        lhs.load_v(od0.rows(mb * 16 + 1, 16));
                        rhs.load(gw.sub([2 * CI, nb * NC], [CI, NC]));
                        acc.mma(lhs, rhs);
                        // tap (1,0)：even[1]
                        lhs.load_v(od1.rows(mb * 16, 16));
                        rhs.load(gw.sub([3 * CI, nb * NC], [CI, NC]));
                        acc.mma(lhs, rhs);
                        // tap (1,1)：odd[1]
                        lhs.load_v(ev1.rows(mb * 16, 16));
                        rhs.load(gw.sub([4 * CI, nb * NC], [CI, NC]));
                        acc.mma(lhs, rhs);
                        // tap (1,2)：even[1] 行 mb*16+1
                        lhs.load_v(od1.rows(mb * 16 + 1, 16));
                        rhs.load(gw.sub([5 * CI, nb * NC], [CI, NC]));
                        acc.mma(lhs, rhs);
                        // tap (2,0)：even[2]
                        lhs.load_v(od2.rows(mb * 16, 16));
                        rhs.load(gw.sub([6 * CI, nb * NC], [CI, NC]));
                        acc.mma(lhs, rhs);
                        // tap (2,1)：odd[2]
                        lhs.load_v(ev2.rows(mb * 16, 16));
                        rhs.load(gw.sub([7 * CI, nb * NC], [CI, NC]));
                        acc.mma(lhs, rhs);
                        // tap (2,2)：even[2] 行 mb*16+1
                        lhs.load_v(od2.rows(mb * 16 + 1, 16));
                        rhs.load(gw.sub([8 * CI, nb * NC], [CI, NC]));
                        acc.mma(lhs, rhs);
                        // 尾处理：+ bias(f32) ⇒ bf16 ⇒ silu ⇒ 写回
                        vo.from_m(acc);
                        vo.add(vo, bcast(vb, &[0, 1]));
                        yh.cvt(vo, Cvt::default());
                        $crate::conv2d_s2_fast!(@act $act, yh, tmp);
                        yh.store(gy.sub([oy, mb, 0, nb * NC], [1, 1, 16, NC]));
                    }
                }
            }
        }
    };
    (@act silu, $yh:ident, $tmp:ident) => { $crate::logistic!(silu, $yh, $yh, $tmp) };
    (@act none, $yh:ident, $tmp:ident) => {};
}
