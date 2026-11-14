/// 双线性上采样 ×2（NHWC、无边框，bf16）—— **通道分块 · 双核版**：输入 `[H][W][C]` ⇒ 输出 `[2H][2W][C]`。
///
/// 语义、数值、每块的计算次序与 `resize_bilinear2_tiled!` **完全相同**（内部直接复用它的 `@pair` / `@h` 分支），
/// 区别只在 `NC = C/CC` 个通道块由两个核各算一半：核 r 处理块 `r*NC/2 .. (r+1)*NC/2`。
/// 各块读写的 DDR 区域互不重叠（输入只读；输出按通道块偏移 `cc*CC*2 B` 分开），核间不需要同步。
///
/// 约束：在 `resize_bilinear2_tiled!` 的基础上多一条 `NC % 2 == 0`（每核整数个块）。
/// 调用：kernel 标 `cores = 2`，必须放进 `#[npu::program(cores = 2)]` 的程序里、由**两个核都调用**
/// （不能包在 `if core_rank() == 0` 里，否则只算前一半通道；见 docs/REVIEW_SD_VAE_2026-11-14.md 的探针实验）。
#[macro_export]
macro_rules! resize_bilinear2_tiled2 {
    ($vis:vis $name:ident: h = $h:expr, w = $w:expr, c = $c:expr, cc = $cc:expr) => {
        $crate::desc!($vis $name: ResizeDesc { h: $h as u32, w: $w as u32, c: $c as u32, scale: 2, mode: $crate::caldnn::Interp::Bilinear });
        #[npu::kernel(cores = 2, numerics = "ks1")]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
            use npu::prelude::*;
            const H: usize = $h;
            const W: usize = $w;
            const C: usize = $c;
            const CC: usize = $cc;
            const NC: usize = C / CC;
            const PER: usize = NC / 2;
            const _: () = assert!(H >= 3 && W >= 3, "resize_bilinear2_tiled2：H、W 至少 3");
            const _: () = assert!(CC >= 32 && CC % 32 == 0 && C % CC == 0, "resize_bilinear2_tiled2：CC 是 >=32 的 32 倍数且整除 C");
            const _: () = assert!(NC % 2 == 0, "resize_bilinear2_tiled2：通道块数 C/CC 必须是偶数（两核各一半）");
            const _: () = assert!(W * CC * 56 <= 384 * 1024, "resize_bilinear2_tiled2：W*CC 太大，中间 cache 放不进 VB");
            let base = core_rank() * PER;
            for k in 0..PER {
                let xoff = (base + k) * CC;
                $crate::resize_bilinear2_tiled!(@pair x, y, $h, $w, $c, $cc, xoff, 0, 0, 1);
                for iy in 1..H - 1 {
                    $crate::resize_bilinear2_tiled!(@pair x, y, $h, $w, $c, $cc, xoff, iy - 1, iy, iy + 1);
                }
                $crate::resize_bilinear2_tiled!(@pair x, y, $h, $w, $c, $cc, xoff, $h - 2, $h - 1, $h - 1);
            }
        }
    };
}
