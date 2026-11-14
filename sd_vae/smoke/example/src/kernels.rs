//! SD1.5 VAE 解码器关键形状的 calBLAS 视觉模板实例（每个宏 = 一个 `#[npu::kernel]`）。
//!
//! 排布口径（见 `calBLAS/src/kernels/vision.rs` 顶部）：
//! - 卷积 / 池化 / 上采样是 **NHWC bf16 + 四周 2 像素边框**：图 = `[H + 4][W + 4][P]`，
//!   `P` = 像素行距（≥ 通道数；边框是 0）。`pin/pout = (行距, 窗口内通道偏移)`，指针参数本身
//!   要给到 32 通道（64 B）对齐处，窗口偏移（0 / 16 / …）是编译期常数。
//! - NCHW 一侧（`nchw_to_nhwc!` / `nhwc_to_nchw!` / `group_norm!`）是紧排 `[C][H·W]`，无边框。
//! - 权重 bf16 `[k·k][Cin][Cout]`（BN 已折入），bias f32 `[Cout]`；`weights = ci_major` 时换成
//!   `[Cin][k·k][Cout]`（本 example 不用）。

// ── 1) conv_in：latent [64][64][4] ⇒ 512 通道，k = 3、s = 1、silu ────────────────────────────
// 模板断言 `CI % 16 == 0`：4 个真实通道补到 **32**（不是 16 —— 见下面「为什么不是 16」）。
// 输入图 [68][68][32]，输出图 [68][68][512]。
// 为什么不是 16：3×3 卷积的抽头 (0,0) 在输入里偏 `((B-1)·WI + (B-1)) · P = (1·68+1)·P` 个元素，
// P = 16 时字节偏移 2208 % 64 = 32 ⇒ DDR⇒L1B 的整段搬源不是 64 B 对齐（又跨 4 KB）⇒ docc 拒收；
// P = 32 时偏移 4416 = 69·64 ⇒ 对齐。⇒ 第一层要做成 32→512 的「4 补到 32」。
calblas::conv2d!(pub vae_conv_in_64x64_4to512_k3s1_p32_512_silu: h = 64, w = 64, cin = 32, cout = 512, k = 3, s = 1, pin = (32, 0), pout = (512, 0), act = silu);

// ── 2) mid block：GroupNorm(32 组、512 通道、HW = 64·64 = 4096) ─────────────────────────────
// 先在 NHWC 与 NCHW 之间换排布（模板只给通道数/图大小是 32 倍数的情形）。
calblas::nhwc_to_nchw!(pub vae_nhwc2nchw_64x64_512_p512: h = 64, w = 64, c = 512, p = 512);
// 512 通道 / 32 组 ⇒ 每组 16 通道 × 4096 像素；act = silu（ResnetBlock 的 non_linearity）。
calblas::group_norm!(pub vae_gn_512ch_32groups_hw4096_silu: hw = 4096, c = 512, groups = 32, eps = 1.0e-6f32, act = silu);
calblas::nchw_to_nhwc!(pub vae_nchw2nhwc_64x64_512_p512: h = 64, w = 64, c = 512, p = 512);

// ── 3) mid resnet 的 3×3 卷积：512 → 512 @ 64×64，s = 1、silu ───────────────────────────────
calblas::conv2d!(pub vae_conv_mid_64x64_512to512_k3s1_silu: h = 64, w = 64, cin = 512, cout = 512, k = 3, s = 1, pin = (512, 0), pout = (512, 0), act = silu);

// ── 4) 注意力 qkv 投影的融合形状：1×1 卷积 512 → 3·512 = 1536 @ 64×64（act = none）─────────
// 1×1 是 conv2d! 支持的另一条分支（只读内部、K = 1）。
calblas::conv2d!(pub vae_qkv_64x64_512to1536_k1s1_none: h = 64, w = 64, cin = 512, cout = 1536, k = 1, s = 1, pin = (512, 0), pout = (1536, 0), act = none);

// ── 5) 最近邻上采样 ×2：64×64×512 ⇒ 128×128×512（up_blocks.0 的 upsampler）────────────────
// 文档写 c ≤ 256；这里先按 VAE 的 512 通道实例化，由 docc 判（见 SMOKE.md 的结论）。
calblas::upsample2!(pub vae_up2_64x64_512_p512_512: h = 64, w = 64, c = 512, pin = 512, pout = 512);

// ── 6) 上采样后的 3×3 卷积：512 → 256 @ 128×128，s = 1、silu（up_blocks.2 的第一个 resnet）──
calblas::conv2d!(pub vae_conv_up_128x128_512to256_k3s1_silu: h = 128, w = 128, cin = 512, cout = 256, k = 3, s = 1, pin = (512, 0), pout = (256, 0), act = silu);

// ── 7) 边框清零（每块 32 行 × 512 B = 16 KiB 的 bf16 全填一个值）─────────────────────────────
// 模板的卷积读「四周 2 像素边框」，边框由调用方清零；上采样 / 换排布只写内部，所以要自己补。
calblas::fill16k!(pub vae_zero_bf16: value = 0.0f32);

// ═══════════════════════════════════════════════════════════════════════════════════════
// 任务 C：SD VAE mid attention 的**单头** flash 注意力（T = 4096、D = 512）
// ═══════════════════════════════════════════════════════════════════════════════════════
//
// 为什么不用 `calblas::attention!`：它的 rhs / acc 是 `mcache::<bf16>([D, KB])` / `mcache::<f32>([16, D])`，
// D = 512 时一块就是 128 KB（KB = 128）⇒ 超过 mcache 单块 32 KB 的上限（既有的实证是头维 ≤ 128）。
// 这里把**头维也分块**（DK = 128 一块，共 4 块），其余结构与 `attention!` 的两遍 flash 完全一致：
//   第一遍：s = bf16(q · Kᵀ) · 1/√d、行最大累加成 m；
//   第二遍：p = exp(s − m)（bf16 硬件近似）、l += Σp、acc[d 块] += p · V[:, d 块]（TU 累加）；
//   输出 o = acc / l（bf16）。
// 两核按 query 块分：T / 16 = 256 块，每核 128 块（`halves!`）。
/// q / v / o 紧排 `[T, D]` bf16（行距 D）；`kt` 是 `transpose2d!` 转好的 `[D, T]`（行距 T）。


#[npu::kernel(cores = 2, numerics = "ks1")]
pub fn vae_attn_4096x512(
    q: npu::Ptr<npu::Ddr, npu::bf16>,
    kt: npu::Ptr<npu::Ddr, npu::bf16>,
    v: npu::Ptr<npu::Ddr, npu::bf16>,
    o: npu::Ptr<npu::Ddr, npu::bf16>,
) {
    use npu::prelude::*;
    const T: usize = 4096;
    const D: usize = 512;
    const DK: usize = 128; // 头维分块：一块 128 维
    const KB: usize = 128; // key 分块
    const NDK: usize = D / DK; // 4
    const NKB: usize = T / KB; // 32
    const QB: usize = 16; // 一块 16 行 query（TU 累加器 16 行）
    const NEG: f32 = -3.0e38;
    const SCALE: f32 = 0.044194173824159216; // 1/√512
    const _: () = assert!(T % KB == 0 && D % DK == 0 && T % QB == 0);

    let gq = q.bind_strided::<npu::Read, 2>([T, D], [D, 1]);
    let gkt = kt.bind_strided::<npu::Read, 2>([D, T], [T, 1]);
    let gv = v.bind_strided::<npu::Read, 2>([T, D], [D, 1]);
    let go = o.bind_strided::<npu::Write, 2>([T, D], [D, 1]);

    let (q0, qn) = calblas::halves!(T / QB);

    // 矩阵 tile：每块 ≤ 32 KB
    let lq = mcache::<bf16, _>([QB, DK]); // q 的 [16, 128] 切片          4 KB
    let rk = mcache::<bf16, _>([DK, KB]); // Kᵀ 的 [128, 128] 块          32 KB
    let sacc = mcache::<f32, _>([QB, KB]); // 分数累加                   8 KB
    let lp = mcache::<bf16, _>([QB, KB]); // p 的 [16, 128]               4 KB
    let rv = mcache::<bf16, _>([KB, DK]); // V 的 [128, 128] 块          32 KB
    // D 的 4 个输出累加器：同一个循环里次序固定（同 conv2d! 的 FAST 路 4 个累加器）
    let (a0, a1) = (mcache::<f32, _>([QB, DK]), mcache::<f32, _>([QB, DK]));
    let (a2, a3) = (mcache::<f32, _>([QB, DK]), mcache::<f32, _>([QB, DK]));
    // 向量 tile：一行都 ≤ 512 B
    let (sf, sb, tf) = (vcache::<f32, _>([QB, KB]), vcache::<bf16, _>([QB, KB]), vcache::<f32, _>([QB, KB]));
    let (m, top) = (vcache::<f32, _>([QB, 1]), vcache::<f32, _>([QB, 1]));
    let (l, rs) = (vcache::<f32, _>([QB, 1]), vcache::<f32, _>([QB, 1]));
    let (mb, lb) = (vcache::<bf16, _>([QB, 1]), vcache::<bf16, _>([QB, 1]));
    let (out, outb) = (vcache::<f32, _>([QB, DK]), vcache::<bf16, _>([QB, DK]));
    // 第二遍的 p 全留在 VB 里：[NKB · 16 行][KB] bf16 = 128 KB（VB 384 KB 之内）
    let pall = vcache::<bf16, _>([NKB * QB, KB]);

    for i in 0..qn {
        let qb = q0 + i;
        // ── 第一遍：行最大值 m（这一层循环里只有「分数」一条累加串）──
        m.fill(NEG);
        for kb in 0..NKB {
            sacc.fill(0.0f32);
            for dk in 0..NDK {
                lq.load(gq.sub([qb * QB, dk * DK], [QB, DK]));
                rk.load(gkt.sub([dk * DK, kb * KB], [DK, KB]));
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
        // ── 第二遍 a：p = exp(s − m) 存进 VB，同时累计行和 l（仍只有「分数」一条累加串）──
        l.fill(0.0f32);
        for kb in 0..NKB {
            sacc.fill(0.0f32);
            for dk in 0..NDK {
                lq.load(gq.sub([qb * QB, dk * DK], [QB, DK]));
                rk.load(gkt.sub([dk * DK, kb * KB], [DK, KB]));
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
            pall.rows(kb * QB, QB).copy(sb);
        }
        lb.cvt(l, Cvt::default());
        // ── 第二遍 b：out[:, 4 个 D 块] = Σ_kb p · V（这一层循环里只有 4 个输出累加器，
        //    次序固定；docc calt-fate 不接受「分数链 + 输出累加器链」在同一个循环里同时在飞）──
        a0.fill(0.0f32);
        a1.fill(0.0f32);
        a2.fill(0.0f32);
        a3.fill(0.0f32);
        for kb in 0..NKB {
            lp.load_v(pall.rows(kb * QB, QB));
            rv.load(gv.sub([kb * KB, 0], [KB, DK]));
            a0.mma(lp, rv);
            rv.load(gv.sub([kb * KB, DK], [KB, DK]));
            a1.mma(lp, rv);
            rv.load(gv.sub([kb * KB, 2 * DK], [KB, DK]));
            a2.mma(lp, rv);
            rv.load(gv.sub([kb * KB, 3 * DK], [KB, DK]));
            a3.mma(lp, rv);
        }
        // ── o = acc / l（bf16），按 [16, D] 连续行写回 ──
        out.from_m(a0);
        outb.cvt(out, Cvt::default());
        outb.div(outb, bcast(lb, &[0, 1]));
        outb.store(go.sub([qb * QB, 0], [QB, DK]));
        out.from_m(a1);
        outb.cvt(out, Cvt::default());
        outb.div(outb, bcast(lb, &[0, 1]));
        outb.store(go.sub([qb * QB, DK], [QB, DK]));
        out.from_m(a2);
        outb.cvt(out, Cvt::default());
        outb.div(outb, bcast(lb, &[0, 1]));
        outb.store(go.sub([qb * QB, 2 * DK], [QB, DK]));
        out.from_m(a3);
        outb.cvt(out, Cvt::default());
        outb.div(outb, bcast(lb, &[0, 1]));
        outb.store(go.sub([qb * QB, 3 * DK], [QB, DK]));
    }
}

// ── k [T, D] 紧排 ⇒ kT [D, T]（`transpose2d!`：32×32 块、行块按核分；这里 cores = 1 的 kernel 由 0 号核调用）──
calblas::transpose2d!(pub vae_transpose_kt_4096x512: rows = 4096, cols = 512);

// ═══════════════════════════════════════════════════════════════════════════════════════
// 搬运（单核）：边框 NHWC [H+4][W+4][P] 的内部 ⇄ 紧排 [H·W, P]
// 每行 W·P 个元素在源/目的里都是连续的，按 512 B 一块搬（vcache 一行上限）。
// ═══════════════════════════════════════════════════════════════════════════════════════

/// 边框 NHWC 内部 ⇒ 紧排 `[H·W, P]`。
macro_rules! border2tight {
    ($vis:vis $name:ident: h = $h:expr, w = $w:expr, p = $p:expr) => {
        #[npu::kernel]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
            use npu::prelude::*;
            const H: usize = $h;
            const W: usize = $w;
            const P: usize = $p;
            const B: usize = 2;
            const WI: usize = W + 2 * B;
            const CC: usize = 256; // 256 个 bf16 = 512 B 一块
            const NC: usize = W * P / CC;
            const _: () = assert!(W * P % CC == 0, "border2tight：W·P 要是 CC 的倍数");
            // 源：内部左上角起，每行 W·P 个元素（行距 WI·P）
            let gx = x.add(((B * WI + B) * P) * 2).bind_strided::<npu::Read, 2>([H, W * P], [WI * P, 1]);
            let gy = y.bind_strided::<npu::Write, 2>([H, W * P], [W * P, 1]);
            let t = vcache::<bf16, _>([1, CC]);
            for iy in 0..H {
                for c in 0..NC {
                    t.load(gx.sub([iy, c * CC], [1, CC]));
                    t.store(gy.sub([iy, c * CC], [1, CC]));
                }
            }
        }
    };
}

/// 紧排 `[H·W, P]` ⇒ 边框 NHWC 的内部（边框要调用方先清零）。
macro_rules! tight2border {
    ($vis:vis $name:ident: h = $h:expr, w = $w:expr, p = $p:expr) => {
        #[npu::kernel]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
            use npu::prelude::*;
            const H: usize = $h;
            const W: usize = $w;
            const P: usize = $p;
            const B: usize = 2;
            const WI: usize = W + 2 * B;
            const CC: usize = 256;
            const NC: usize = W * P / CC;
            const _: () = assert!(W * P % CC == 0, "tight2border：W·P 要是 CC 的倍数");
            let gx = x.bind_strided::<npu::Read, 2>([H, W * P], [W * P, 1]);
            let gy = y.add(((B * WI + B) * P) * 2).bind_strided::<npu::Write, 2>([H, W * P], [WI * P, 1]);
            let t = vcache::<bf16, _>([1, CC]);
            for iy in 0..H {
                for c in 0..NC {
                    t.load(gx.sub([iy, c * CC], [1, CC]));
                    t.store(gy.sub([iy, c * CC], [1, CC]));
                }
            }
        }
    };
}

border2tight!(pub vae_border2tight_64x64_p1536: h = 64, w = 64, p = 1536);
border2tight!(pub vae_border2tight_64x64_p512: h = 64, w = 64, p = 512);
tight2border!(pub vae_tight2border_64x64_p1536: h = 64, w = 64, p = 1536);
tight2border!(pub vae_tight2border_64x64_p512: h = 64, w = 64, p = 512);
