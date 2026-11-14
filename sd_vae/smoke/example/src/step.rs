//! SD1.5 VAE 解码器关键形状的**整个程序**（单芯粒、双核）：
//! `cargo build --lib --features step --target dopamine-unknown-none-cvx` 出一份 `.cvx`。
//!
//! 顺序（每个 `cores = 2` 的 kernel 后面跟一条 `sync_cores()`；`cores = 1` 的包在 `if core_rank() == 0` 里）：
//!
//! ```text
//! conv_in 4(补到 32)→512 k3s1 @64² ⇒ NHWC→NCHW ⇒ GroupNorm(32组) ⇒ NCHW→NHWC ⇒ conv 512→512 k3s1 @64²
//!   ⇒ qkv 1×1 512→1536 ⇒ upsample2 ×2 (64²→128², 512ch) ⇒ conv 512→256 k3s1 @128² ⇒ @out_y
//! 任务 C：transpose2d(k ⇒ kT) ⇒ 单头 flash 注意力 T=4096 / D=512（q/k/v/o 紧排 [T,512]）
//! 搬运：border NHWC 内部 ⇒ 紧排 [H·W,P]（P=1536/512）、紧排 ⇒ border 内部（P=1536/512）
//! ```
use crate::kernels::*;
use npu::prelude::*;

// ─── @w 里的偏移（字节；每块 4096 对齐）。卷积权重 bf16 [k·k][Cin][Cout]，bias f32 [Cout]，
//     GroupNorm 的 w/b 是主机展开的 f32 [C·HW/128][8] ───
const W_CONV1: u64 = 0; // [9][32][512] bf16 = 294,912
const B_CONV1: u64 = 294_912; // [512] f32 = 2,048
const W_GN: u64 = 299_008; // [16384][8] f32 = 524,288
const B_GN: u64 = 823_296; // 同上 = 524,288
const W_CONV2: u64 = 1_347_584; // [9][512][512] bf16 = 4,718,592
const B_CONV2: u64 = 6_066_176; // [512] f32 = 2,048
const W_QKV: u64 = 6_070_272; // [1][512][1536] bf16 = 1,572,864
const B_QKV: u64 = 7_643_136; // [1536] f32 = 6,144
const W_CONV3: u64 = 7_651_328; // [9][512][256] bf16 = 2,359,296
const B_CONV3: u64 = 10_010_624; // [256] f32 = 1,024
const W_BYTES: u64 = 10_014_720;

// ─── @act 里的偏移（字节；每个缓冲都按 4096 对齐）───
const A_H0: u64 = 0; // conv_in 输出 NHWC [68][68][512] = 4,734,976
const A_NCHW1: u64 = 4_734_976; // NHWC→NCHW 输出 [512][4096] = 4,194,304
const A_NCHW2: u64 = 8_929_280; // GroupNorm 输出（NCHW）= 4,194,304
const A_H1: u64 = 13_123_584; // NCHW→NHWC 输出 NHWC [68][68][512] = 4,734,976
const A_H2: u64 = 17_858_560; // conv_mid 输出 = 4,734,976
const A_QKV: u64 = 22_593_536; // qkv 1×1 输出 NHWC [68][68][1536] = 14,204,928
const A_UP: u64 = 36_798_464; // upsample2 输出 NHWC [132][132][512] = 17,842,176
const A_Q: u64 = 54_640_640; // q [4096, 512] 紧排 = 4,194,304
const A_K: u64 = 58_834_944; // k = 4,194,304
const A_KT: u64 = 63_029_248; // kT [512, 4096] = 4,194,304
const A_V: u64 = 67_223_552; // v = 4,194,304
const A_O: u64 = 71_417_856; // o = 4,194,304
const A_T1536: u64 = 75_612_160; // 紧排 [4096, 1536] = 12,582,912
const A_T512: u64 = 88_195_072; // 紧排 [4096, 512] = 4,194,304
const A_B1536: u64 = 92_389_376; // 边框 NHWC [68][68][1536] = 14,204,928
const A_B512: u64 = 106_594_304; // 边框 NHWC [68][68][512] = 4,734,976
const ACT_BYTES: u64 = 111_329_280;

npu::decls! {
    in_img: input, bytes = 295936;      // [68][68][32] bf16（4 个真实通道 + 28 个 0）
    out_y: output, bytes = 8921088;     // conv 512→256 输出 NHWC [132][132][256]
    w: weights, bytes = 10014720;       // 卷积权重 + bias + GroupNorm 的 w/b
    act: workspace, bytes = 111329280;
}

const _: () = assert!(W_BYTES <= 10_014_720, "@w 不够：改 decls! 的 bytes");
const _: () = assert!(ACT_BYTES <= 111_329_280, "@act 不够：改 decls! 的 bytes");

#[npu::program(chiplets = 1, cores = 2)]
pub fn step() {
    // 1) conv_in：latent [64][64][4]（补到 32 通道）⇒ 512，k3 s1 silu
    vae_conv_in_64x64_4to512_k3s1_p32_512_silu(
        in_img().cast(),
        w().at(W_CONV1).cast(),
        w().at(B_CONV1).cast(),
        act().at(A_H0).cast(),
    );
    sync_cores();

    // 2) NHWC → NCHW → GroupNorm(512 通道 / 32 组 / HW=4096) → NHWC
    if core_rank() == 0 {
        vae_nhwc2nchw_64x64_512_p512(act().at(A_H0).cast(), act().at(A_NCHW1).cast());
    }
    sync_cores();
    vae_gn_512ch_32groups_hw4096_silu(
        act().at(A_NCHW1).cast(),
        w().at(W_GN).cast(),
        w().at(B_GN).cast(),
        act().at(A_NCHW2).cast(),
    );
    sync_cores();
    if core_rank() == 0 {
        // NCHW→NHWC 只写内部 ⇒ 先把整块边框缓冲清 0（4,734,976 = 289 × 16 KiB）
        vae_zero_bf16(act().at(A_H1).cast(), 289usize);
    }
    sync_cores();
    if core_rank() == 0 {
        vae_nchw2nhwc_64x64_512_p512(act().at(A_NCHW2).cast(), act().at(A_H1).cast());
    }
    sync_cores();

    // 3) mid resnet 的 3×3 卷积：512 → 512 @64²
    vae_conv_mid_64x64_512to512_k3s1_silu(
        act().at(A_H1).cast(),
        w().at(W_CONV2).cast(),
        w().at(B_CONV2).cast(),
        act().at(A_H2).cast(),
    );
    sync_cores();

    // 4) 注意力 qkv 投影的融合形状：1×1 卷积 512 → 1536 @64²
    vae_qkv_64x64_512to1536_k1s1_none(
        act().at(A_H2).cast(),
        w().at(W_QKV).cast(),
        w().at(B_QKV).cast(),
        act().at(A_QKV).cast(),
    );
    sync_cores();

    // 5) 最近邻上采样 ×2：64×64×512 ⇒ 128×128×512
    if core_rank() == 0 {
        vae_zero_bf16(act().at(A_UP).cast(), 1089usize); // 17,842,176 = 1089 × 16 KiB
    }
    sync_cores();
    if core_rank() == 0 {
        vae_up2_64x64_512_p512_512(act().at(A_H2).cast(), act().at(A_UP).cast());
    }
    sync_cores();

    // 6) 上采样后的 3×3 卷积：512 → 256 @128² ⇒ @out_y
    vae_conv_up_128x128_512to256_k3s1_silu(
        act().at(A_UP).cast(),
        w().at(W_CONV3).cast(),
        w().at(B_CONV3).cast(),
        out_y().cast(),
    );
    sync_cores();

    // 7) 【任务 C】单头 flash 注意力：k ⇒ kT（transpose2d），再 q·kT / v（T = 4096、D = 512、无掩码）
    if core_rank() == 0 {
        vae_transpose_kt_4096x512(act().at(A_K).cast(), act().at(A_KT).cast());
    }
    sync_cores();
    vae_attn_4096x512(
        act().at(A_Q).cast(),
        act().at(A_KT).cast(),
        act().at(A_V).cast(),
        act().at(A_O).cast(),
    );
    sync_cores();

    // 8) 搬运：边框 NHWC 内部 ⇒ 紧排 [H·W, P]（P = 1536 / 512）
    if core_rank() == 0 {
        vae_border2tight_64x64_p1536(act().at(A_QKV).cast(), act().at(A_T1536).cast());
        vae_border2tight_64x64_p512(act().at(A_H0).cast(), act().at(A_T512).cast());
    }
    sync_cores();

    // 9) 搬运：紧排 ⇒ 边框 NHWC 内部（边框先清 0）
    if core_rank() == 0 {
        vae_zero_bf16(act().at(A_B1536).cast(), 867usize); // 14,204,928 = 867 × 16 KiB
        vae_zero_bf16(act().at(A_B512).cast(), 289usize); // 4,734,976 = 289 × 16 KiB
    }
    sync_cores();
    if core_rank() == 0 {
        vae_tight2border_64x64_p1536(act().at(A_T1536).cast(), act().at(A_B1536).cast());
        vae_tight2border_64x64_p512(act().at(A_T512).cast(), act().at(A_B512).cast());
    }
    sync_cores();
}
