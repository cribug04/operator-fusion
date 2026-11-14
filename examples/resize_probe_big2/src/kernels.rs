//! resize_bilinear2_big2 的尺寸特化：32×32×256 bf16 ⇒ 64×64×256 bf16（NHWC、align_corners=False、通道分块 cc=64）。双核：两核各 2 块。
calblas::resize_bilinear2_tiled2!(pub resize_bilinear2_big2: h = 32, w = 32, c = 256, cc = 64);
