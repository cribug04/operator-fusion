//! resize_bilinear2 的尺寸特化：32×32×16 bf16 ⇒ 64×64×16 bf16（NHWC、align_corners=False）。
calblas::resize_bilinear2!(pub resize_bilinear2: h = 32, w = 32, c = 16);
