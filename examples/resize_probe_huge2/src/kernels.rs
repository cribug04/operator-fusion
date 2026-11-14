//! resize_bilinear2_huge2 的尺寸特化：128×128×256 bf16 ⇒ 256×256×256 bf16（NHWC、align_corners=False、通道分块 cc=32）。双核：两核各 4 块。
calblas::resize_bilinear2_tiled2!(pub resize_bilinear2_huge2: h = 128, w = 128, c = 256, cc = 32);
