//! resize_bilinear2_huge 的尺寸特化：128×128×256 bf16 ⇒ 256×256×256 bf16（NHWC、align_corners=False、通道分块 cc=32）。单核计时基线（8 块）。
calblas::resize_bilinear2_tiled!(pub resize_bilinear2_huge: h = 128, w = 128, c = 256, cc = 32);
