//! 通道分块版 resize_bilinear2：32×32×256 ⇒ 64×64×256（CC=64，共 4 块）。
calblas::resize_bilinear2_tiled!(pub resize_bilinear2_big: h = 32, w = 32, c = 256, cc = 64);
