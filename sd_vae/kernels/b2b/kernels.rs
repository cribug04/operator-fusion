//! SD1.5 VAE decoder kernel 特化（gen.py 生成，勿手改）

calblas::upsample2!(pub k_up_128_256_p512_512: h = 128, w = 128, c = 256, pin = 512, pout = 512);
calblas::conv2d!(pub k_conv_256_512to512_k3_p512_512: h = 256, w = 256, cin = 512, cout = 512, k = 3, s = 1, pin = (512, 0), pout = (512, 0), act = none);

// 图像 slot 边框清零：n 是运行期标量（常量循环体会被展开成 1 MiB 代码，链接器分支够不着）
calblas::fill16k!(pub k_fill0: value = 0.0f32);

// ---------------------------------------------------------------- 手写搬运 kernel
//
// border NHWC 图（[H+4][H+4][PIN]，内部 [H][H] 像素）与紧排 [H·H][POUT] 互搬。
// 一次搬「一行图像 = H 个像素 × 256 个通道」：视图 [像素, 通道]（行距 PIN/POUT），
// 每个像素的 256 个通道连续（512 B = 一行 VB）。


// ---------------------------------------------------------------- 手写：单头 flash 注意力（VAE mid attention）
//
// q / k / v / o 都是紧排 [T, D] bf16；kT 是 [D, T]（transpose2d 转好）。
// 两遍：第一遍逐 key 块算分数取行最大；第二遍重算分数、exp、行和，V 也分块累加。
// 单头、无 mask、scale = 1/√D；query 块两核各一半。
