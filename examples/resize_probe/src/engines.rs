//! 引擎表：Resize 这一种运算、32×32×16 这一个特化。`Plan::new` 按图里 `Op::Resize` 的 `ResizeDesc` 在表里找同形的 kernel。
use crate::kernels as k;
use calblas::engines::Resize;
use calblas::entry;
use caldnn::Engine;

pub const ENGINES: &[&dyn Engine] = &[&Resize(&[entry!(k::resize_bilinear2)])];
