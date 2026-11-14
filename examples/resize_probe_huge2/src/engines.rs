//! 引擎表：Resize 这一种运算、128×128×256（cc = 32）这一个特化。
use crate::kernels as k;
use calblas::engines::Resize;
use calblas::entry;
use caldnn::Engine;

pub const ENGINES: &[&dyn Engine] = &[&Resize(&[entry!(k::resize_bilinear2_huge2)])];
