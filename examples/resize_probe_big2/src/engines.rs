//! 引擎表：Resize 这一种运算、32×32×256（cc = 64）这一个特化。
use crate::kernels as k;
use calblas::engines::Resize;
use calblas::entry;
use caldnn::Engine;

pub const ENGINES: &[&dyn Engine] = &[&Resize(&[entry!(k::resize_bilinear2_big2)])];
