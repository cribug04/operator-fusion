//! SD1.5 VAE 解码器 stage a（gen.py --stage a 生成）。
#![no_std]

pub mod kernels;

#[cfg(feature = "step")]
mod step;
