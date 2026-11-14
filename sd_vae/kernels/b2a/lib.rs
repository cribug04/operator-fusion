//! SD1.5 VAE 解码器 stage b2a（gen.py --stage b2a 生成）。
#![no_std]

pub mod kernels;

#[cfg(feature = "step")]
mod step;
