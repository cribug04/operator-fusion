//! SD1.5 VAE 解码器 stage b2b（gen.py --stage b2b 生成）。
#![no_std]

pub mod kernels;

#[cfg(feature = "step")]
mod step;
