//! SD1.5 VAE 解码器 stage b1（gen.py --stage b1 生成）。
#![no_std]

pub mod kernels;

#[cfg(feature = "step")]
mod step;
