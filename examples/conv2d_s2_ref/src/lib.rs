//! 慢路对照：同形状的 `conv2d!`（k=3、s=2、silu），用于和 `conv2d_s2_fast!` 在**同一块板、同一份数据**上逐位对比。
#![no_std]

pub mod kernels;

#[cfg(feature = "step")]
mod step;
