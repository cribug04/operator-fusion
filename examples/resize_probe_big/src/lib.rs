//! resize 通道分块探针：32×32×256 bf16 ⇒ 64×64×256 bf16（C=256，原版 C≤128 放不下，用分块版 cc=64）。
#![no_std]

pub mod kernels;

#[cfg(feature = "step")]
mod step;
