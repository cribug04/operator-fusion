//! conv2d_s2_fast 探针：YOLO26n 的 `conv3s2_80x80_64to64` 同形状（80×80×64 → 40×40×64，k=3、s=2、silu）。
#![no_std]

pub mod kernels;

#[cfg(feature = "step")]
mod step;
