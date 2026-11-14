//! 实证探针：`calblas::attention!` 的头维上限（D = 128 能用 / D = 512 被拒）。
#![no_std]

pub mod kernels;

#[cfg(feature = "step")]
mod step;
