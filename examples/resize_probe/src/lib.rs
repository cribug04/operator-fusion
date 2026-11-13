//! resize 探针 example：一个 CalVX 程序只跑 `resize_bilinear2`，输入/输出由主机按 decls 写入/读回。
#![no_std]

pub mod kernels;

#[cfg(feature = "step")]
mod step;
