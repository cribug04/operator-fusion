//! resize 探针 example：一个 CalVX 程序只跑 `resize_bilinear2`，输入/输出由主机按 decls 写入/读回。
#![no_std]

pub mod kernels;

/// 引擎表（主机侧：图 ⇒ 计划）。设备 target 上不编（kernel 的主机桩只在主机上有）。
#[cfg(not(all(target_arch = "riscv32", target_os = "none")))]
pub mod engines;

#[cfg(feature = "step")]
mod step;
