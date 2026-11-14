//! resize 通道分块·双核探针：与 resize_probe_big 同形状（32×32×256），kernel 标 cores = 2，两核各算一半通道块；用于与单核版在真机上对比。
#![no_std]

pub mod kernels;

/// 引擎表（主机侧：图 ⇒ 计划）。设备 target 上不编（kernel 的主机桩只在主机上有）。
#[cfg(not(all(target_arch = "riscv32", target_os = "none")))]
pub mod engines;

#[cfg(feature = "step")]
mod step;
