//! resize 通道分块·大尺寸双核：128×128×256 ⇒ 256×256×256，kernel 标 cores = 2，与 resize_probe_huge 同形状、同数值，用于真机计时对比。
#![no_std]

pub mod kernels;

/// 引擎表（主机侧：图 ⇒ 计划）。设备 target 上不编（kernel 的主机桩只在主机上有）。
#[cfg(not(all(target_arch = "riscv32", target_os = "none")))]
pub mod engines;

#[cfg(feature = "step")]
mod step;
