//! resize 通道分块探针：32×32×256 bf16 ⇒ 64×64×256 bf16（C=256，原版 C≤128 放不下，用分块版 cc=64）。
#![no_std]

pub mod kernels;

/// 引擎表（主机侧：图 ⇒ 计划）。设备 target 上不编（kernel 的主机桩只在主机上有）。
#[cfg(not(all(target_arch = "riscv32", target_os = "none")))]
pub mod engines;

#[cfg(feature = "step")]
mod step;
