//! SD1.5 VAE 解码器关键形状的「平台侧」冒烟 example：
//! 只验证 calBLAS 视觉模板能否按这些形状实例化、能编出设备遍、能过 `docc calbin`
//! （**没有真机、没有 `cvx` 模拟器 ⇒ 数值/性能未验证**）。
#![no_std]

pub mod kernels;

#[cfg(feature = "step")]
mod step;
