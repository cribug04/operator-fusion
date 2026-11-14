//! **docc 0.4.1 riscvc 21 位分支 bug 的最小复现**：只调用一次 128² 的 `group_norm!`。
//! - 缺省：`group_norm!` 的常量 NH 被 docc 全展开 ⇒ 单函数 .text ≈ 1.12 MB > 2^20 ⇒ `from-calvx` 失败；
//! - `--features step,sram`：NH 改成从 SRAM 读的运行期值 ⇒ 循环不展开，`.text` ≈ 27 KB，通过。
#![no_std]

pub mod kernels;

#[cfg(feature = "step")]
mod step;
