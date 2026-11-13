#!/usr/bin/env python3
"""resize_bilinear2 的 CPU 参考（numpy）。

严格复刻 kernel 的运算次序（f32 每一步 mul/add 各舍一次，最后一次性 RNE 舍回 bf16）：
  行内：he[x] = 0.75*f[x] + 0.25*f[x-1]（x=0 用 f[0]）
        ho[x] = 0.75*f[x] + 0.25*f[x+1]（x=W-1 用 f[W-1]）
  垂直：out[2y]   = 0.25*he(y-1) + 0.75*he(y)   （y=0 时 he(-1)=he(0)）
        out[2y+1] = 0.75*he(y)   + 0.25*he(y+1) （y=H-1 时 he(H)=he(H-1)）
语义等价于 torch.nn.functional.interpolate(scale_factor=2, mode="bilinear", align_corners=False)。
"""
import numpy as np

F32 = np.float32
H, W, C = 32, 32, 16


def to_bf16_bits(a: np.ndarray) -> np.ndarray:
    """f32 -> bf16（RNE，返回 uint16 位型）。"""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).copy()
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    r = (u + np.uint32(0x7FFF) + lsb) >> np.uint32(16)
    return r.astype(np.uint16)


def bf16_bits_to_f32(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


def hori(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """f: [H, W, C] f32 -> (he, ho)，he/ho 对应输出偶数列/奇数列（长度 W）。"""
    he = f * F32(0.75)
    t = np.zeros_like(f)
    t[:, 0:W - 1, :] = f[:, 0:W - 1, :] * F32(0.25)   # t[x] = f[x]*0.25
    he[:, 1:W, :] = he[:, 1:W, :] + t[:, 0:W - 1, :]   # he[x] += f[x-1]*0.25
    he[:, 0, :] = f[:, 0, :]                            # x=0 clamp
    ho = f * F32(0.75)
    t2 = np.zeros_like(f)
    t2[:, 0:W - 1, :] = f[:, 1:W, :] * F32(0.25)       # t2[x] = f[x+1]*0.25
    ho[:, 0:W - 1, :] = ho[:, 0:W - 1, :] + t2[:, 0:W - 1, :]
    ho[:, W - 1, :] = f[:, W - 1, :]                    # x=W-1 clamp
    return he, ho


def resize_kernel_order(x_f32: np.ndarray) -> np.ndarray:
    """严格按 kernel 的运算次序，返回 [2H, 2W, C] f32（未舍 bf16）。"""
    f = np.ascontiguousarray(x_f32, dtype=np.float32)
    he, ho = hori(f)                                    # [H, W, C]
    out = np.zeros((2 * H, 2 * W, C), dtype=np.float32)
    for iy in range(H):
        ia = max(iy - 1, 0)
        ic = min(iy + 1, H - 1)
        # 行 2iy：0.25*row(ia) + 0.75*row(iy)
        oe = he[ia] * F32(0.25)
        t = he[iy] * F32(0.75)
        oe = oe + t
        out[2 * iy, 0::2, :] = oe
        oo = ho[ia] * F32(0.25)
        t = ho[iy] * F32(0.75)
        oo = oo + t
        out[2 * iy, 1::2, :] = oo
        # 行 2iy+1：0.75*row(iy) + 0.25*row(ic)
        oe = he[iy] * F32(0.75)
        t = he[ic] * F32(0.25)
        oe = oe + t
        out[2 * iy + 1, 0::2, :] = oe
        oo = ho[iy] * F32(0.75)
        t = ho[ic] * F32(0.25)
        oo = oo + t
        out[2 * iy + 1, 1::2, :] = oo
    return out


def resize_math_reference(x64: np.ndarray) -> np.ndarray:
    """独立实现（float64、按坐标公式、align_corners=False），用于交叉验证语义。"""
    x = x64.astype(np.float64)
    def src(i, n):
        u = (i + 0.5) / 2.0 - 0.5
        u = max(0.0, min(u, n - 1.0))
        lo = int(np.floor(u))
        hi = min(lo + 1, n - 1)
        return lo, hi, u - lo
    out = np.zeros((2 * H, 2 * W, C), dtype=np.float64)
    for oy in range(2 * H):
        y0, y1, fy = src(oy, H)
        for ox in range(2 * W):
            x0, x1, fx = src(ox, W)
            out[oy, ox] = ((1 - fy) * ((1 - fx) * x[y0, x0] + fx * x[y0, x1])
                           + fy * ((1 - fx) * x[y1, x0] + fx * x[y1, x1]))
    return out


def main():
    rng = np.random.default_rng(20261113)
    # 确定性输入：bf16 网格上的随机值（先 f32 -> bf16 -> f32）
    x_in = rng.uniform(-1.0, 1.0, size=(H, W, C)).astype(np.float32)
    x_bf16_bits = to_bf16_bits(x_in)
    x_bf16 = bf16_bits_to_f32(x_bf16_bits)

    y_kernel_f32 = resize_kernel_order(x_bf16)              # kernel 次序（f32）
    y_bf16_bits = to_bf16_bits(y_kernel_f32)                # 设备输出（bf16）
    y_math64 = resize_math_reference(x_bf16)                # 独立语义参照（f64）

    y_bf16_as_f32 = bf16_bits_to_f32(y_bf16_bits)
    d = np.abs(y_bf16_as_f32.astype(np.float64) - y_math64)
    print(f"input  : H={H} W={W} C={C} bf16, range [{x_bf16.min():.4f}, {x_bf16.max():.4f}]")
    print(f"output : {2*H}x{2*W}x{C} bf16")
    print(f"kernel-order f32 vs math f64: max|d| = {d.max():.3e}, mean|d| = {d.mean():.3e}")
    # bf16 ULP 量级检查：相邻 bf16 间距约为 |y|*2^-8
    ulp = np.maximum(np.abs(y_math64) * (2.0 ** -8), 2.0 ** -133)
    print(f"max|d| / max(ulp) = {(d / ulp)[d > 0].max() if np.any(d > 0) else 0:.3f} ULP(bf16)")
    print(f"exact-match ratio vs f64-derived rounding: {np.mean(y_bf16_bits == to_bf16_bits(y_math64)):.6f}")

    np.save("input_32x32x16_f32.npy", x_bf16.astype(np.float32))
    x_bf16_bits.astype(np.uint16).tofile("input_32x32x16_bf16.bin")
    y_bf16_bits.astype(np.uint16).tofile("expected_64x64x16_bf16.bin")
    np.save("expected_64x64x16_bf16.npy", y_bf16_as_f32)
    print("\nwrote: input_32x32x16_bf16.bin (32768 B), expected_64x64x16_bf16.bin (131072 B)")
    print("board 用法（将来有真机时）：")
    print("  dort calbin run -i 0 -f <calbin目录> -m resize_pkg --raw-in input_32x32x16_bf16.bin \\")
    print("    --raw-out out.bin --in-tensor in_x --out-tensor out_y")
    print("  然后逐元素比较 out.bin 与 expected_64x64x16_bf16.bin")


if __name__ == "__main__":
    main()
