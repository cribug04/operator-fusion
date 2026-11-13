#!/usr/bin/env python3
"""resize_bilinear2_tiled 的 CPU 参考（32×32×256 → 64×64×256 bf16）。
与 kernel 完全相同的运算次序：f32 每步 mul/add 各舍一次，最后 RNE 舍 bf16；边缘 clamp。
通道分块不影响数值（逐通道独立），所以参考同一套。"""
import numpy as np

F32 = np.float32
H, W, C = 32, 32, 256


def to_bf16_bits(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).copy()
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    return ((u + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_f32(b):
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


def hori(f):
    he = f * F32(0.75)
    t = np.zeros_like(f)
    t[:, 0:W - 1, :] = f[:, 0:W - 1, :] * F32(0.25)
    he[:, 1:W, :] = he[:, 1:W, :] + t[:, 0:W - 1, :]
    he[:, 0, :] = f[:, 0, :]
    ho = f * F32(0.75)
    t2 = np.zeros_like(f)
    t2[:, 0:W - 1, :] = f[:, 1:W, :] * F32(0.25)
    ho[:, 0:W - 1, :] = ho[:, 0:W - 1, :] + t2[:, 0:W - 1, :]
    ho[:, W - 1, :] = f[:, W - 1, :]
    return he, ho


def resize(f):
    he, ho = hori(f)
    out = np.zeros((2 * H, 2 * W, C), dtype=np.float32)
    for iy in range(H):
        ia, ic = max(iy - 1, 0), min(iy + 1, H - 1)
        oe = he[ia] * F32(0.25); oe = oe + he[iy] * F32(0.75)
        out[2 * iy, 0::2, :] = oe
        oo = ho[ia] * F32(0.25); oo = oo + ho[iy] * F32(0.75)
        out[2 * iy, 1::2, :] = oo
        oe = he[iy] * F32(0.75); oe = oe + he[ic] * F32(0.25)
        out[2 * iy + 1, 0::2, :] = oe
        oo = ho[iy] * F32(0.75); oo = oo + ho[ic] * F32(0.25)
        out[2 * iy + 1, 1::2, :] = oo
    return out


def math_ref(x64):
    def src(i, n):
        u = max(0.0, min((i + 0.5) / 2.0 - 0.5, n - 1.0))
        lo = int(np.floor(u)); hi = min(lo + 1, n - 1)
        return lo, hi, u - lo
    out = np.zeros((2 * H, 2 * W, C), dtype=np.float64)
    for oy in range(2 * H):
        y0, y1, fy = src(oy, H)
        for ox in range(2 * W):
            x0, x1, fx = src(ox, W)
            out[oy, ox] = ((1 - fy) * ((1 - fx) * x64[y0, x0] + fx * x64[y0, x1])
                           + fy * ((1 - fx) * x64[y1, x0] + fx * x64[y1, x1]))
    return out


def main():
    rng = np.random.default_rng(20261113 + 1)
    x = rng.uniform(-1.0, 1.0, size=(H, W, C)).astype(np.float32)
    xb = to_bf16_bits(x)
    xf = bf16_bits_to_f32(xb)
    yk = resize(xf)
    yb = to_bf16_bits(yk)
    ym = math_ref(xf)
    d = np.abs(bf16_bits_to_f32(yb).astype(np.float64) - ym)
    ulp = np.maximum(np.abs(ym) * (2.0 ** -8), 2.0 ** -133)
    print(f"input  : {H}x{W}x{C} bf16")
    print(f"output : {2*H}x{2*W}x{C} bf16")
    print(f"kernel-order f32 vs math f64: max|d|={d.max():.3e}  mean|d|={d.mean():.3e}")
    print(f"max|d|/ULP = {(d/ulp)[d>0].max():.3f}")
    print(f"bf16 舍入一致率 = {np.mean(yb == to_bf16_bits(ym)):.6f}")
    xb.astype(np.uint16).tofile("input_32x32x256_bf16.bin")
    yb.astype(np.uint16).tofile("expected_64x64x256_bf16.bin")
    print("wrote input_32x32x256_bf16.bin (524288 B), expected_64x64x256_bf16.bin (2097152 B)")


if __name__ == "__main__":
    main()
