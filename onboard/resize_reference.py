#!/usr/bin/env python3
"""resize_bilinear2 / resize_bilinear2_tiled 的形状通用 CPU 参考与地址仿真。

只在 CPU 上运行，不涉及 NPU。三条彼此独立的路径：

1. ``kernel_order``        —— 逐段复刻 kernel 宏的运算次序（第 0 行 / 中间行 / 最后一行三段展开，
                              行内 he/ho 公式，f32 每步 mul/add 各舍一次，最后 RNE 舍 bf16）。
2. ``emulate_flat``        —— 用 kernel 里 ``bind_strided`` / ``sub`` / ``add`` 的**地址算式**在一维内存上
                              仿真 load/store，同时检查每个输出地址恰好写一次；用于验证通道分块偏移与
                              四相位输出视图的地址映射，而不是靠 numpy 切片“默认正确”。
3. ``math_reference``      —— 独立坐标公式（float64，``u = (o + 0.5) / 2 - 0.5``，clamp，lerp），
                              与 kernel 的写法没有共享任何中间步骤；并提供独立的 bf16 RNE 舍入实现。

约束检查 ``check_plain`` / ``check_tiled`` 严格镜像 ``kernels/resize_bilinear2*.rs`` 里的 ``const _: () = assert!``。
"""
from __future__ import annotations

import numpy as np

F32 = np.float32

# ---------------------------------------------------------------- bf16 转换（两套互相独立）

def to_bf16_bits(a: np.ndarray) -> np.ndarray:
    """f32 -> bf16 位型（RNE），与 evidence/*/reference*.py 相同的整数加法写法。"""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).copy()
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    return ((u + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype(np.uint16)


def to_bf16_bits_independent(a: np.ndarray) -> np.ndarray:
    """独立的 bf16 RNE 实现：用 frexp 把尾数量化到 8 个有效位（np.rint 为 round-half-even），
    再取 f32 位型的高 16 位。只处理 0 与正规数（断言低 16 位为零，保证没有信息丢失）。"""
    x = np.asarray(a, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('独立 bf16 舍入只接受有限值')
    mant, exp = np.frexp(x)                       # x = mant * 2**exp, 0.5 <= |mant| < 1
    scale = np.ldexp(1.0, exp - 8)                # 8 个有效位
    q = np.rint(x / scale) * scale
    q32 = q.astype(np.float32)
    if (np.abs(q32[q32 != 0]) < np.float32(2.0 ** -126)).any():
        raise ValueError('独立 bf16 舍入不处理非正规数')
    bits = q32.view(np.uint32)
    if (bits & np.uint32(0xFFFF)).any():
        raise AssertionError('独立 bf16 舍入结果不在 bf16 网格上')
    return (bits >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_f32(b: np.ndarray) -> np.ndarray:
    return (np.asarray(b).astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16_grid(a: np.ndarray) -> np.ndarray:
    """把任意 f32 数据落到 bf16 网格上（返回 f32）。"""
    return bf16_bits_to_f32(to_bf16_bits(a))


# ---------------------------------------------------------------- 约束（镜像 kernel 的 const assert）

VB_CACHE_BYTES = 384 * 1024
CACHE_BYTES_PER_ELEMENT = 56          # 12 份 f32 + 4 份 bf16 中间 cache


def check_plain(h: int, w: int, c: int) -> None:
    """resize_bilinear2!：H、W >= 3；1 <= C <= 128。"""
    if not (h >= 3 and w >= 3):
        raise ValueError(f'resize_bilinear2：H、W 至少 3，收到 H={h} W={w}')
    if not (1 <= c <= 128):
        raise ValueError(f'resize_bilinear2：C 必须 1..=128，收到 C={c}')


def check_tiled(h: int, w: int, c: int, cc: int) -> None:
    """resize_bilinear2_tiled!：H、W >= 3；CC >= 32 且 CC % 32 == 0 且 C % CC == 0；W*CC*56 <= 384 KiB。"""
    if not (h >= 3 and w >= 3):
        raise ValueError(f'resize_bilinear2_tiled：H、W 至少 3，收到 H={h} W={w}')
    if not (cc >= 32 and cc % 32 == 0 and c % cc == 0):
        raise ValueError(f'resize_bilinear2_tiled：CC 是 >=32 的 32 倍数且整除 C，收到 C={c} CC={cc}')
    if w * cc * CACHE_BYTES_PER_ELEMENT > VB_CACHE_BYTES:
        raise ValueError(f'resize_bilinear2_tiled：W*CC*56 B = {w * cc * CACHE_BYTES_PER_ELEMENT} 超过 384 KiB')


# ---------------------------------------------------------------- 路径 1：kernel 运算次序（numpy 切片）

def _hori(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """f: [W, C] f32 -> (he, ho)，严格按 @h 宏的 mul/add/copy 次序。"""
    w = f.shape[0]
    he = f * F32(0.75)
    t = np.zeros_like(f)
    t[0:w - 1] = f[0:w - 1] * F32(0.25)
    he[1:w] = he[1:w] + t[0:w - 1]
    he[0] = f[0]
    ho = f * F32(0.75)
    t[0:w - 1] = f[1:w] * F32(0.25)
    ho[0:w - 1] = ho[0:w - 1] + t[0:w - 1]
    ho[w - 1] = f[w - 1]
    return he, ho


def _row_triples(h: int):
    """kernel 的三段展开：(ia, iy, ic)。第 0 行 (0,0,1)；中间 (iy-1, iy, iy+1)；最后一行 (H-2, H-1, H-1)。"""
    yield 0, 0, 1
    for iy in range(1, h - 1):
        yield iy - 1, iy, iy + 1
    yield h - 2, h - 1, h - 1


def kernel_order(x_f32: np.ndarray) -> np.ndarray:
    """x: [H, W, C] f32（bf16 网格）-> [2H, 2W, C] f32（未舍 bf16），按 kernel 次序。"""
    x = np.ascontiguousarray(x_f32, dtype=np.float32)
    h, w, c = x.shape
    out = np.zeros((2 * h, 2 * w, c), dtype=np.float32)
    rows = [_hori(x[i]) for i in range(h)]
    for ia, iy, ic in _row_triples(h):
        (he_a, ho_a), (he_b, ho_b), (he_c, ho_c) = rows[ia], rows[iy], rows[ic]
        oe = he_a * F32(0.25); t = he_b * F32(0.75); oe = oe + t
        out[2 * iy, 0::2] = oe
        oo = ho_a * F32(0.25); t = ho_b * F32(0.75); oo = oo + t
        out[2 * iy, 1::2] = oo
        oe = he_b * F32(0.75); t = he_c * F32(0.25); oe = oe + t
        out[2 * iy + 1, 0::2] = oe
        oo = ho_b * F32(0.75); t = ho_c * F32(0.25); oo = oo + t
        out[2 * iy + 1, 1::2] = oo
    return out


# ---------------------------------------------------------------- 路径 2：一维内存地址仿真（含通道分块）

class _Strided:
    """模拟 ``Ptr::add(off_bytes).bind_strided::<_, N>(shape, strides)`` 后的 ``sub(start, size)``，
    产出一维内存里的元素地址数组。所有量都以 bf16 元素为单位（kernel 里 byte 偏移是 ``*2``）。"""

    def __init__(self, base_elems: int, shape, strides):
        self.base = base_elems
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        assert len(self.shape) == len(self.strides)

    def sub(self, start, size) -> np.ndarray:
        for s, n, lim in zip(start, size, self.shape):
            if not (0 <= s and s + n <= lim):
                raise IndexError(f'sub 越界：start={start} size={size} shape={self.shape}')
        grids = np.meshgrid(*[np.arange(s, s + n) for s, n in zip(start, size)], indexing='ij')
        addr = np.full(grids[0].shape, self.base, dtype=np.int64)
        for g, st in zip(grids, self.strides):
            addr = addr + g * st
        return addr


def emulate_flat(x_bits_flat: np.ndarray, h: int, w: int, c: int, cc: int | None = None) -> np.ndarray:
    """按 kernel 的地址算式在一维 bf16 内存上仿真。

    ``cc=None`` 对应 ``resize_bilinear2!``（单块、xoff=0）；给定 ``cc`` 对应 ``resize_bilinear2_tiled!``。
    返回输出的一维 bf16 位型（长度 2H*2W*C）。每个输出地址必须恰好写一次，否则抛错。
    """
    x_bits_flat = np.asarray(x_bits_flat, dtype=np.uint16).ravel()
    if x_bits_flat.size != h * w * c:
        raise ValueError(f'输入长度 {x_bits_flat.size} != H*W*C = {h * w * c}')
    if cc is None:
        check_plain(h, w, c)
        cc, nc = c, 1
    else:
        check_tiled(h, w, c, cc)
        nc = c // cc
    wo = 2 * w
    out_len = 2 * h * wo * c
    y = np.zeros(out_len, dtype=np.uint16)
    writes = np.zeros(out_len, dtype=np.int32)

    def load(view: _Strided, start, size) -> np.ndarray:
        addr = view.sub(start, size)
        if addr.min() < 0 or addr.max() >= x_bits_flat.size:
            raise IndexError('load 越过输入缓冲')
        return bf16_bits_to_f32(x_bits_flat[addr]).reshape(size[-2], size[-1])

    def store(view: _Strided, start, size, val_f32: np.ndarray) -> None:
        addr = view.sub(start, size)
        if addr.min() < 0 or addr.max() >= out_len:
            raise IndexError('store 越过输出缓冲')
        y[addr.ravel()] = to_bf16_bits(val_f32).ravel()
        np.add.at(writes, addr.ravel(), 1)

    for cci in range(nc):
        xoff = cci * cc
        # gx = x.add(xoff*2).bind_strided([H, W, CC], [W*C, C, 1])
        gx = _Strided(xoff, [h, w, cc], [w * c, c, 1])
        # g_ab = y.add(((a*WO + b)*C + xoff)*2).bind_strided([H*WO, CC], [2*C, 1])
        g = {(a, b): _Strided((a * wo + b) * c + xoff, [h * wo, cc], [2 * c, 1]) for a in (0, 1) for b in (0, 1)}
        for ia, iy, ic in _row_triples(h):
            af = load(gx, [ia, 0, 0], [1, w, cc])
            bf = load(gx, [iy, 0, 0], [1, w, cc])
            cf = load(gx, [ic, 0, 0], [1, w, cc])
            he_a, ho_a = _hori(af)
            he_b, ho_b = _hori(bf)
            he_c, ho_c = _hori(cf)
            oe = he_a * F32(0.25); t = he_b * F32(0.75); oe = oe + t
            store(g[(0, 0)], [iy * wo, 0], [w, cc], oe)
            oo = ho_a * F32(0.25); t = ho_b * F32(0.75); oo = oo + t
            store(g[(0, 1)], [iy * wo, 0], [w, cc], oo)
            oe = he_b * F32(0.75); t = he_c * F32(0.25); oe = oe + t
            store(g[(1, 0)], [iy * wo, 0], [w, cc], oe)
            oo = ho_b * F32(0.75); t = ho_c * F32(0.25); oo = oo + t
            store(g[(1, 1)], [iy * wo, 0], [w, cc], oo)

    if not (writes == 1).all():
        missing = int(np.count_nonzero(writes == 0))
        dup = int(np.count_nonzero(writes > 1))
        raise AssertionError(f'输出地址覆盖异常：未写 {missing} 个，重复写 {dup} 个')
    return y


# ---------------------------------------------------------------- 路径 3：独立坐标公式（float64）

def source_coord(o: int, n: int) -> tuple[int, int, float]:
    """align_corners=False、scale=2：u = (o+0.5)/2 - 0.5，clamp 到 [0, n-1]；返回 (lo, hi, frac)。"""
    u = (o + 0.5) / 2.0 - 0.5
    u = min(max(u, 0.0), float(n - 1))
    lo = int(np.floor(u))
    hi = min(lo + 1, n - 1)
    return lo, hi, u - lo


def axis_weights(n: int) -> np.ndarray:
    """返回 [2n, n] 矩阵 Wm，Wm[o, i] = 输入下标 i 对输出 o 的权重（每行和为 1）。"""
    m = np.zeros((2 * n, n), dtype=np.float64)
    for o in range(2 * n):
        lo, hi, fr = source_coord(o, n)
        m[o, lo] += 1.0 - fr
        m[o, hi] += fr
    return m


def math_reference(x: np.ndarray) -> np.ndarray:
    """x: [H, W, C] -> [2H, 2W, C] float64：行权重矩阵 × x × 列权重矩阵ᵀ（与 kernel 写法无关）。"""
    x64 = np.asarray(x, dtype=np.float64)
    h, w, _ = x64.shape
    wy, wx = axis_weights(h), axis_weights(w)
    return np.einsum('oy,ywc,pw->opc', wy, x64, wx)


# ---------------------------------------------------------------- 辅助

def ulp_distance_bf16(a_bits: np.ndarray, b_bits: np.ndarray) -> np.ndarray:
    """两组 bf16 位型之间的近似 ULP 距离（把位型映射到单调整数后相减）。"""
    def ordered(bits):
        b = np.asarray(bits).astype(np.int64)
        return np.where(b & 0x8000, -(b & 0x7FFF), b)
    return np.abs(ordered(a_bits) - ordered(b_bits))
