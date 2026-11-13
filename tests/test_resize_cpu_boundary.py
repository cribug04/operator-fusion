"""Resize ×2 的 CPU 边界测试（只验证 CPU 参考与地址仿真，不涉及 NPU）。

三条独立路径互相对照（见 onboard/resize_reference.py）：
  kernel_order  —— 复刻 kernel 宏的运算次序；
  emulate_flat  —— 用 kernel 的 bind_strided/sub/add 地址算式在一维内存上仿真（含通道分块）；
  math_reference —— 独立坐标公式（float64）+ 独立 bf16 RNE 舍入。
测试尺寸全部满足 kernel 的 const assert：H、W >= 3；普通版 1 <= C <= 128；
分块版 CC >= 32、CC % 32 == 0、C % CC == 0、W*CC*56 B <= 384 KiB。
"""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'onboard'))
import resize_reference as R  # noqa: E402

PLAIN_SHAPES = [(3, 3, 16), (3, 5, 16), (5, 3, 16), (4, 7, 32), (7, 4, 16), (8, 8, 128), (32, 32, 16)]
TILED_SHAPES = [(3, 3, 64, 32), (4, 5, 96, 32), (5, 4, 128, 64), (6, 6, 256, 64), (3, 7, 320, 32), (32, 32, 256, 64)]


def grid(a):
    """落到 bf16 网格的 f32 数组。"""
    return R.f32_to_bf16_grid(np.asarray(a, dtype=np.float32))


def run_all(x):
    """返回 (kernel_order 位型, emulate_flat 位型, 独立公式位型)，形状均为 [2H, 2W, C] uint16。"""
    h, w, c = x.shape
    ko = R.to_bf16_bits(R.kernel_order(x))
    em = R.emulate_flat(R.to_bf16_bits(x).ravel(), h, w, c).reshape(2 * h, 2 * w, c)
    mi = R.to_bf16_bits_independent(R.math_reference(x))
    return ko, em, mi


class ResizeBoundaryTests(unittest.TestCase):
    def assert_three_way(self, x, exact_math=True):
        ko, em, mi = run_all(x)
        self.assertEqual(int(np.count_nonzero(ko != em)), 0, 'kernel 次序 vs 地址仿真不一致')
        d = R.ulp_distance_bf16(ko, mi)
        if exact_math:
            self.assertEqual(int(np.count_nonzero(d)), 0, 'kernel 次序 vs 独立坐标公式不一致')
        else:
            self.assertLessEqual(int(d.max()), 1, '与独立公式差距超过 1 ULP')
            self.assertGreaterEqual(float(np.mean(d == 0)), 0.999, '与独立公式逐位相同率低于 99.9%')
        return ko

    # ------------------------------------------------------------ 常量
    def test_constant(self):
        for value in [0.0, 1.0, -1.0, 0.5, 3.0, -2.75, 1e-3, 1e4]:
            for h, w, c in PLAIN_SHAPES[:4]:
                with self.subTest(value=value, shape=(h, w, c)):
                    x = grid(np.full((h, w, c), value))
                    out = self.assert_three_way(x)
                    self.assertTrue((out == R.to_bf16_bits(x[0, 0, 0])).all(), '常量输入的输出必须处处等于该常量')

    # ------------------------------------------------------------ 脉冲
    def test_impulse_interior_weights(self):
        h, w, c = 5, 7, 16
        x = np.zeros((h, w, c), dtype=np.float32)
        x[2, 3, 5] = 1.0
        out = R.bf16_bits_to_f32(self.assert_three_way(x))
        # 2×、align_corners=False：权重只有 0.75/0.25；脉冲在 (2,3) 影响输出行 3..6、列 5..8
        exp = np.zeros((2 * h, 2 * w), dtype=np.float32)
        wy = {3: 0.25, 4: 0.75, 5: 0.75, 6: 0.25}
        wx = {5: 0.25, 6: 0.75, 7: 0.75, 8: 0.25}
        for oy, a in wy.items():
            for ox, b in wx.items():
                exp[oy, ox] = a * b
        np.testing.assert_array_equal(out[:, :, 5], exp)
        self.assertEqual(float(np.abs(out[:, :, [k for k in range(c) if k != 5]]).max()), 0.0, '脉冲不能串到其他通道')

    def test_impulse_corners_and_edges_clamp(self):
        h, w, c = 4, 5, 16
        cases = {
            (0, 0): ({0: 1.0, 1: 0.75, 2: 0.25}, {0: 1.0, 1: 0.75, 2: 0.25}),
            (h - 1, w - 1): ({2 * h - 1: 1.0, 2 * h - 2: 0.75, 2 * h - 3: 0.25}, {2 * w - 1: 1.0, 2 * w - 2: 0.75, 2 * w - 3: 0.25}),
            (0, 2): ({0: 1.0, 1: 0.75, 2: 0.25}, {3: 0.25, 4: 0.75, 5: 0.75, 6: 0.25}),
            (2, 0): ({3: 0.25, 4: 0.75, 5: 0.75, 6: 0.25}, {0: 1.0, 1: 0.75, 2: 0.25}),
        }
        for (iy, ix), (wy, wx) in cases.items():
            with self.subTest(impulse=(iy, ix)):
                x = np.zeros((h, w, c), dtype=np.float32)
                x[iy, ix, :] = 1.0
                out = R.bf16_bits_to_f32(self.assert_three_way(x))
                exp = np.zeros((2 * h, 2 * w), dtype=np.float32)
                for oy, a in wy.items():
                    for ox, b in wx.items():
                        exp[oy, ox] = a * b
                for k in range(c):
                    np.testing.assert_array_equal(out[:, :, k], exp)

    def test_impulse_every_position_minimal_shape(self):
        h, w, c = 3, 3, 16
        for iy in range(h):
            for ix in range(w):
                with self.subTest(pos=(iy, ix)):
                    x = np.zeros((h, w, c), dtype=np.float32)
                    x[iy, ix, :] = 1.0
                    out = R.bf16_bits_to_f32(self.assert_three_way(x))
                    # 每个输出像素的权重和为 1 ⇒ 对所有脉冲位置求和后处处为 1；这里检查单脉冲时每个输出 ∈ [0,1]
                    self.assertTrue((out >= 0).all() and (out <= 1).all())

    def test_impulse_partition_of_unity(self):
        h, w, c = 4, 6, 16
        total = np.zeros((2 * h, 2 * w, c), dtype=np.float64)
        for iy in range(h):
            for ix in range(w):
                x = np.zeros((h, w, c), dtype=np.float32)
                x[iy, ix, :] = 1.0
                total += R.bf16_bits_to_f32(R.to_bf16_bits(R.kernel_order(x)))
        np.testing.assert_array_equal(total, 1.0)

    # ------------------------------------------------------------ 渐变
    def test_horizontal_gradient(self):
        for h, w, c in PLAIN_SHAPES:
            with self.subTest(shape=(h, w, c)):
                x = grid(np.broadcast_to(np.arange(w, dtype=np.float32)[None, :, None], (h, w, c)))
                out = R.bf16_bits_to_f32(self.assert_three_way(x))
                # 线性函数的双线性插值 = 源坐标本身（clamp 后）
                u = np.array([min(max((ox + 0.5) / 2 - 0.5, 0.0), w - 1.0) for ox in range(2 * w)], dtype=np.float32)
                np.testing.assert_array_equal(out, np.broadcast_to(u[None, :, None], out.shape))

    def test_vertical_gradient(self):
        for h, w, c in PLAIN_SHAPES:
            with self.subTest(shape=(h, w, c)):
                x = grid(np.broadcast_to(np.arange(h, dtype=np.float32)[:, None, None], (h, w, c)))
                out = R.bf16_bits_to_f32(self.assert_three_way(x))
                u = np.array([min(max((oy + 0.5) / 2 - 0.5, 0.0), h - 1.0) for oy in range(2 * h)], dtype=np.float32)
                np.testing.assert_array_equal(out, np.broadcast_to(u[:, None, None], out.shape))

    def test_diagonal_gradient_rows_cols_independent(self):
        h, w, c = 6, 5, 32
        x = grid(np.arange(h)[:, None, None] * 4.0 + np.arange(w)[None, :, None] + np.zeros((1, 1, c)))
        out = R.bf16_bits_to_f32(self.assert_three_way(x))
        uy = np.array([min(max((o + 0.5) / 2 - 0.5, 0.0), h - 1.0) for o in range(2 * h)])
        ux = np.array([min(max((o + 0.5) / 2 - 0.5, 0.0), w - 1.0) for o in range(2 * w)])
        np.testing.assert_array_equal(out, (uy[:, None, None] * 4.0 + ux[None, :, None] + np.zeros((1, 1, c))).astype(np.float32))

    # ------------------------------------------------------------ 负值
    def test_negative_gradient_and_sign_symmetry(self):
        for h, w, c in PLAIN_SHAPES[:5]:
            with self.subTest(shape=(h, w, c)):
                x = grid(-np.broadcast_to(np.arange(w, dtype=np.float32)[None, :, None], (h, w, c)) - 0.5)
                out = self.assert_three_way(x)
                neg = R.to_bf16_bits(R.kernel_order(-x))
                # RNE 对称：resize(-x) 的位型 = resize(x) 的位型取反符号位（0 的符号除外）
                flipped = (out ^ np.uint16(0x8000)).astype(np.uint16)
                zero = (out & np.uint16(0x7FFF)) == 0
                self.assertTrue(((neg == flipped) | zero).all())

    def test_mixed_sign_random(self):
        rng = np.random.default_rng(2026_11_14)
        for h, w, c in PLAIN_SHAPES:
            with self.subTest(shape=(h, w, c)):
                x = grid(rng.uniform(-8.0, 8.0, size=(h, w, c)))
                self.assert_three_way(x, exact_math=False)

    def test_wide_dynamic_range(self):
        rng = np.random.default_rng(7)
        h, w, c = 5, 6, 16
        mag = np.exp(rng.uniform(np.log(1e-3), np.log(1e3), size=(h, w, c)))
        x = grid(mag * rng.choice([-1.0, 1.0], size=(h, w, c)))
        self.assert_three_way(x, exact_math=False)

    # ------------------------------------------------------------ 边缘 clamp
    def test_edge_clamp_random_input(self):
        rng = np.random.default_rng(99)
        for h, w, c in PLAIN_SHAPES:
            with self.subTest(shape=(h, w, c)):
                x = grid(rng.uniform(-1.0, 1.0, size=(h, w, c)))
                out = self.assert_three_way(x, exact_math=False)
                xb = R.to_bf16_bits(x)
                # 四个角：所有权重都 clamp 到同一输入像素 ⇒ 逐位相同
                np.testing.assert_array_equal(out[0, 0], xb[0, 0])
                np.testing.assert_array_equal(out[0, -1], xb[0, -1])
                np.testing.assert_array_equal(out[-1, 0], xb[-1, 0])
                np.testing.assert_array_equal(out[-1, -1], xb[-1, -1])
                # 首/末输出行只依赖首/末输入行；首/末输出列只依赖首/末输入列
                row0 = R.to_bf16_bits(R.kernel_order(np.repeat(x[0:1], h, axis=0)))
                np.testing.assert_array_equal(out[0], row0[0])
                rowl = R.to_bf16_bits(R.kernel_order(np.repeat(x[h - 1:h], h, axis=0)))
                np.testing.assert_array_equal(out[-1], rowl[-1])
                col0 = R.to_bf16_bits(R.kernel_order(np.repeat(x[:, 0:1], w, axis=1)))
                np.testing.assert_array_equal(out[:, 0], col0[:, 0])
                coll = R.to_bf16_bits(R.kernel_order(np.repeat(x[:, w - 1:w], w, axis=1)))
                np.testing.assert_array_equal(out[:, -1], coll[:, -1])

    def test_edge_clamp_does_not_bleed_across_boundary(self):
        # 只有边界行/列非零时，输出不能越过 clamp 把值“镜像”到更远处
        h, w, c = 5, 5, 16
        x = np.zeros((h, w, c), dtype=np.float32)
        x[0, :, :] = 1.0
        out = R.bf16_bits_to_f32(self.assert_three_way(x))
        np.testing.assert_array_equal(out[0], 1.0)
        np.testing.assert_array_equal(out[1], 0.75)
        np.testing.assert_array_equal(out[2], 0.25)
        np.testing.assert_array_equal(out[3:], 0.0)

    # ------------------------------------------------------------ 通道分块
    def test_tiled_matches_plain_and_formula(self):
        rng = np.random.default_rng(1234)
        for h, w, c, cc in TILED_SHAPES:
            with self.subTest(shape=(h, w, c, cc)):
                x = grid(rng.uniform(-1.0, 1.0, size=(h, w, c)))
                xb = R.to_bf16_bits(x)
                tiled = R.emulate_flat(xb.ravel(), h, w, c, cc).reshape(2 * h, 2 * w, c)
                ko = R.to_bf16_bits(R.kernel_order(x))
                self.assertEqual(int(np.count_nonzero(tiled != ko)), 0, '分块地址仿真 vs 不分块 kernel 次序不一致')
                mi = R.to_bf16_bits_independent(R.math_reference(x))
                d = R.ulp_distance_bf16(tiled, mi)
                self.assertLessEqual(int(d.max()), 1)
                self.assertGreaterEqual(float(np.mean(d == 0)), 0.999)

    def test_tiled_channel_block_boundaries(self):
        """每个通道写入独立标识值：某通道块的输出若混进其他块的值即可检出。"""
        for h, w, c, cc in [(3, 3, 64, 32), (4, 5, 96, 32), (3, 4, 256, 64)]:
            with self.subTest(shape=(h, w, c, cc)):
                chan = np.arange(c, dtype=np.float32) + 1.0  # 1..C，bf16 可精确表示（C <= 256）
                x = grid(np.broadcast_to(chan[None, None, :], (h, w, c)))
                tiled = R.emulate_flat(R.to_bf16_bits(x).ravel(), h, w, c, cc).reshape(2 * h, 2 * w, c)
                out = R.bf16_bits_to_f32(tiled)
                np.testing.assert_array_equal(out, np.broadcast_to(chan[None, None, :], out.shape))
                # 只在块边界通道 (cc-1, cc) 放脉冲，确认块偏移 xoff 正确
                for k in [cc - 1, cc, c - cc - 1, c - 1]:
                    xi = np.zeros((h, w, c), dtype=np.float32)
                    xi[1, 1, k] = 1.0
                    t = R.bf16_bits_to_f32(R.emulate_flat(R.to_bf16_bits(xi).ravel(), h, w, c, cc).reshape(2 * h, 2 * w, c))
                    others = [j for j in range(c) if j != k]
                    self.assertEqual(float(np.abs(t[:, :, others]).max()), 0.0, f'通道 {k} 的脉冲串到了其他通道')
                    self.assertEqual(float(t[2, 2, k]), 0.5625)
                    self.assertEqual(float(t[3, 3, k]), 0.5625)

    def test_tiled_single_block_equals_plain_addresses(self):
        h, w, c = 5, 6, 64
        rng = np.random.default_rng(5)
        xb = R.to_bf16_bits(grid(rng.uniform(-1, 1, size=(h, w, c)))).ravel()
        np.testing.assert_array_equal(R.emulate_flat(xb, h, w, c, None), R.emulate_flat(xb, h, w, c, 64))
        np.testing.assert_array_equal(R.emulate_flat(xb, h, w, c, 64), R.emulate_flat(xb, h, w, c, 32))

    # ------------------------------------------------------------ 约束
    def test_constraints_mirror_kernel_asserts(self):
        for bad in [(2, 3, 16), (3, 2, 16), (3, 3, 0), (3, 3, 129)]:
            with self.subTest(plain=bad):
                self.assertRaises(ValueError, R.check_plain, *bad)
        for bad in [(2, 3, 64, 32), (3, 3, 64, 16), (3, 3, 64, 48), (3, 3, 96, 64), (3, 3, 320, 128), (3, 400, 256, 32)]:
            with self.subTest(tiled=bad):
                self.assertRaises(ValueError, R.check_tiled, *bad)
        for h, w, c in PLAIN_SHAPES:
            R.check_plain(h, w, c)
        for h, w, c, cc in TILED_SHAPES:
            R.check_tiled(h, w, c, cc)

    def test_emulation_rejects_wrong_input_length(self):
        self.assertRaises(ValueError, R.emulate_flat, np.zeros(10, dtype=np.uint16), 3, 3, 16)

    # ------------------------------------------------------------ 已提交用例回归
    def test_committed_cases_three_way(self):
        root = pathlib.Path(__file__).resolve().parents[1] / 'onboard/cases'
        for folder, c, cc in [('resize_small', 16, None), ('resize_big', 256, 64)]:
            with self.subTest(case=folder):
                xb = np.fromfile(root / folder / 'input.bin', dtype='<u2')
                eb = np.fromfile(root / folder / 'expected.bin', dtype='<u2')
                self.assertEqual(xb.size, 32 * 32 * c)
                em = R.emulate_flat(xb, 32, 32, c, cc)
                np.testing.assert_array_equal(em, eb)
                x = R.bf16_bits_to_f32(xb).reshape(32, 32, c)
                mi = R.to_bf16_bits_independent(R.math_reference(x)).ravel()
                np.testing.assert_array_equal(mi, eb)


if __name__ == '__main__':
    unittest.main()
