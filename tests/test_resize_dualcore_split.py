"""resize_bilinear2_tiled2!（双核通道块分配）的 CPU 检查：两核各处理 NC/2 个块、块集合不重不漏、结果与单核分块版逐位相同。"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('resize_reference', ROOT / 'onboard' / 'resize_reference.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

# 与 kernel 相同的形状特化：(H, W, C, CC)
SHAPES = [(32, 32, 256, 64), (128, 128, 256, 32), (3, 3, 64, 32), (5, 4, 128, 64), (4, 5, 192, 32)]


def core_blocks(nc, rank, num_cores=2):
    """复刻宏里的 base = core_rank()*PER; for k in 0..PER: cci = base + k。"""
    per = nc // num_cores
    return [rank * per + k for k in range(per)]


def emulate_two_cores(x_bits_flat, h, w, c, cc):
    """两个核各自按块次序写同一条一维内存；每个地址仍应恰好写一次。"""
    nc = c // cc
    per_core = [core_blocks(nc, r) for r in range(2)]
    y = np.full(2 * h * 2 * w * c, 0xFFFF, dtype=np.uint32)
    written = np.zeros(y.size, dtype=np.uint8)
    for blocks in per_core:
        for cci in blocks:
            full = ref.emulate_flat(x_bits_flat, h, w, c, cc)  # 单核版的完整结果
            # 只取本块通道窗口，模拟本核写入的地址集合
            yy = full.reshape(2 * h, 2 * w, c)[:, :, cci * cc:(cci + 1) * cc]
            idx = (np.arange(2 * h)[:, None, None] * (2 * w * c) + np.arange(2 * w)[None, :, None] * c
                   + np.arange(cci * cc, (cci + 1) * cc)[None, None, :]).reshape(-1)
            if written[idx].any():
                raise AssertionError(f'块 {cci} 的地址被重复写')
            written[idx] = 1
            y[idx] = yy.reshape(-1)
    if not written.all():
        raise AssertionError('有地址没写到')
    return y.astype(np.uint16)


class DualCoreSplit(unittest.TestCase):
    def test_blocks_partition(self):
        for h, w, c, cc in SHAPES:
            nc = c // cc
            with self.subTest(shape=(h, w, c, cc)):
                self.assertEqual(nc % 2, 0, '宏的 const assert：块数必须是偶数')
                b0, b1 = core_blocks(nc, 0), core_blocks(nc, 1)
                self.assertEqual(sorted(b0 + b1), list(range(nc)))
                self.assertFalse(set(b0) & set(b1))
                self.assertEqual(len(b0), len(b1))

    def test_single_core_call_misses_half(self):
        """反例：只由 0 号核调用（VAE 审查里的那类错误）时，只有前一半通道块会被写。"""
        nc = 8
        self.assertEqual(core_blocks(nc, 0), [0, 1, 2, 3])

    def test_two_core_result_equals_tiled(self):
        rng = np.random.default_rng(7)
        for h, w, c, cc in SHAPES[2:]:
            with self.subTest(shape=(h, w, c, cc)):
                x = ref.f32_to_bf16_grid(rng.standard_normal((h, w, c), dtype=np.float32))
                xb = ref.to_bf16_bits(x).reshape(-1)
                two = emulate_two_cores(xb, h, w, c, cc)
                one = ref.emulate_flat(xb, h, w, c, cc)
                self.assertTrue(np.array_equal(two, one))
                self.assertTrue(np.array_equal(two.reshape(2 * h, 2 * w, c), ref.to_bf16_bits(ref.kernel_order(x))))


if __name__ == '__main__':
    unittest.main()
