"""CLI 回归测试：错误数据必须返回失败，严格/宽松判据必须分开。"""
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / 'onboard/compare_bf16.py'

class CompareTests(unittest.TestCase):
    def check(self, actual, expected, code, *options):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            def pack(values):
                return values if isinstance(values, bytes) else struct.pack('<' + 'H' * len(values), *values)
            (root / 'actual').write_bytes(pack(actual))
            (root / 'expected').write_bytes(pack(expected))
            result = subprocess.run([sys.executable, str(SCRIPT), str(root/'actual'), str(root/'expected'), *options], capture_output=True, text=True)
            self.assertEqual(result.returncode, code, result.stdout + result.stderr)

    def test_exact(self):
        self.check([0, 0x3f80, 0xbf80], [0, 0x3f80, 0xbf80], 0, '--mode', 'exact')

    def test_strict_rejects_one_bit_difference(self):
        self.check([0x3f81] + [0x3f80]*999, [0x3f80]*1000, 1, '--mode', 'exact')

    def test_resize_allows_rare_rounding(self):
        self.check([0x3f81] + [0x3f80]*999, [0x3f80]*1000, 0)

    def test_resize_rejects_widespread_difference(self):
        self.check([0x3f81]*10, [0x3f80]*10, 1)

    def test_conv_cpu_tolerance(self):
        self.check([0x3f83]*10, [0x3f80]*10, 0, '--mode', 'conv-cpu')

    def test_conv_cpu_ulp_limit(self):
        self.check([0x3f85]*10, [0x3f80]*10, 1, '--mode', 'conv-cpu')

    def test_conv_cpu_cosine_limit(self):
        self.check([0x3f84, 0x3f7c], [0x3f80, 0x3f80], 1, '--mode', 'conv-cpu')

    def test_size_mismatch(self):
        self.check([0x3f80], [0x3f80]*2, 2)

    def test_empty_and_truncated(self):
        for values in [b'', b'\x00']:
            with self.subTest(values=values):
                self.check(values, [0], 2)

    def test_nonfinite_even_when_identical(self):
        for value in [0x7fc0, 0x7f80, 0xff80]:
            with self.subTest(value=value):
                self.check([value], [value], 2)

    def test_zero_vectors(self):
        self.check([0]*8, [0]*8, 0, '--mode', 'conv-cpu')
        self.check([0]*8, [0x3f80]*8, 1, '--mode', 'conv-cpu')

    def test_crop_validation(self):
        for options in [('--crop','0:1,0:1'), ('--actual-shape','2,2,1','--crop','0:3,0:1'), ('--actual-shape','2,2,1','--crop','1:1,0:1')]:
            with self.subTest(options=options):
                self.check([0]*4, [0]*4, 2, *options)

    def test_shape_mismatch_same_count(self):
        self.check([0]*4,[0]*4,2,'--actual-shape','2,2,1','--expected-shape','1,4,1')

    def test_valid_crop(self):
        self.check([0,0x3f80,0,0], [0x3f80], 0, '--actual-shape','2,2,1','--crop','0:1,1:2','--mode','exact')

if __name__ == '__main__':
    unittest.main()
