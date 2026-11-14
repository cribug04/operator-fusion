#!/usr/bin/env python3
"""生成 Resize 大尺寸计时用例（128×128×256 bf16 ⇒ 256×256×256），供 resize_probe_huge / resize_probe_huge2 使用。

输入由固定种子生成，期望值用 onboard/resize_reference.py 的三路算法交叉验证后写出；
文件较大（输入 8 MiB、期望 32 MiB），不入库，在用板机器上现场生成，sha256 记在 docs/RESIZE_DUALCORE.md。
用法：python3 gen_resize_huge_case.py [输出目录，默认 onboard/cases/resize_huge]
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

H, W, C, CC = 128, 128, 256, 32
SEED = 20261114

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('resize_reference', ROOT / 'resize_reference.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'cases' / 'resize_huge'
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = ref.f32_to_bf16_grid(rng.standard_normal((H, W, C), dtype=np.float32) * 2.0)
    x_bits = ref.to_bf16_bits(x)
    expected = ref.to_bf16_bits(ref.kernel_order(x))
    independent = ref.to_bf16_bits_independent(ref.math_reference(x.astype(np.float64)))
    diff = int(np.count_nonzero(expected != independent))
    if diff:
        raise SystemExit(f'kernel 次序与独立公式不一致：{diff} 个元素')
    emu = ref.emulate_flat(x_bits.reshape(-1), H, W, C, CC).reshape(2 * H, 2 * W, C)
    diff = int(np.count_nonzero(expected != emu))
    if diff:
        raise SystemExit(f'kernel 次序与分块地址仿真不一致：{diff} 个元素')
    x_bits.astype('<u2').tofile(out / 'input.bin')
    expected.astype('<u2').tofile(out / 'expected.bin')
    for name in ('input.bin', 'expected.bin'):
        p = out / name
        print(f'{name}: {p.stat().st_size} B  sha256 {sha(p)}')
    print(f'PASS: {expected.size} 个输出元素三路一致（kernel 次序 / 独立公式 / 分块地址仿真）')


if __name__ == '__main__':
    main()
