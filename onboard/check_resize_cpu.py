#!/usr/bin/env python3
"""重新验证两个已提交 Resize 用例；只验证 CPU 参考与数据，不验证 NPU。"""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load(path):
    spec = importlib.util.spec_from_file_location('reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    for folder, reference, channels in [
        ('resize_small', 'resize/reference.py', 16),
        ('resize_big', 'resize_big/reference_big.py', 256),
    ]:
        module = load(ROOT / 'evidence' / reference)
        case = ROOT / 'onboard/cases' / folder
        bits = np.fromfile(case / 'input.bin', dtype='<u2').reshape(32, 32, channels)
        expected = np.fromfile(case / 'expected.bin', dtype='<u2').reshape(64, 64, channels)
        values = module.bf16_bits_to_f32(bits)
        kernel_order = getattr(module, 'resize_kernel_order', None) or module.resize
        mathematical = getattr(module, 'resize_math_reference', None) or module.math_ref
        result = module.to_bf16_bits(kernel_order(values))
        independent = module.to_bf16_bits(mathematical(values.astype(np.float64)))
        if not np.isfinite(module.bf16_bits_to_f32(expected)).all():
            raise ValueError(f'{folder}: 参考包含 NaN/Inf')
        for name, actual in [('按 kernel 次序计算', result), ('独立坐标公式计算', independent)]:
            count = int(np.count_nonzero(actual != expected))
            print(f'{folder}: {name}; {expected.size} 元素; 差异 {count}')
            if count:
                raise ValueError(f'{folder}: CPU 参考不一致')
    print('PASS: 两个已提交 Resize 用例的 CPU 参考与数据一致；未执行 NPU。')

if __name__ == '__main__':
    main()
