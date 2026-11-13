#!/usr/bin/env python3
"""bf16 原始二进制对比工具。需要 numpy；板子所在主机的 Python 3 或开发机都能跑。

用法：
  python3 compare_bf16.py ACTUAL.bin EXPECTED.bin
  python3 compare_bf16.py ACTUAL.bin EXPECTED.bin --actual-shape 84,84,64 --crop 2:82,2:82
  python3 compare_bf16.py ACTUAL.bin EXPECTED.bin --expected-shape 80,80,64

输出：元素数、逐位相同比例、bf16 ULP 距离（最大/平均）、f32 域的 max/mean |diff|、余弦相似度、结论。
退出码始终 0（便于在脚本里继续跑）；按输出自行判定阈值。
"""
import argparse
import sys

import numpy as np


def load_bf16(path):
    a = np.fromfile(path, dtype=np.uint16)
    return a


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def parse_shape(s):
    return tuple(int(x) for x in s.split(","))


def parse_crop(s):
    parts = s.split(",")
    out = []
    for p in parts:
        a, b = p.split(":")
        out.append((int(a), int(b)))
    return out


def bf16_ulp(v):
    """近似 bf16 ULP：normal: 2^(floor(log2|v|)-7)，subnormal/0: 2^-133。"""
    a = np.abs(v.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.floor(np.log2(a))
    ulp = np.power(2.0, e - 7.0)
    ulp = np.where(a > 0, ulp, 2.0 ** -133)
    return np.maximum(ulp, 2.0 ** -133)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actual")
    ap.add_argument("expected")
    ap.add_argument("--actual-shape", help="H,W,C（先把 actual 展平成这个形状）")
    ap.add_argument("--expected-shape", help="H,W,C")
    ap.add_argument("--crop", help="对 actual 裁剪（别名，同 --actual-crop）：y0:y1,x0:x1")
    ap.add_argument("--actual-crop", help="对 actual 裁剪：y0:y1,x0:x1（在 --actual-shape 之后）")
    ap.add_argument("--expected-crop", help="对 expected 裁剪：y0:y1,x0:x1（在 --expected-shape 之后）")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    a_bits = load_bf16(args.actual)
    e_bits = load_bf16(args.expected)
    if args.actual_shape:
        a_bits = a_bits.reshape(parse_shape(args.actual_shape))
    a_crop = args.actual_crop or args.crop
    if a_crop and args.actual_shape:
        (y0, y1), (x0, x1) = parse_crop(a_crop)
        a_bits = a_bits[y0:y1, x0:x1, :]
    if args.expected_shape:
        e_bits = e_bits.reshape(parse_shape(args.expected_shape))
    if args.expected_crop and args.expected_shape:
        (y0, y1), (x0, x1) = parse_crop(args.expected_crop)
        e_bits = e_bits[y0:y1, x0:x1, :]

    a_bits = a_bits.reshape(-1)
    e_bits = e_bits.reshape(-1)
    if a_bits.size != e_bits.size:
        print(f"SIZE MISMATCH: actual {a_bits.size} elems vs expected {e_bits.size} elems")
        print(f"  actual file {args.actual}: {a_bits.size*2} B, expected {args.expected}: {e_bits.size*2} B")
        sys.exit(0)

    a = bf16_to_f32(a_bits)
    e = bf16_to_f32(e_bits)
    same = a_bits == e_bits
    d = np.abs(a.astype(np.float64) - e.astype(np.float64))
    ulp = bf16_ulp(e)
    ulp_d = d / ulp
    denom = np.linalg.norm(a.astype(np.float64)) * np.linalg.norm(e.astype(np.float64))
    cos = float(np.dot(a.astype(np.float64), e.astype(np.float64)) / denom) if denom > 0 else float("nan")

    label = f"[{args.label}] " if args.label else ""
    print(f"{label}elements        : {a_bits.size}")
    print(f"{label}exact matches   : {same.sum()} / {a_bits.size}  ({100.0*same.mean():.6f}%)")
    print(f"{label}diff elements   : {(~same).sum()}")
    print(f"{label}max |diff| f32  : {d.max():.6e}")
    print(f"{label}mean|diff| f32  : {d.mean():.6e}")
    print(f"{label}max diff (bf16 ULP, approx): {ulp_d.max():.3f}")
    print(f"{label}mean diff (bf16 ULP, approx): {ulp_d.mean():.3f}")
    print(f"{label}cosine          : {cos:.9f}")
    if same.all():
        print(f"{label}VERDICT: EXACT（逐位相同）")
    elif same.mean() >= 0.999 and ulp_d.max() <= 2.0:
        print(f"{label}VERDICT: NEAR（少量 ≤2 ULP 差异；检查是否 f32 rnu/RNE 或近似函数导致）")
    else:
        print(f"{label}VERDICT: DIFF（不可接受，请按文档排错）")


if __name__ == "__main__":
    main()
