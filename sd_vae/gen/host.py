#!/usr/bin/env python3
"""SD VAE 板上跑图的主机侧小工具（不依赖 torch，numpy + PIL 即可）。

子命令：
  input   <latent.bin> <out_in_x.bin>            latent（f32 [4,64,64]）⇒ stage A 的 in_x（border bf16 [68][68][16]）
  image   <out_y_raw.bin> <out.npy> [out.png]    stage C 的 out_y（border bf16 [516][516][16]）⇒ RGB 图像
  stage   <raw.bin> <h> <p> [ref.npy]            任一中间段 out_y（border bf16）⇒ 内部区统计/参考对比
  raw     <raw.bin>                              打印文件字节数与 sha256
  compare <img.npy> <ref_image.npy> [报告.json]  两张 [3,512,512] f32 图像比数值
"""
import hashlib
import json
import os
import sys

import numpy as np

B = 2


def to_bf16_bits(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16_bits(u):
    return ((np.asarray(u, dtype=np.uint32) << 16).view(np.float32)).copy()


def cmd_input(latent_path, out_path):
    z = np.fromfile(latent_path, dtype='<f4').reshape(4, 64, 64)
    buf = np.zeros((64 + 2 * B, 64 + 2 * B, 16), dtype=np.uint16)
    buf[B:B + 64, B:B + 64, :4] = to_bf16_bits(z.transpose(1, 2, 0))
    raw = buf.astype('<u2').tobytes()
    with open(out_path, 'wb') as f:
        f.write(raw)
    print(f'in_x: {len(raw)} B sha256={hashlib.sha256(raw).hexdigest()}')


def cmd_image(raw_path, npy_path, png_path=None):
    raw = open(raw_path, 'rb').read()
    h = 512
    need = (h + 2 * B) * (h + 2 * B) * 16 * 2
    assert len(raw) >= need, (len(raw), need)
    buf = np.frombuffer(raw[:need], dtype='<u2').reshape(h + 2 * B, h + 2 * B, 16)
    img = from_bf16_bits(buf[B:B + h, B:B + h, :3]).transpose(2, 0, 1)  # [3,512,512] f32 in [-1,1]
    np.save(npy_path, img)
    print('image: min=%.4f max=%.4f mean=%.4f std=%.4f' %
          (img.min(), img.max(), img.mean(), img.std()))
    if png_path:
        try:
            from PIL import Image
        except ImportError:
            print('png skipped: PIL not installed (npy saved)')
            return
        a = ((np.clip(img, -1, 1) + 1) * 127.5).round().astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(a).save(png_path)
        print('png:', png_path)


def cmd_compare(a_path, b_path, out_json=None):
    a = np.load(a_path).astype(np.float64)
    b = np.load(b_path).astype(np.float64)
    if a.ndim == 4:
        a = a[0]
    if b.ndim == 4:
        b = b[0]
    assert a.shape == b.shape, (a.shape, b.shape)
    d = np.abs(a - b)
    mse = float((d ** 2).mean())
    psnr = 10 * np.log10(4.0 / mse) if mse > 0 else 999.0
    res = dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
               rmse=float(np.sqrt(mse)), psnr=psnr, shape=list(a.shape))
    print(json.dumps(res, indent=1))
    if out_json:
        with open(out_json, 'w') as f:
            json.dump(res, f, indent=1)


def cmd_stage(raw_path, h, p, ref_path=None):
    raw = open(raw_path, 'rb').read()
    need = (h + 2 * B) * (h + 2 * B) * p * 2
    assert len(raw) >= need, (len(raw), need)
    buf = np.frombuffer(raw[:need], dtype='<u2').reshape(h + 2 * B, h + 2 * B, p)
    inner = buf[B:B + h, B:B + h, :].copy()
    vals = from_bf16_bits(inner)
    print(f'{raw_path}: raw={len(raw)} B  inner[{h}x{h}x{p}] bf16 '
          f'min={vals.min():.4f} max={vals.max():.4f} '
          f'mean={vals.mean():.4f} std={vals.std():.4f} '
          f'sha256_inner={hashlib.sha256(inner.astype("<u2").tobytes()).hexdigest()[:16]}')
    if ref_path and os.path.exists(ref_path):
        ref = np.load(ref_path).astype(np.float32)
        if ref.ndim == 4:
            ref = ref[0]
        ref_hwc = ref.reshape(ref.shape[0], h, h).transpose(1, 2, 0)
        ours = vals.astype(np.float64)
        d = np.abs(ours - ref_hwc.astype(np.float64))
        mse = float((d ** 2).mean())
        psnr = 10 * np.log10(4.0 / mse) if mse > 0 else 999.0
        print(f'  vs {ref_path}: max|d|={d.max():.5f} mean|d|={d.mean():.6f} PSNR={psnr:.2f} dB')


def cmd_raw(raw_path):
    raw = open(raw_path, 'rb').read()
    print(f'{raw_path}: {len(raw)} B sha256={hashlib.sha256(raw).hexdigest()}')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'input':
        cmd_input(sys.argv[2], sys.argv[3])
    elif cmd == 'image':
        cmd_image(sys.argv[2], sys.argv[3],
                  sys.argv[4] if len(sys.argv) > 4 else None)
    elif cmd == 'stage':
        cmd_stage(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                  sys.argv[5] if len(sys.argv) > 5 else None)
    elif cmd == 'raw':
        cmd_raw(sys.argv[2])
    elif cmd == 'compare':
        cmd_compare(sys.argv[2], sys.argv[3],
                    sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        raise SystemExit('usage: host.py input|image|compare ...')
