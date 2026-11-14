#!/usr/bin/env python3
"""按 gen.py 的 params.json 把 SD1.5 VAE 的 fp32 safetensors 权重打成参数块。

用法：python3 pack.py <params.json> <vae_fp32.safetensors> <outdir>
产出：<outdir>/param_blk0.bin、<outdir>/pack_report.json
自校验：从参数块读回每个张量，和源张量（bf16 舍入后）逐字节比对。
"""
import hashlib
import json
import os
import sys

import numpy as np

BLK_ALIGN = 4096


# ---------------------------------------------------------------- safetensors

def load_safetensors(path):
    with open(path, 'rb') as f:
        n = int.from_bytes(f.read(8), 'little')
        hdr = json.loads(f.read(n))
        raw = f.read()
    out = {}
    for k, v in hdr.items():
        if k == '__metadata__':
            continue
        a, b = v['data_offsets']
        dt = v['dtype']
        if dt == 'F32':
            arr = np.frombuffer(raw[a:b], dtype='<f4').reshape(v['shape']).astype(np.float32)
        elif dt == 'F16':
            arr = np.frombuffer(raw[a:b], dtype='<f2').reshape(v['shape']).astype(np.float32)
        else:
            raise RuntimeError(f'unsupported dtype {dt} for {k}')
        out[k] = arr
    return out


def to_bf16_bits(a):
    """f32 ⇒ bf16 位型（就近舍入到偶数），返回 uint16 数组。"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return r


def pad_channels(a, ci, co):
    """[co_act][ci_act]（可再带 k）⇒ 零填充到 [co][ci]。"""
    out = np.zeros((co, ci) + a.shape[2:], dtype=np.float32)
    out[:a.shape[0], :a.shape[1]] = a
    return out


def pack_conv_weight(W, co, ci, k):
    """torch [co_act, ci_act, k, k]（或 2 维 Linear）⇒ bf16 位型 [k*k][ci][co]，C 序。"""
    if W.ndim == 2:
        W = W.reshape(W.shape[0], W.shape[1], 1, 1)
    co_act, ci_act = W.shape[0], W.shape[1]
    assert W.shape[2] == k and W.shape[3] == k, (W.shape, k)
    assert co_act <= co and ci_act <= ci
    flat = np.zeros((k * k, ci, co), dtype=np.float32)
    for ky in range(k):
        for kx in range(k):
            flat[ky * k + kx, :ci_act, :co_act] = W[:, :, ky, kx].T
    return to_bf16_bits(flat.reshape(-1))


def pack_conv_bias(b, co):
    out = np.zeros(co, dtype=np.float32)
    out[:b.shape[0]] = b
    return out.astype('<f4')


def pack_gn_one(v, c):
    """gamma/beta [c] ⇒ f32 [c][16]（每行同一个值写 16 份，一行 64 B，自定义 GN kernel 用）。"""
    v = np.asarray(v, dtype=np.float32)
    assert v.size == c, (v.size, c)
    return np.repeat(v[:, None], 16, axis=1).astype('<f4')


def put(blk, off, arr):
    raw = np.frombuffer(arr.tobytes(), dtype=np.uint8)
    blk[off:off + raw.size] = raw


# ---------------------------------------------------------------- main

def main():
    params_path, st_path, outdir = sys.argv[1:4]
    with open(params_path) as f:
        meta = json.load(f)
    blk_bytes = meta['param_bytes']
    src = load_safetensors(st_path)
    blk = np.zeros(blk_bytes, dtype=np.uint8)
    report = dict(param_bytes=blk_bytes, tensors=[])
    checks = []          # (name, offset, kind, expected_bytes)

    for p in meta['params']:
        name, kind, off = p['name'], p['kind'], p['offset']
        if kind == 'conv_weight':
            W = src[p['src'] + '.weight']
            arr = pack_conv_weight(W, p['co'], p['ci'], p['k'])
            put(blk, off, arr)
            checks.append((name, off, arr.nbytes, arr.tobytes()))
        elif kind == 'conv_bias':
            b = src[p['src'] + '.bias']
            arr = pack_conv_bias(b, p['co'])
            put(blk, off, arr)
            checks.append((name, off, arr.nbytes, arr.tobytes()))
        elif kind == 'fused_conv_in_weight':
            Wpq = src['post_quant_conv.weight'][:, :, 0, 0]     # [4][4]
            Win = src['decoder.conv_in.weight']                  # [512][4][3][3]
            Weff = np.einsum('cjky,ji->ciky', Win.astype(np.float64),
                             Wpq.astype(np.float64)).astype(np.float32)
            arr = pack_conv_weight(Weff, p['co'], p['ci'], p['k'])
            put(blk, off, arr)
            checks.append((name, off, arr.nbytes, arr.tobytes()))
        elif kind == 'fused_conv_in_bias':
            bpq = src['post_quant_conv.bias']                    # [4]
            Win = src['decoder.conv_in.weight']
            bin_ = src['decoder.conv_in.bias']
            beff = bin_.astype(np.float64) + np.einsum('cjky,j->c',
                                                       Win.astype(np.float64),
                                                       bpq.astype(np.float64))
            arr = pack_conv_bias(beff.astype(np.float32), p['co'])
            put(blk, off, arr)
            checks.append((name, off, arr.nbytes, arr.tobytes()))
        elif kind in ('gn_gamma', 'gn_beta'):
            key = p['src'] + ('.weight' if kind == 'gn_gamma' else '.bias')
            arr = pack_gn_one(src[key], p['c'])
            put(blk, off, arr)
            checks.append((name, off, arr.nbytes, arr.tobytes()))
        else:
            raise RuntimeError('unknown kind ' + kind)
        report['tensors'].append(dict(name=name, kind=kind, offset=off, size=p['size']))

    # 自校验：参数块读回 == 期望
    bad = []
    for (name, off, nbytes, expect) in checks:
        got = blk[off:off + nbytes].tobytes()
        if got != expect:
            bad.append(name)
    report['checks'] = dict(total=len(checks), bad=bad)
    report['sha256'] = hashlib.sha256(blk.tobytes()).hexdigest()

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'param_blk0.bin'), 'wb') as f:
        f.write(blk.tobytes())
    with open(os.path.join(outdir, 'pack_report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(f'packed {len(checks)} tensors into {blk_bytes} B '
          f'({blk_bytes/2**20:.2f} MiB), sha256={report["sha256"][:16]}…, bad={len(bad)}')
    if bad:
        print('BAD:', bad[:10])
        sys.exit(1)


if __name__ == '__main__':
    main()
