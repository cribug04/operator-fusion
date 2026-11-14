#!/usr/bin/env python3
"""SD1.5 VAE decoder 的「程序语义」黄金仿真（torch CPU）。

不是重新实现 VAE，而是**按 step.rs/plan.json 的算子顺序和布局**逐条执行我们生成的程序：
  - 权重从 param_blk0.bin 里按 gen.py 的偏移读回（bf16 反量化）
  - 激活按我们的 DDR 布局（border NHWC / NCHW / tight）存，bf16 舍入按 kernel 的存储点做
  - conv 用 torch conv2d（padding=1 等价于 border 环；通道补零）
用途：在没有板子/模拟器时，先把「打包 + 布局语义 + 层序」验到和 torch 参考一致。

用法：python3 emulate.py <params.json> <param_blk0.bin> <plan.json> <latent.bin> <ref_image.npy> <outdir>
产出：emulated_image.npy / .png、compare.json、每层的 dump（可选）
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count() or 8)

B = 2


def bf16(t):
    return t.to(torch.bfloat16).float()


class Emu:
    def __init__(self, params_path, blk_path, plan_path):
        with open(params_path) as f:
            meta = json.load(f)
        with open(plan_path) as f:
            self.plan = json.load(f)
        self.blk = np.memmap(blk_path, dtype=np.uint8, mode='r')
        self.tensors = {}
        self.params = {p['name']: p for p in meta['params']}
        self._wcache = {}

    # ---------- 参数块读取 ----------
    def load_bf16(self, off, count):
        u = np.frombuffer(self.blk[off:off + count * 2].tobytes(), dtype=np.uint16)
        f = (u.astype(np.uint32) << 16).view(np.float32).copy()
        return torch.from_numpy(f)

    def load_f32(self, off, count):
        a = np.frombuffer(self.blk[off:off + count * 4].tobytes(), dtype='<f4')
        return torch.from_numpy(a.copy())

    def conv_weight(self, woff, co, ci, k):
        key = (woff, co, ci, k)
        if key in self._wcache:
            return self._wcache[key]
        flat = self.load_bf16(woff, k * k * ci * co).reshape(k * k, ci, co)
        wt = flat.permute(2, 1, 0).reshape(co, ci, k, k).contiguous()
        self._wcache[key] = wt
        return wt

    def conv_bias(self, boff, co):
        return self.load_f32(boff, co)

    def gn_params(self, woff, boff, c, hw, cc):
        rows = c * hw // cc
        per = hw // cc
        g = self.load_f32(woff, rows * 8).reshape(rows, 8)[:, 0].reshape(c, per)[:, 0]
        b = self.load_f32(boff, rows * 8).reshape(rows, 8)[:, 0].reshape(c, per)[:, 0]
        return g.contiguous(), b.contiguous()

    # ---------- buffer 形状 ----------
    def buffer(self, name, shape):
        t = self.tensors.get(name)
        if t is None:
            t = torch.zeros(*shape, dtype=torch.float32)
            self.tensors[name] = t
        return t

    # ---------- 算子 ----------
    def conv(self, op):
        h, cin, cout, k, pin, pout, has_res = op['key'][1:]
        x = self.tensors[op['x']]
        w = self.conv_weight(op['w'], cout, cin, k)
        b = self.conv_bias(op['b'], cout)
        xi = x[B:B + h, B:B + h, :].permute(2, 0, 1).unsqueeze(0)   # [1,pin,h,h]
        y = F.conv2d(xi, w, b, padding=(k // 2))
        y = y[0].permute(1, 2, 0)                                   # [h,h,cout]
        y = bf16(y)
        if op.get('res'):
            r = self.tensors[op['res']]
            ri = r[B:B + h, B:B + h, :cout]
            y = bf16(y + ri)
        out = self.buffer(op['y'], (h + 2 * B, h + 2 * B, pout))
        out[B:B + h, B:B + h, :cout] = y
        return out

    def layout(self, op):
        d, h, c, p = op['key']
        x = self.tensors[op['x']]
        if d == 'nhwc2nchw':
            y = x[B:B + h, B:B + h, :c].permute(2, 0, 1).reshape(c, h * h).contiguous()
            self.tensors[op['y']] = y
        else:
            y = self.buffer(op['y'], (h + 2 * B, h + 2 * B, p))
            y[B:B + h, B:B + h, :c] = x.reshape(c, h, h).permute(1, 2, 0)
        return None

    def gn_stats(self, op):
        hw, c, g = op['key'][1], op['key'][2], op['key'][3]
        x = bf16(self.tensors[op['x']])                # [c, hw]
        cg = c // g
        xg = x.reshape(g, cg * hw)                     # 一组 = CG 通道 × HW 像素
        mean_g = xg.mean(dim=1)
        ex2_g = (xg * xg).mean(dim=1)
        var_g = ex2_g - mean_g * mean_g
        rstd_g = 1.0 / torch.sqrt(var_g + 1e-6)
        mean = mean_g.repeat_interleave(cg)
        rstd = rstd_g.repeat_interleave(cg)
        st = torch.zeros(2 * c, 16)
        st[:c, :] = mean[:, None]
        st[c:, :] = rstd[:, None]
        self.tensors[op['y']] = st

    def gn_apply(self, op):
        hw, c, act = op['key'][1], op['key'][2], op['key'][3]
        x = bf16(self.tensors[op['x']])
        st = self.tensors[op['s']]
        gamma = self.load_f32(op['g'], c * 16).reshape(c, 16)[:, 0]
        beta = self.load_f32(op['b'], c * 16).reshape(c, 16)[:, 0]
        mean = st[:c, 0]
        rstd = st[c:, 0]
        y = bf16((x - mean[:, None]) * rstd[:, None] * gamma[:, None] + beta[:, None])
        if act == 'silu':
            y = bf16(y * torch.sigmoid(y))
        self.tensors[op['x']] = y

    def up(self, op):
        _, h, c, pin, pout = op['key']
        x = self.tensors[op['x']]
        out = self.buffer(op['y'], (2 * h + 2 * B, 2 * h + 2 * B, pout))
        # 指针偏移在 step.rs 里由 xoff/yoff 给出：这里按通道窗口搬
        xoff = op.get('xoff') or 0
        yoff = op.get('yoff') or 0
        cch = c
        xi = x[B:B + h, B:B + h, xoff // 2:xoff // 2 + cch]
        up = xi.repeat_interleave(2, 0).repeat_interleave(2, 1)
        out[B:B + 2 * h, B:B + 2 * h, yoff // 2:yoff // 2 + cch] = up
        return out

    def crop(self, op):
        _, h, pin, pout = op['key']
        x = self.tensors[op['x']]
        y = x[B:B + h, B:B + h, :pout].reshape(h * h, pout).contiguous()
        self.tensors[op['y']] = y
        return None

    def pad(self, op):
        _, h, pin, pout = op['key']
        x = self.tensors[op['x']]
        out = self.buffer(op['y'], (h + 2 * B, h + 2 * B, pin))
        out[B:B + h, B:B + h, :pout] = x.reshape(h, h, pout)
        return out

    def transpose(self, op):
        r, c = op['key'][1], op['key'][2]
        x = self.tensors[op['x']]
        self.tensors[op['y']] = x.transpose(0, 1).contiguous()
        return None

    def attn(self, op):
        h, c = op['key'][1], op['key'][2]
        q = self.tensors[op['q']]
        kt = self.tensors[op['kt']]
        v = self.tensors[op['v']]
        scale = 1.0 / math.sqrt(c)
        qb = bf16(q)
        kb = bf16(kt)
        vb = bf16(v)
        s = bf16(torch.matmul(qb, kb))
        s = bf16(s * scale)
        m = s.max(dim=1, keepdim=True).values
        p = bf16(torch.exp(s - m))
        l = p.sum(dim=1, keepdim=True)                     # f32 累加
        out = bf16(torch.matmul(p, vb) / l)
        self.tensors[op['o']] = out
        return None

    # ---------- 主循环 ----------
    def run_ops(self):
        """按 plan.json 的算子顺序执行；输入缓冲 in_x 由调用方先塞进 self.tensors。"""
        n = 0
        for op in self.plan['ops']:
            kind = op['kind']
            if kind == 'conv':
                self.conv(op)
            elif kind == 'layout':
                self.layout(op)
            elif kind == 'gn_stats':
                self.gn_stats(op)
            elif kind == 'gn_apply':
                self.gn_apply(op)
            elif kind == 'up':
                self.up(op)
            elif kind == 'crop':
                self.crop(op)
            elif kind == 'pad':
                self.pad(op)
            elif kind == 'transpose':
                self.transpose(op)
            elif kind == 'attn':
                self.attn(op)
            else:
                raise RuntimeError('unknown op ' + kind)
            n += 1
            if n % 20 == 0:
                print(f'  {n}/{len(self.plan["ops"])} ops', flush=True)
        last = self.plan['ops'][-1]
        return self.tensors[last.get('y') or last.get('o')]


def bf16_np(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(torch.bfloat16).float().numpy()


def save_png(img_chw, path):
    from PIL import Image
    a = ((np.clip(img_chw, -1, 1) + 1) * 127.5).round().astype(np.uint8)
    Image.fromarray(a.transpose(1, 2, 0)).save(path)


def main():
    build_root, blk_root, latent_path, ref_path, outdir = sys.argv[1:6]
    os.makedirs(outdir, exist_ok=True)

    def border_from_chw(chw, h, p):
        chw = np.asarray(chw, dtype=np.float32)
        c = chw.shape[0]
        hwc = chw.transpose(1, 2, 0)
        x = torch.zeros(h + 2 * B, h + 2 * B, p, dtype=torch.float32)
        x[B:B + h, B:B + h, :c] = bf16(torch.from_numpy(hwc))
        return x

    z = np.fromfile(latent_path, dtype='<f4').reshape(4, 64, 64)
    cur = border_from_chw(z, 64, 16)
    for stage in ('a', 'b1', 'b2a', 'b2b', 'c'):
        print(f'== stage {stage} ==', flush=True)
        emu = Emu(os.path.join(build_root, stage, 'params.json'),
                  os.path.join(blk_root.format(stage=stage), 'param_blk0.bin'),
                  os.path.join(build_root, stage, 'plan.json'))
        emu.tensors['in_x'] = cur
        cur = emu.run_ops()
        # 每段边界存成 NCHW npy（和 torch dump 同布局），便于逐段比
        info = {'a': (64, 512), 'b1': (128, 512), 'b2a': (128, 512),
                'b2b': (256, 512), 'c': (512, 3)}[stage]
        hh, cc = info
        m = cur[B:B + hh, B:B + hh, :cc].permute(2, 0, 1).contiguous().numpy()
        np.save(os.path.join(outdir, f'stage_{stage}.npy'), m)
    h = 512
    img = cur[B:B + h, B:B + h, :3].permute(2, 0, 1).contiguous().numpy()
    np.save(os.path.join(outdir, 'emulated_image.npy'), img)
    save_png(img, os.path.join(outdir, 'emulated_image.png'))
    print('emulated image stats: min=%.4f max=%.4f mean=%.4f std=%.4f' %
          (img.min(), img.max(), img.mean(), img.std()))
    if os.path.exists(ref_path):
        if ref_path.endswith('.npy'):
            ref = np.load(ref_path).astype(np.float32)
        else:
            ref = np.fromfile(ref_path, dtype='<f4').reshape(3, 512, 512)
        if ref.ndim == 4:
            ref = ref[0]
        if ref.shape[0] != 3 and ref.shape[-1] == 3:
            ref = ref.transpose(2, 0, 1)
        ref = ref.reshape(3, 512, 512)
        d = np.abs(img.astype(np.float64) - ref.astype(np.float64))
        mse = float((d ** 2).mean())
        psnr = 10 * math.log10(4.0 / mse) if mse > 0 else 999.0
        res = dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                   rmse=float(math.sqrt(mse)), psnr=psnr)
        with open(os.path.join(outdir, 'compare.json'), 'w') as f:
            json.dump(res, f, indent=1)
        print('vs torch reference: max|d|=%.5f mean|d|=%.6f PSNR=%.2f dB' %
              (res['max_abs'], res['mean_abs'], res['psnr']))
    print('wrote', outdir)


if __name__ == '__main__':
    main()
