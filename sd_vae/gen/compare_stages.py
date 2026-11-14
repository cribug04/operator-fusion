#!/usr/bin/env python3
"""把 emulate.py 的逐段输出和 torch 逐层 dump 对上：
   a   ⇔ decoder.mid_block.resnets.1
   b1  ⇔ decoder.up_blocks.0
   b2a ⇔ decoder.up_blocks.1
   b2b ⇔ decoder.up_blocks.1.upsamplers.0
   c   ⇔ decoder.conv_out
用法：python3 compare_stages.py <emu_out_dir> [dump_dir]
"""
import json
import math
import os
import sys

import numpy as np

EMU = {
    'a':   ('dump_026_decoder_mid_block_resnets_1_out0.npy', 'mid_block.resnets.1'),
    'b1':  ('dump_055_decoder_up_blocks_0_out0.npy', 'up_blocks.0'),
    'b2a': ('dump_079_decoder_up_blocks_1_resnets_2_out0.npy', 'up_blocks.1.resnets.2'),
    'b2b': ('dump_083_decoder_up_blocks_1_out0.npy', 'up_blocks.1（含 upsampler）'),
    'c':   ('dump_141_decoder_conv_out_out0.npy', 'conv_out'),
}


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else '/root/sd_vae/emu_out'
    dumpdir = sys.argv[2] if len(sys.argv) > 2 else '/root/sd_vae/ref'
    res = {}
    print(f'{"stage":5s} {"shape":18s} {"max|d|":>10s} {"mean|d|":>10s} {"PSNR":>8s}  module')
    for stage, (dfile, mod) in EMU.items():
        ours_path = os.path.join(outdir, f'stage_{stage}.npy')
        dpath = os.path.join(dumpdir, dfile)
        if not (os.path.exists(ours_path) and os.path.exists(dpath)):
            print(f'{stage:5s} SKIP (missing)')
            continue
        ours = np.load(ours_path).astype(np.float64)
        ref = np.load(dpath).astype(np.float64)
        if ref.ndim == 4:
            ref = ref[0]
        d = np.abs(ours - ref)
        mse = float((d ** 2).mean())
        psnr = 10 * math.log10(4.0 / mse) if mse > 0 else 999.0
        res[stage] = dict(module=mod, max_abs=float(d.max()), mean_abs=float(d.mean()),
                          rmse=float(math.sqrt(mse)), psnr=psnr,
                          ref_min=float(ref.min()), ref_max=float(ref.max()),
                          ref_mean=float(ref.mean()), ref_std=float(ref.std()))
        print(f'{stage:5s} {str(ours.shape):18s} {d.max():10.5f} {d.mean():10.6f} '
              f'{psnr:8.2f}  {mod}')
    with open(os.path.join(outdir, 'stage_compare.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print('wrote', os.path.join(outdir, 'stage_compare.json'))


if __name__ == '__main__':
    main()
