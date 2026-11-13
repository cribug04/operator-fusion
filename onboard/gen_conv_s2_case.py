#!/usr/bin/env python3
"""为 conv2d_s2_fast 原型生成真机测试数据（在开发机的普通 Python 3 + numpy 上运行）。

产物（默认写到本文件所在目录的 conv_s2_case/）：
  input_164x164x64_bf16.bin     = in_x 的完整内容（3,441,152 B）：padded NHWC [164][164][64]，边框 0
  param_blk0.bin                = @w 的完整内容（73,984 B）：[9][64][64] bf16 权重 + [64] f32 bias
  expected_80x80x64_bf16.bin    = CPU 参考（80×80×64，819,200 B）：conv(f32 累加, 近似) + bias + bf16 舍入 + SiLU
  meta.json                     = 形状/种子/容差说明

注意：expected 是**宽松参考**（f64 累加→f32→bf16→f32 SiLU→bf16），与板子不会逐位一致；
      严格对拍请用同形状慢路 `conv2d!`（examples/conv2d_s2_ref）在同一块板上对比，见 REAL_DEVICE_VERIFICATION.md。
"""
import json
import os
import numpy as np

F32 = np.float32
H, W, CI, CO = 80, 80, 64, 64
B = 2
HI, WI = H * 2 + 2 * B, W * 2 + 2 * B  # 164, 164
OUT_H, OUT_W = H, W                     # stride=2、k=3、pad=1 ⇒ 输出 80×80
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conv_s2_case")
SEED = 20261113 + 2


def to_bf16_bits(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).copy()
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    return ((u + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_f32(b):
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # 输入：raw [160][160][64] 随机（落 bf16 网格），放进 padded [164][164][64]（边框 0）
    raw = rng.uniform(-1.0, 1.0, size=(H * 2, W * 2, CI)).astype(np.float32)
    raw_b = to_bf16_bits(raw)
    xpad = np.zeros((HI, WI, CI), dtype=np.float32)
    xpad[B:B + H * 2, B:B + W * 2, :] = bf16_bits_to_f32(raw_b)

    # 权重 [9][CI][CO] bf16（对应 conv2d 的 [抽头][Cin][Cout]），bias f32[CO]
    w = rng.uniform(-0.06, 0.06, size=(9, CI, CO)).astype(np.float32)
    wb = to_bf16_bits(w)
    wf = bf16_bits_to_f32(wb)
    bias = rng.uniform(-0.01, 0.01, size=(CO,)).astype(np.float32)

    # CPU 参考：conv(pad=1, stride=2) + bias，按 f64 累加 → f32 → bf16 → f32 SiLU → bf16
    acc = np.zeros((OUT_H, OUT_W, CO), dtype=np.float64)
    x64 = xpad.astype(np.float64)
    for ky in range(3):
        for kx in range(3):
            # 输出 (oy,ox) 的输入 padded 像素 = (oy*2+ky, ox*2+kx)
            patch = x64[ky:ky + OUT_H * 2:2, kx:kx + OUT_W * 2:2, :].reshape(OUT_H, OUT_W, CI)
            acc += np.einsum("hwc,cd->hwd", patch, wf[ky * 3 + kx].astype(np.float64), optimize=True)
    acc_f32 = (acc + bias.astype(np.float64)[None, None, :]).astype(np.float32)
    y_bf = bf16_bits_to_f32(to_bf16_bits(acc_f32))      # 舍 bf16（kernel 在 silu 前舍一次）
    y_silu = (y_bf / (np.float32(1.0) + np.exp(-y_bf, dtype=np.float32))).astype(np.float32)
    out_bits = to_bf16_bits(y_silu)

    # 文件
    x_out = to_bf16_bits(xpad)                          # padded 全图（边框 0 也在内）
    x_out.astype(np.uint16).tofile(os.path.join(OUT_DIR, "input_164x164x64_bf16.bin"))
    with open(os.path.join(OUT_DIR, "param_blk0.bin"), "wb") as f:
        wb.reshape(-1).astype("<u2").tofile(f)      # [9][64][64] bf16 小端
        bias.astype("<f4").tofile(f)                 # f32 bias 小端（@w 偏移 73728）
    out_bits.astype(np.uint16).tofile(os.path.join(OUT_DIR, "expected_80x80x64_bf16.bin"))
    meta = {
        "op": "conv2d_s2_fast probe (k=3, s=2, pad=1, silu)",
        "seed": SEED,
        "input": {"file": "input_164x164x64_bf16.bin", "shape": [HI, WI, CI], "dtype": "bf16", "bytes": int(HI * WI * CI * 2)},
        "params": {"file": "param_blk0.bin", "weights_shape": [9, CI, CO], "bias": CO, "bytes": int(9 * CI * CO * 2 + CO * 4)},
        "expected": {"file": "expected_80x80x64_bf16.bin", "shape": [OUT_H, OUT_W, CO], "dtype": "bf16", "bytes": int(OUT_H * OUT_W * CO * 2)},
        "board_output": {"decl": "out_y", "shape": [84, 84, CO], "bytes": int((W + 2 * B) * (W + 2 * B) * CO * 2),
                          "compare_crop_y": [2, 82], "compare_crop_x": [2, 82]},
        "tolerance": {"strict": "与 examples/conv2d_s2_ref（同形状慢路）逐位相同",
                       "loose_cpu": "cosine >= 0.9999 且 max|diff| <= 4 bf16 ULP（f32 累加次序 + SiLU 近似）"},
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    import hashlib
    print("wrote to", OUT_DIR)
    for fn in ("input_164x164x64_bf16.bin", "param_blk0.bin", "expected_80x80x64_bf16.bin", "meta.json"):
        p = os.path.join(OUT_DIR, fn)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        print(f"  {fn:32s} {os.path.getsize(p):9d} B  sha256 {h[:16]}…")
    print("expected: 80×80×64 bf16  (板子输出 out_y 是 84×84×64，比较时裁掉外圈 2 像素：--crop 2:82,2:82)")


if __name__ == "__main__":
    main()
