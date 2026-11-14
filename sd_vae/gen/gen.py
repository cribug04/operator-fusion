#!/usr/bin/env python3
"""SD1.5 VAE decoder（全 NPU）整步程序生成器。

输出：
  <outdir>/kernels.rs   kernel 特化（calBLAS 模板 + 手写搬运/注意力 kernel）
  <outdir>/step.rs      整步程序（npu::decls! + #[npu::program]）
  <outdir>/params.json  参数块布局（给 pack.py 用）
  <outdir>/plan.json    缓冲/层表记录

用法：python3 gen.py <outdir>
"""
import json
import os
import sys

B = 2              # 视觉模板边框宽
EPS = 1.0e-6       # VAE GroupNorm eps
ALIGN = 4096       # 参数字节对齐
FILL_BLK = 16384   # 清零一块的字节数
FP32 = 4

# ---------------------------------------------------------------- layout 数学

def align_up(x, a=ALIGN):
    return (x + a - 1) // a * a


def img_size(h, p):
    """border NHWC 图 [h+4][h+4][p] 的字节数（h=w），补到 16 KiB。"""
    return align_up((h + 2 * B) * (h + 2 * B) * p * 2, FILL_BLK)


def norm_cols(n, hw):
    cc = hw if hw < 128 else 128
    while cc > 16 and (hw % cc != 0 or (n // cc) % 8 != 0):
        cc //= 2
    return cc


def norm_parts(rows, cols):
    parts = 1
    while rows * cols * 10 // parts > 256 * 1024 and (rows // (parts * 2)) % 8 == 0:
        parts *= 2
    return parts


def gn_geom(c, hw, g):
    cg = c // g
    n = cg * hw
    cc = norm_cols(n, hw)
    r = n // cc
    nh = norm_parts(r, cc)
    rp = r // nh
    assert c % g == 0 and hw % cc == 0 and rp % 8 == 0, (c, hw, g, cc, r, nh, rp)
    return dict(cg=cg, cc=cc, r=r, nh=nh, rp=rp)


# ---------------------------------------------------------------- plan

class Plan:
    def __init__(self):
        self.bufs = {}
        self.order = []
        self.ops = []
        self.params = []
        self.seq = 0
        self.act_bytes = 0
        self.param_bytes = 0
        self.fills = []
        self.slots_info = []

    def new_buf(self, tag, klass, size):
        self.seq += 1
        name = f'{tag}_{self.seq}'
        self.bufs[name] = dict(tag=tag, klass=klass, size=size)
        self.order.append(name)
        return name

    def new_decl(self, name, kind, size):
        """主机进/出的静态存储（in_x / out_y）：不参与 @act 分配。"""
        self.bufs[name] = dict(tag=name, klass=(kind,), size=size, decl=name)
        self.order.append(name)
        return name

    def img(self, tag, h, p):
        return self.new_buf(tag, ('img', h, p), img_size(h, p))

    def tight(self, tag, nbytes):
        return self.new_buf(tag, ('tight',), align_up(nbytes, FILL_BLK))

    def op(self, kind, **kw):
        kw['kind'] = kind
        self.ops.append(kw)
        return kw

    def param(self, name, kind, size, **kw):
        p = dict(name=name, kind=kind, size=size, offset=None, **kw)
        self.params.append(p)
        return p


# ---------------------------------------------------------------- 模型结构

def build(stage='a'):
    P = Plan()
    C0 = P.new_buf('nchw_scratch', ('nchw',), 0)   # 共享 NCHW 暂存

    # ---------- 通用小块 ----------
    def conv(tag, h, cin, cout, k, x, y, wsrc, res=None, pin=None, pout=None):
        pin = cin if pin is None else pin
        pout = cout if pout is None else pout
        wp = P.param(tag + '.w', 'conv_weight', cout * cin * k * k * 2,
                     src=wsrc, co=cout, ci=cin, k=k)
        bp = P.param(tag + '.b', 'conv_bias', cout * FP32, src=wsrc, co=cout)
        key = ('conv', h, cin, cout, k, pin, pout, res is not None)
        return P.op('conv', key=key, x=x, y=y, res=res, w=wp, b=bp, tag=tag)

    def layout(direction, h, c, p, x, y):
        return P.op('layout', key=(direction, h, c, p), x=x, y=y)

    def gn(tag, h, c, g, x, act):
        """GroupNorm：先 gn_stats 求每通道 mean/rstd，再 gn_apply 归一+仿射(+silu)。
        不用 calblas::group_norm!：hw=16384,c=512 那组形状会让 docc 生成 ~1 MiB 代码
        （riscvc 链接器 21 位分支够不着）。自定义 kernel 的段数走运行期标量，代码恒定。"""
        hw = h * h
        st = P.new_buf('gnstats', ('gstat',), align_up(2 * c * 16 * FP32, ALIGN))
        gp = P.param(tag + '.gamma', 'gn_gamma', c * 16 * FP32, src=tag, c=c)
        bp = P.param(tag + '.beta', 'gn_beta', c * 16 * FP32, src=tag, c=c)
        nchunk = hw // 256
        P.op('gn_stats', key=('gn_stats', hw, c, g), x=x, y=st, nchunk=nchunk, tag=tag)
        return P.op('gn_apply', key=('gn_apply', hw, c, act), x=x, y=x,
                    g=gp, b=bp, s=st, nchunk=nchunk, tag=tag)

    def resnet(tag, h, cin, cout, x, out=None):
        if cin != cout:
            s = P.img('shortcut', h, cout)
            conv(tag + '.conv_shortcut', h, cin, cout, 1, x, s, tag + '.conv_shortcut')
        else:
            s = x
        layout('nhwc2nchw', h, cin, cin, x, C0)
        gn(tag + '.norm1', h, cin, 32, C0, 'silu')
        t = P.img('n1_out', h, cin)
        layout('nchw2nhwc', h, cin, cin, C0, t)
        y = P.img('c1_out', h, cout)
        conv(tag + '.conv1', h, cin, cout, 3, t, y, tag + '.conv1')
        layout('nhwc2nchw', h, cout, cout, y, C0)
        gn(tag + '.norm2', h, cout, 32, C0, 'silu')
        t2 = P.img('n2_out', h, cout)
        layout('nchw2nhwc', h, cout, cout, C0, t2)
        z = out if out is not None else P.img('res_out', h, cout)
        conv(tag + '.conv2', h, cout, cout, 3, t2, z, tag + '.conv2',
             res=s, pin=cout, pout=cout)
        return z

    def upsample_conv(tag, h, c, x, out=None):
        up = P.img('up', 2 * h, c)
        if c <= 256:
            P.op('up', key=('up', h, c, c, c), x=x, y=up, xoff=0, yoff=0)
        else:
            half = c // 2
            P.op('up', key=('up', h, half, c, c), x=x, y=up, xoff=0, yoff=0)
            P.op('up', key=('up', h, half, c, c), x=x, y=up,
                 xoff=half * 2, yoff=half * 2)
        dst = out if out is not None else P.img('up_conv_out', 2 * h, c)
        conv(tag + '.conv', 2 * h, c, c, 3, up, dst, tag + '.conv')
        return dst

    def attention(tag, h, c, x):
        layout('nhwc2nchw', h, c, c, x, C0)
        gn(tag + '.group_norm', h, c, 32, C0, 'none')
        t = P.img('attn_gn_out', h, c)
        layout('nchw2nhwc', h, c, c, C0, t)
        q = P.tight('q', h * h * c * 2)
        k = P.tight('k', h * h * c * 2)
        v = P.tight('v', h * h * c * 2)
        kt = P.tight('kt', c * h * h * 2)
        ob = P.tight('attn_o', h * h * c * 2)
        qb = P.img('qkv_border', h, c)
        for name, dst in (('query', q), ('key', k), ('value', v)):
            conv(tag + '.' + name, h, c, c, 1, t, qb, tag + '.' + name)
            P.op('crop', key=('crop', h, c, c), x=qb, y=dst)
        P.op('transpose', key=('t2d', h * h, c), x=k, y=kt)
        P.op('attn', key=('attn', h, c), q=q, k=k, v=v, kt=kt, o=ob)
        ob_border = P.img('attn_o_border', h, c)
        P.op('pad', key=('pad', h, c, c), x=ob, y=ob_border)
        # diffusers 的 attention 有残差：out = proj(attn) + x（rescale=1.0）⇒ proj 的 1x1 conv 直接带 res
        out = P.img('attn_out', h, c)
        conv(tag + '.proj_attn', h, c, c, 1, ob_border, out, tag + '.proj_attn', res=x)
        return out

    # ---------- 分三段（整份程序的 CCU 代码超 1 MiB，链接器 21 位分支够不着）----------
    if stage == 'a':
        # z ⇒ conv_in ⇒ mid resnets + attention ⇒ 64² 512
        in_x = P.new_decl('in_x', 'input', img_size(64, 16))
        out_y = P.new_decl('out_y', 'output', img_size(64, 512))
        z = P.img('conv_in_out', 64, 512)
        wp = P.param('conv_in.w', 'fused_conv_in_weight', 512 * 16 * 9 * 2,
                     src='decoder.conv_in', co=512, ci=16, k=3)
        bp = P.param('conv_in.b', 'fused_conv_in_bias', 512 * FP32,
                     src='decoder.conv_in', co=512)
        P.op('conv', key=('conv', 64, 16, 512, 3, 16, 512, False), x=in_x, y=z,
             res=None, w=wp, b=bp, tag='conv_in')
        cur = resnet('decoder.mid_block.resnets.0', 64, 512, 512, z)
        cur = attention('decoder.mid_block.attentions.0', 64, 512, cur)
        cur = resnet('decoder.mid_block.resnets.1', 64, 512, 512, cur, out=out_y)
    elif stage == 'b1':
        # 64² 512 ⇒ up_blocks 0（64²）+ upsample ⇒ 128² 512
        in_x = P.new_decl('in_x', 'input', img_size(64, 512))
        out_y = P.new_decl('out_y', 'output', img_size(128, 512))
        cur = in_x
        for ri in range(3):
            cur = resnet(f'decoder.up_blocks.0.resnets.{ri}', 64, 512, 512, cur)
        cur = upsample_conv('decoder.up_blocks.0.upsamplers.0', 64, 512, cur, out=out_y)
    elif stage == 'b2a':
        # 128² 512 ⇒ up_blocks 1 的 3 个 resnet ⇒ 128² 512
        in_x = P.new_decl('in_x', 'input', img_size(128, 512))
        out_y = P.new_decl('out_y', 'output', img_size(128, 512))
        cur = in_x
        for ri in range(3):
            cur = resnet(f'decoder.up_blocks.1.resnets.{ri}', 128, 512, 512, cur, out=out_y if ri == 2 else None)
    elif stage == 'b2b':
        # 128² 512 ⇒ upsample2 + conv ⇒ 256² 512
        in_x = P.new_decl('in_x', 'input', img_size(128, 512))
        out_y = P.new_decl('out_y', 'output', img_size(256, 512))
        cur = upsample_conv('decoder.up_blocks.1.upsamplers.0', 128, 512, in_x, out=out_y)
    else:
        # 256² 512 ⇒ up_blocks 2（256²）+ upsample ⇒ up_blocks 3（512²）+ norm_out + conv_out
        in_x = P.new_decl('in_x', 'input', img_size(256, 512))
        out_y = P.new_decl('out_y', 'output', img_size(512, 16))
        cur = resnet('decoder.up_blocks.2.resnets.0', 256, 512, 256, in_x)
        cur = resnet('decoder.up_blocks.2.resnets.1', 256, 256, 256, cur)
        cur = resnet('decoder.up_blocks.2.resnets.2', 256, 256, 256, cur)
        cur = upsample_conv('decoder.up_blocks.2.upsamplers.0', 256, 256, cur)
        cur = resnet('decoder.up_blocks.3.resnets.0', 512, 256, 128, cur)
        cur = resnet('decoder.up_blocks.3.resnets.1', 512, 128, 128, cur)
        cur = resnet('decoder.up_blocks.3.resnets.2', 512, 128, 128, cur)
        layout('nhwc2nchw', 512, 128, 128, cur, C0)
        gn('decoder.conv_norm_out', 512, 128, 32, C0, 'silu')
        t = P.img('norm_out', 512, 128)
        layout('nchw2nhwc', 512, 128, 128, C0, t)
        conv('decoder.conv_out', 512, 128, 16, 3, t, out_y, 'decoder.conv_out')

    finish(P)
    return P


# ---------------------------------------------------------------- finish：分配

def finish(P):
    birth = {n: 10 ** 9 for n in P.bufs}
    last = {n: -1 for n in P.bufs}
    for i, op in enumerate(P.ops):
        for key in ('y', 'o'):
            n = op.get(key)
            if n:
                birth[n] = min(birth[n], i)
                last[n] = max(last[n], i)
        for key in ('x', 'res', 'q', 'k', 'v', 'kt'):
            n = op.get(key)
            if n:
                last[n] = max(last[n], i)

    max_nchw = 0
    for op in P.ops:
        if op['kind'] == 'gn':
            max_nchw = max(max_nchw, op['key'][2] * op['key'][1] * 2)
    for n in P.bufs:
        b = P.bufs[n]
        if b['klass'][0] == 'nchw':
            b['size'] = max_nchw
    max_tight = max([P.bufs[n]['size'] for n in P.bufs
                     if P.bufs[n]['klass'][0] == 'tight'] + [0])
    for n in P.bufs:
        if P.bufs[n]['klass'][0] == 'tight':
            P.bufs[n]['size'] = max_tight

    for n in list(P.bufs):
        b = P.bufs[n]
        if b['klass'][0] in ('input', 'output'):
            continue
        b['birth'] = birth[n]
        b['last'] = last[n]

    classes = {}
    for n in P.order:
        b = P.bufs[n]
        if b['klass'][0] in ('input', 'output'):
            continue
        classes.setdefault(b['klass'], []).append(n)

    total = 0
    fills = []
    slots_info = []
    for klass in sorted(classes, key=lambda k: str(k)):
        names = sorted(classes[klass], key=lambda n: P.bufs[n]['birth'])
        slots = []
        for n in names:
            b = P.bufs[n]
            size = b['size']
            born = b['birth']
            choose = None
            for s in slots:
                if s['free_after'] < born and s['size'] >= size:
                    choose = s
                    break
            if choose is None:
                choose = dict(base=None, free_after=b['last'], size=size)
                slots.append(choose)
            else:
                choose['free_after'] = max(choose['free_after'], b['last'])
                choose['size'] = max(choose['size'], size)
            b['slot'] = choose
        for s in slots:
            s['base'] = total
            total = align_up(total + s['size'], ALIGN)
            slots_info.append(dict(klass=str(klass), base=s['base'], size=s['size']))
        for n in names:
            P.bufs[n]['offset'] = P.bufs[n]['slot']['base']
        if klass[0] == 'img':
            for s in slots:
                fills.append((s['base'], s['size'] // FILL_BLK))

    P.act_bytes = total
    P.fills = fills
    P.slots_info = slots_info

    wtotal = 0
    for p in P.params:
        p['offset'] = align_up(wtotal, ALIGN)
        wtotal = p['offset'] + align_up(p['size'], 64)
    P.param_bytes = align_up(wtotal, ALIGN)


# ---------------------------------------------------------------- 名字

def conv_name(key):
    h, cin, cout, k, pin, pout, has_res = key[1:]
    return f'k_conv_{h}_{cin}to{cout}_k{k}_p{pin}_{pout}' + ('_r' if has_res else '')


def gn_name(key):
    hw, c, g, act = key[1:]
    return f'k_gn_{hw}_{c}_{g}_{act}'


def gn_stats_name(key):
    _, hw, c, g = key
    return f'k_gns_{hw}_{c}_{g}'


def gn_apply_name(key):
    _, hw, c, act = key
    return f'k_gna_{hw}_{c}_{act}'


def layout_name(key):
    d, h, c, p = key[0], key[1], key[2], key[3]
    return f'k_{d}_{h}_{c}_p{p}'


def up_name(key):
    h, c, pin, pout = key[1:]
    return f'k_up_{h}_{c}_p{pin}_{pout}'


# ---------------------------------------------------------------- kernels.rs

CUSTOM_MOVE = r'''
// ---------------------------------------------------------------- 手写搬运 kernel
//
// border NHWC 图（[H+4][H+4][PIN]，内部 [H][H] 像素）与紧排 [H·H][POUT] 互搬。
// 一次搬「一行图像 = H 个像素 × 256 个通道」：视图 [像素, 通道]（行距 PIN/POUT），
// 每个像素的 256 个通道连续（512 B = 一行 VB）。
'''

CUSTOM_ATTN = r'''
// ---------------------------------------------------------------- 手写：单头 flash 注意力（VAE mid attention）
//
// q / k / v / o 都是紧排 [T, D] bf16；kT 是 [D, T]（transpose2d 转好）。
// 两遍：第一遍逐 key 块算分数取行最大；第二遍重算分数、exp、行和，V 也分块累加。
// 单头、无 mask、scale = 1/√D；query 块两核各一半。
'''


def emit_fill(name, blocks):
    return f'''
/// 清零 {blocks} 块 × 16 KiB（图像 slot 的边框一次清零）。
#[npu::kernel]
pub fn {name}(p: npu::Ptr<npu::Ddr, npu::bf16>) {{
    use npu::prelude::*;
    let t = vcache::<bf16, _>([32, 256]);
    t.fill(0.0f32);
    let g = p.bind_strided::<npu::Write, 3>([{blocks}, 32, 256], [32 * 256, 256, 1]);
    for i in 0..{blocks} {{
        t.store(g.sub([i, 0, 0], [1, 32, 256]));
    }}
}}
'''


def emit_move(name, crop, h, pin, pout):
    opname = 'crop' if crop else 'pad'
    cbool = 'true' if crop else 'false'
    return f'''
/// {opname}：h={h}, pin={pin}, pout={pout}（border ⇔ 紧排）
#[npu::kernel]
pub fn {name}(x: npu::Ptr<npu::Ddr, npu::bf16>, y: npu::Ptr<npu::Ddr, npu::bf16>) {{
    use npu::prelude::*;
    const H: usize = {h};
    const PIN: usize = {pin};
    const POUT: usize = {pout};
    const WI: usize = H + 4;
    const CH: usize = 256;
    const NC: usize = POUT / CH;
    let gx_border = x.add((2 * WI + 2) * PIN * 2).bind_strided::<npu::Read, 2>([H * WI, PIN], [PIN, 1]);
    let gy_border = y.add((2 * WI + 2) * PIN * 2).bind_strided::<npu::Write, 2>([H * WI, PIN], [PIN, 1]);
    let gx_tight = x.bind_strided::<npu::Read, 2>([H * H, POUT], [POUT, 1]);
    let gy_tight = y.bind_strided::<npu::Write, 2>([H * H, POUT], [POUT, 1]);
    let t = vcache::<bf16, _>([H, CH]);
    for r in 0..H {{
        for c in 0..NC {{
            if {cbool} {{
                t.load(gx_border.sub([r * WI, c * CH], [H, CH]));
                t.store(gy_tight.sub([r * H, c * CH], [H, CH]));
            }} else {{
                t.load(gx_tight.sub([r * H, c * CH], [H, CH]));
                t.store(gy_border.sub([r * WI, c * CH], [H, CH]));
            }}
        }}
    }}
}}
'''


def emit_gn_stats(hw, c, g):
    cg = c // g
    inv = 1.0 / (cg * hw)
    return f'''
/// GroupNorm 统计：x [C][HW] bf16 ⇒ stats [2C][16] f32（前 C 行 16 份 mean，后 C 行 16 份 rstd）。
/// 均值/方差在「组」上求：一组 CG = C/G 个通道 × HW 个像素（同组各通道共享同一 mean/rstd）。
#[npu::kernel(cores = 2, numerics = "ks1")]
pub fn k_gns_{hw}_{c}_{g}(x: npu::Ptr<npu::Ddr, npu::bf16>, stats: npu::Ptr<npu::Ddr, f32>,
                          nchunk: usize) {{
    use npu::prelude::*;
    const C: usize = {c};
    const HW: usize = {hw};
    const CH: usize = 256;
    const G: usize = {g};
    const CG: usize = C / G;
    const INV_N: f32 = {inv:.12e}f32;
    let gx = x.bind_strided::<npu::Read, 2>([C, HW], [HW, 1]);
    let gs = stats.bind_strided::<npu::Write, 2>([2 * C, 16], [16, 1]);
    let (bb, f) = (vcache::<bf16, _>([1, CH]), vcache::<f32, _>([1, CH]));
    let (sm, sq) = (vcache::<f32, _>([1, 1]), vcache::<f32, _>([1, 1]));
    let t = vcache::<f32, _>([1, CH]);
    let (s1, m2, v16) = (vcache::<f32, _>([1, 1]), vcache::<f32, _>([1, 1]),
                         vcache::<f32, _>([1, 16]));
    let (g0, gn) = calblas::halves!(G);
    for gi in 0..gn {{
        let g = g0 + gi;
        sm.fill(0.0f32);
        sq.fill(0.0f32);
        for ci in 0..CG {{
            let c = g * CG + ci;
            for j in 0..nchunk {{
                bb.load(gx.sub([c, j * CH], [1, CH]));
                f.cvt(bb, Cvt::default());
                s1.reduce(Combine::Sum, f, 1);
                sm.add(sm, s1);
                t.mul(f, f);
                s1.reduce(Combine::Sum, t, 1);
                sq.add(sq, s1);
            }}
        }}
        sm.mul(sm, INV_N);
        sq.mul(sq, INV_N);
        m2.mul(sm, sm);
        sq.sub(sq, m2);
        sq.add(sq, 1.0e-6f32);
        sq.rsqrt_approx(sq);
        for ci in 0..CG {{
            let c = g * CG + ci;
            v16.fill(0.0f32);
            v16.add(v16, bcast(sm, &[0, 1]));
            v16.store(gs.sub([c, 0], [1, 16]));
            v16.fill(0.0f32);
            v16.add(v16, bcast(sq, &[0, 1]));
            v16.store(gs.sub([C + c, 0], [1, 16]));
        }}
    }}
}}
'''


def emit_gn_apply(hw, c, act):
    act_true = 'true' if act == 'silu' else 'false'
    return f'''
/// GroupNorm 应用（原地）：y = [(x − mean) · rstd · gamma + beta]（可选 silu），x/y [C][HW] bf16。
#[npu::kernel(cores = 2, numerics = "ks1")]
pub fn k_gna_{hw}_{c}_{act}(x: npu::Ptr<npu::Ddr, npu::bf16>, gamma: npu::Ptr<npu::Ddr, f32>,
                            beta: npu::Ptr<npu::Ddr, f32>, stats: npu::Ptr<npu::Ddr, f32>,
                            y: npu::Ptr<npu::Ddr, npu::bf16>, nchunk: usize) {{
    use npu::prelude::*;
    const C: usize = {c};
    const HW: usize = {hw};
    const CH: usize = 256;
    const ACT: bool = {act_true};
    let gx = x.bind_strided::<npu::Read, 2>([C, HW], [HW, 1]);
    let gy = y.bind_strided::<npu::Write, 2>([C, HW], [HW, 1]);
    let gs = stats.bind_strided::<npu::Read, 2>([2 * C, 16], [16, 1]);
    let (gg, gb2) = (gamma.bind_strided::<npu::Read, 2>([C, 16], [16, 1]),
                     beta.bind_strided::<npu::Read, 2>([C, 16], [16, 1]));
    let (bb, f) = (vcache::<bf16, _>([1, CH]), vcache::<f32, _>([1, CH]));
    let (m16, mm) = (vcache::<f32, _>([1, 16]), vcache::<f32, _>([1, 1]));
    let (rstd, ga, be) = (vcache::<f32, _>([1, 1]), vcache::<f32, _>([1, 1]),
                          vcache::<f32, _>([1, 1]));
    let tmp = vcache::<bf16, _>([1, CH]);
    const CORE: usize = C / 2;
    let mine = core_rank() * CORE;
    for i in 0..CORE {{
        let c = mine + i;
        m16.load(gs.sub([c, 0], [1, 16]));
        mm.reduce(Combine::Max, m16, 1);
        m16.load(gs.sub([C + c, 0], [1, 16]));
        rstd.reduce(Combine::Max, m16, 1);
        m16.load(gg.sub([c, 0], [1, 16]));
        ga.reduce(Combine::Max, m16, 1);
        m16.load(gb2.sub([c, 0], [1, 16]));
        be.reduce(Combine::Max, m16, 1);
        for j in 0..nchunk {{
            bb.load(gx.sub([c, j * CH], [1, CH]));
            f.cvt(bb, Cvt::default());
            f.sub(f, bcast(mm, &[0, 1]));
            f.mul(f, bcast(rstd, &[0, 1]));
            f.mul(f, bcast(ga, &[0, 1]));
            f.add(f, bcast(be, &[0, 1]));
            bb.cvt(f, Cvt::default());
            if ACT {{
                calblas::logistic!(silu, bb, bb, tmp);
            }}
            bb.store(gy.sub([c, j * CH], [1, CH]));
        }}
    }}
}}
'''


def emit_attn(h, c):
    T = h * h
    return f'''
/// VAE mid attention：T={T}, D={c}, 单头，无 mask，无残差（残差在图外面）。
///
/// mma 累加链要避免「分数链 + 4 个 PV 累加器链」同时在飞（docc calt-fate 会拒）：
/// 第一遍只算分数取行最大；第二遍先算 p 全存进 VB（{T // 128} 块 × 16 行 × 128），再单独一段 4 个
/// PV 累加器按块累加（次序固定，和 conv FAST 路的四个累加器同构）。
#[npu::kernel(cores = 2, numerics = "ks1")]
pub fn sd_vae_attn(q: npu::Ptr<npu::Ddr, npu::bf16>, kt: npu::Ptr<npu::Ddr, npu::bf16>,
                   v: npu::Ptr<npu::Ddr, npu::bf16>, o: npu::Ptr<npu::Ddr, npu::bf16>) {{
    use npu::prelude::*;
    const T: usize = {T};
    const D: usize = {c};
    const KB: usize = 128;
    const NB: usize = T / KB;
    const DC: usize = 128;
    const ND: usize = D / DC;
    const SCALE: f32 = 0.044194174f32;
    let gq = q.bind_strided::<npu::Read, 2>([T, D], [D, 1]);
    let gkt = kt.bind_strided::<npu::Read, 2>([D, T], [T, 1]);
    let gv = v.bind_strided::<npu::Read, 2>([T, D], [D, 1]);
    let go = o.bind_strided::<npu::Write, 2>([T, D], [D, 1]);
    let (lq, rk, sacc) = (mcache::<bf16, _>([16, DC]), mcache::<bf16, _>([DC, KB]),
                          mcache::<f32, _>([16, KB]));
    let (lp, rv) = (mcache::<bf16, _>([16, KB]), mcache::<bf16, _>([KB, DC]));
    let (a0, a1) = (mcache::<f32, _>([16, DC]), mcache::<f32, _>([16, DC]));
    let (a2, a3) = (mcache::<f32, _>([16, DC]), mcache::<f32, _>([16, DC]));
    let (sf, sb, tf) = (vcache::<f32, _>([16, KB]), vcache::<bf16, _>([16, KB]),
                        vcache::<f32, _>([16, KB]));
    let (m, top, mb) = (vcache::<f32, _>([16, 1]), vcache::<f32, _>([16, 1]),
                        vcache::<bf16, _>([16, 1]));
    let (l, rs, lb) = (vcache::<f32, _>([16, 1]), vcache::<f32, _>([16, 1]),
                       vcache::<bf16, _>([16, 1]));
    let (out, outb) = (vcache::<f32, _>([16, DC]), vcache::<bf16, _>([16, DC]));
    let pall = vcache::<bf16, _>([NB * 16, KB]);
    const QB: usize = T / 16;
    let mine = core_rank() * (QB / num_cores());
    let cnt = if core_rank() + 1 < num_cores() {{ QB / num_cores() }} else {{ QB - mine }};
    for i in 0..cnt {{
        let qb = mine + i;
        // ── 第一遍：行最大
        m.fill(-3.0e38f32);
        for kb in 0..NB {{
            sacc.fill(0.0f32);
            for dk in 0..ND {{
                lq.load(gq.sub([qb * 16, dk * DC], [16, DC]));
                rk.load(gkt.sub([dk * DC, kb * KB], [DC, KB]));
                sacc.mma(lq, rk);
            }}
            sf.from_m(sacc);
            sb.cvt(sf, Cvt::default());
            sb.mul(sb, SCALE);
            tf.cvt(sb, Cvt::default());
            top.reduce(Combine::Max, tf, 1);
            m.max(m, top);
        }}
        mb.cvt(m, Cvt::default());
        // ── 第二遍 a：p = exp(s − max)，存进 VB，同时累计行和
        l.fill(0.0f32);
        for kb in 0..NB {{
            sacc.fill(0.0f32);
            for dk in 0..ND {{
                lq.load(gq.sub([qb * 16, dk * DC], [16, DC]));
                rk.load(gkt.sub([dk * DC, kb * KB], [DC, KB]));
                sacc.mma(lq, rk);
            }}
            sf.from_m(sacc);
            sb.cvt(sf, Cvt::default());
            sb.mul(sb, SCALE);
            sb.sub(sb, bcast(mb, &[0, 1]));
            sb.exp_approx(sb);
            tf.cvt(sb, Cvt::default());
            rs.reduce(Combine::Sum, tf, 1);
            l.add(l, rs);
            pall.rows(kb * 16, 16).copy(sb);
        }}
        lb.cvt(l, Cvt::default());
        // ── 第二遍 b：out = Σ p·v（4 个头维块，累加器次序固定）
        a0.fill(0.0f32);
        a1.fill(0.0f32);
        a2.fill(0.0f32);
        a3.fill(0.0f32);
        for kb in 0..NB {{
            lp.load_v(pall.rows(kb * 16, 16));
            rv.load(gv.sub([kb * KB, 0 * DC], [KB, DC]));
            a0.mma(lp, rv);
            rv.load(gv.sub([kb * KB, 1 * DC], [KB, DC]));
            a1.mma(lp, rv);
            rv.load(gv.sub([kb * KB, 2 * DC], [KB, DC]));
            a2.mma(lp, rv);
            rv.load(gv.sub([kb * KB, 3 * DC], [KB, DC]));
            a3.mma(lp, rv);
        }}
        out.from_m(a0);
        outb.cvt(out, Cvt::default());
        outb.div(outb, bcast(lb, &[0, 1]));
        outb.store(go.sub([qb * 16, 0 * DC], [16, DC]));
        out.from_m(a1);
        outb.cvt(out, Cvt::default());
        outb.div(outb, bcast(lb, &[0, 1]));
        outb.store(go.sub([qb * 16, 1 * DC], [16, DC]));
        out.from_m(a2);
        outb.cvt(out, Cvt::default());
        outb.div(outb, bcast(lb, &[0, 1]));
        outb.store(go.sub([qb * 16, 2 * DC], [16, DC]));
        out.from_m(a3);
        outb.cvt(out, Cvt::default());
        outb.div(outb, bcast(lb, &[0, 1]));
        outb.store(go.sub([qb * 16, 3 * DC], [16, DC]));
    }}
}}
'''


def gen_kernels(P):
    L = ['//! SD1.5 VAE decoder kernel 特化（gen.py 生成，勿手改）', '']
    seen = set()
    for op in P.ops:
        kind = op['kind']
        if kind == 'conv':
            k = op['key']
            if k in seen:
                continue
            seen.add(k)
            h, cin, cout, kk, pin, pout, has_res = k[1:]
            res = f', res = ({pout}, 0)' if has_res else ''
            L.append(f"calblas::conv2d!(pub {conv_name(k)}: h = {h}, w = {h}, cin = {cin},"
                     f" cout = {cout}, k = {kk}, s = 1, pin = ({pin}, 0), pout = ({pout}, 0),"
                     f" act = none{res});")
        elif kind == 'gn_stats':
            k = op['key']
            if k in seen:
                continue
            seen.add(k)
            L.append(emit_gn_stats(op['key'][1], op['key'][2], op['key'][3]))
        elif kind == 'gn_apply':
            k = op['key']
            if k in seen:
                continue
            seen.add(k)
            L.append(emit_gn_apply(k[1], k[2], k[3]))
        elif kind == 'layout':
            k = op['key']
            if k in seen:
                continue
            seen.add(k)
            d, h, c, p = k[0], k[1], k[2], k[3]
            macro = 'nhwc_to_nchw' if d == 'nhwc2nchw' else 'nchw_to_nhwc'
            L.append(f"calblas::{macro}!(pub {layout_name(k)}: h = {h}, w = {h}, c = {c}, p = {p});")
        elif kind == 'up':
            k = op['key']
            if k in seen:
                continue
            seen.add(k)
            h, c, pin, pout = k[1:]
            L.append(f"calblas::upsample2!(pub {up_name(k)}: h = {h}, w = {h}, c = {c},"
                     f" pin = {pin}, pout = {pout});")
        elif kind == 'transpose':
            k = op['key']
            if k in seen:
                continue
            seen.add(k)
            r, c = k[1], k[2]
            L.append(f"calblas::transpose2d!(pub k_t2d_{r}_{c}: rows = {r}, cols = {c});")
    L.append('')

    L.append('// 图像 slot 边框清零：n 是运行期标量（常量循环体会被展开成 1 MiB 代码，链接器分支够不着）')
    L.append('calblas::fill16k!(pub k_fill0: value = 0.0f32);')

    L.append(CUSTOM_MOVE)
    mov_seen = set()
    for op in P.ops:
        if op['kind'] == 'crop' and op['key'] not in mov_seen:
            mov_seen.add(op['key'])
            _, h, pin, pout = op['key']
            L.append(emit_move(f'k_crop_{h}_p{pin}_{pout}', True, h, pin, pout))
    for op in P.ops:
        if op['kind'] == 'pad' and op['key'] not in mov_seen:
            mov_seen.add(op['key'])
            _, h, pin, pout = op['key']
            L.append(emit_move(f'k_pad_{h}_p{pin}_{pout}', False, h, pin, pout))
    L.append(CUSTOM_ATTN)
    done_attn = False
    for op in P.ops:
        if op['kind'] == 'attn' and not done_attn:
            done_attn = True
            L.append(emit_attn(op['key'][1], op['key'][2]))
    return '\n'.join(L)


# ---------------------------------------------------------------- step.rs

def off(P, name):
    b = P.bufs[name]
    if b['klass'][0] in ('input', 'output'):
        return f"{b['decl']}()"
    return f'act().at({b["offset"]})'


def gen_step(P):
    L = ['//! SD1.5 VAE decoder 整步程序（gen.py 生成，勿手改）。',
         'use crate::kernels::*;',
         'use npu::prelude::*;',
         '',
         'npu::decls! {']
    for n in P.order:
        b = P.bufs[n]
        if b['klass'][0] == 'input':
            L.append(f'    {b["decl"]}: input, bytes = {b["size"]};')
        elif b['klass'][0] == 'output':
            L.append(f'    {b["decl"]}: output, bytes = {b["size"]};')
    L.append(f'    w: weights, bytes = {P.param_bytes};')
    L.append(f'    act: workspace, bytes = {P.act_bytes};')
    L.append('}')
    L.append('')
    L.append('#[npu::program(chiplets = 1, cores = 2)]')
    L.append('pub fn step() {')
    L.append('    // 所有图像 slot 一次清零：边框之后不再被写')
    L.append('    if core_rank() == 0 {')
    for (o, blocks) in P.fills:
        L.append(f'        k_fill0(act().at({o}).cast(), {blocks});')
    # 输出声明的边框也清零：段间用裸字节衔接（out_a.bin 直接喂 b1 的 in_x），
    # 而 conv 只写内部区域，边框不先清零就会把上一段内存里的残留当 padding 读进去。
    for n in P.order:
        b = P.bufs[n]
        if b['klass'][0] == 'output':
            L.append(f'        k_fill0(out_y().cast(), {b["size"] // FILL_BLK});')
    L.append('    }')
    L.append('    sync_cores();')

    for op in P.ops:
        kind = op['kind']
        if kind == 'conv':
            args = [f'{off(P, op["x"])}.cast()',
                    f'w().at({op["w"]["offset"]}).cast()',
                    f'w().at({op["b"]["offset"]}).cast()']
            if op['res']:
                args.append(f'{off(P, op["res"])}.cast()')
            args.append(f'{off(P, op["y"])}.cast()')
            L.append(f'    {conv_name(op["key"])}({", ".join(args)});')
            L.append('    sync_cores();')
        elif kind == 'gn_stats':
            L.append(f'    {gn_stats_name(op["key"])}({off(P, op["x"])}.cast(), '
                     f'{off(P, op["y"])}.cast(), {op["nchunk"]});')
            L.append('    sync_cores();')
        elif kind == 'gn_apply':
            L.append(f'    {gn_apply_name(op["key"])}({off(P, op["x"])}.cast(), '
                     f'w().at({op["g"]["offset"]}).cast(), w().at({op["b"]["offset"]}).cast(), '
                     f'{off(P, op["s"])}.cast(), {off(P, op["y"])}.cast(), {op["nchunk"]});')
            L.append('    sync_cores();')
        elif kind == 'layout':
            L.append(f'    if core_rank() == 0 {{ {layout_name(op["key"])}('
                     f'{off(P, op["x"])}.cast(), {off(P, op["y"])}.cast()); }}')
            L.append('    sync_cores();')
        elif kind == 'up':
            L.append(f'    if core_rank() == 0 {{ {up_name(op["key"])}('
                     f'{off(P, op["x"])}.add({op["xoff"]}).cast(), '
                     f'{off(P, op["y"])}.add({op["yoff"]}).cast()); }}')
            L.append('    sync_cores();')
        elif kind in ('crop', 'pad'):
            name = f'k_{kind}_{op["key"][1]}_p{op["key"][2]}_{op["key"][3]}'
            L.append(f'    if core_rank() == 0 {{ {name}('
                     f'{off(P, op["x"])}.cast(), {off(P, op["y"])}.cast()); }}')
            L.append('    sync_cores();')
        elif kind == 'transpose':
            L.append(f'    if core_rank() == 0 {{ k_t2d_{op["key"][1]}_{op["key"][2]}('
                     f'{off(P, op["x"])}.cast(), {off(P, op["y"])}.cast()); }}')
            L.append('    sync_cores();')
        elif kind == 'attn':
            L.append(f'    sd_vae_attn({off(P, op["q"])}.cast(), {off(P, op["kt"])}.cast(), '
                     f'{off(P, op["v"])}.cast(), {off(P, op["o"])}.cast());')
            L.append('    sync_cores();')
    L.append('}')
    L.append('')
    return '\n'.join(L)


# ---------------------------------------------------------------- main

def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    stage = 'a'
    if '--stage' in sys.argv:
        stage = sys.argv[sys.argv.index('--stage') + 1]
    P = build(stage)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'kernels.rs'), 'w') as f:
        f.write(gen_kernels(P))
    with open(os.path.join(outdir, 'step.rs'), 'w') as f:
        f.write(gen_step(P))
    with open(os.path.join(outdir, 'params.json'), 'w') as f:
        json.dump(dict(param_bytes=P.param_bytes,
                       params=[{k: v for k, v in p.items()} for p in P.params]),
                  f, indent=1)
    with open(os.path.join(outdir, 'plan.json'), 'w') as f:
        json.dump(dict(act_bytes=P.act_bytes, param_bytes=P.param_bytes,
                       n_ops=len(P.ops), n_kernels=len({str(o['key']) for o in P.ops}),
                       fills=P.fills, slots=P.slots_info,
                       buffers={n: dict(tag=b['tag'], klass=list(b['klass']), size=b['size'],
                                        offset=b.get('offset'), birth=b.get('birth'),
                                        last=b.get('last'))
                                for n, b in P.bufs.items() if b.get('birth') is not None},
                       ops=[dict(kind=o['kind'], key=list(o.get('key') or []),
                                 x=o.get('x'), y=o.get('y'), res=o.get('res'),
                                 q=o.get('q'), k=o.get('k'), v=o.get('v'),
                                 kt=o.get('kt'), o=o.get('o'),
                                 xoff=o.get('xoff'), yoff=o.get('yoff'),
                                 w=(o['w']['offset'] if 'w' in o else None),
                                 b=(o['b']['offset'] if 'b' in o else None),
                                 g=(o['g']['offset'] if 'g' in o else None),
                                 s=o.get('s'), nchunk=o.get('nchunk'))
                            for o in P.ops]), f, indent=1)
    print(f'stage={stage} ops={len(P.ops)} kernels={len({str(o["key"]) for o in P.ops})} '
          f'act={P.act_bytes/2**20:.2f} MiB params={P.param_bytes/2**20:.2f} MiB '
          f'fill_calls={len(P.fills)}')
    from collections import Counter
    print('kinds:', dict(Counter(o['kind'] for o in P.ops)))


if __name__ == '__main__':
    main()
