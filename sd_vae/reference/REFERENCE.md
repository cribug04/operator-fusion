# SD1.5 VAE 解码器黄金参考（diffusers / torch CPU）

本文档由 `ref.py` 真实执行生成（2026-11-14，WSL2 Ubuntu 24.04）。所有数值均来自实际运行，不是估计值。

## 0. 一句话结论

- 参考实现：diffusers `AutoencoderKL.from_pretrained(<SD1.5 repo>, subfolder="vae")`，**fp32**，CPU，`torch.no_grad()`。
- 输入 latent：`np.random.default_rng(1234).standard_normal((1,4,64,64)).astype(np.float32)`，按 diffusers 语义除以 `scaling_factor = 0.18215` 得到 z。
- 解码链：`z -> post_quant_conv(1x1) -> decoder`，输出 `[1,3,512,512]` fp32 NCHW，范围约 [-1.8671, 1.6532]。
- `reference_image.bin` sha256 = `5dcfd42c0a36f4727ddaff2ef14c5b01c4cbcf6e36ed852acb9648858e03475b`。
- 逐层 dump 共 **143 个事件**（16233.4 MB），完整清单见 `dumps_index.json`。

## 1. 环境与版本

| 项 | 值 |
|---|---|
| torch | 2.15.0+cpu |
| diffusers | 0.43.0 |
| transformers | 5.23.0 |
| numpy | 2.5.4 |
| safetensors | 0.8.0 |
| Pillow | 13.0.0 |
| python | 3.12.3 |
| platform | Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| CPU | x86_64 |
| torch.num_threads | 20 |
| device | CPU |

安装命令（WSL2 Ubuntu 24.04, root）：

```bash
export MSYS2_ARG_CONV_EXCL='*'
wsl.exe -d Ubuntu -u root -- bash -s <<'EOF'
apt-get update && apt-get install -y python3 python3-pip python3-venv
python3 -m venv /root/sd_vae/venv
/root/sd_vae/venv/bin/pip install --upgrade pip
# torch CPU index 可用（实测 HTTP 200），直接装 CPU 包，不拉 CUDA：
/root/sd_vae/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/root/sd_vae/venv/bin/pip install safetensors numpy pillow diffusers transformers
EOF
```

权重下载（hf-mirror.com 可用；huggingface.co 不通）：

```bash
cd /root/sd_vae/model
curl -L -o model_index.json https://hf-mirror.com/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/model_index.json
cd vae
curl -L -o config.json https://hf-mirror.com/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/vae/config.json
curl -L -o diffusion_pytorch_model.fp16.safetensors https://hf-mirror.com/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors
curl -L -o diffusion_pytorch_model.safetensors https://hf-mirror.com/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.safetensors
```

运行：

```bash
cd /root/sd_vae && venv/bin/python ref.py > ref/run.log 2>&1
```

## 2. latent（解码输入）

- latent_raw：`np.random.default_rng(1234).standard_normal((1,4,64,64)).astype(np.float32)`，shape `[1,4,64,64]`，min=-3.818862 max=4.218847 mean=-0.008695 std=1.000865
- `scaling_factor = 0.18215`（config 实际值）
- **z = latent_raw / scaling_factor**（fp32），stats: min=-20.965477 max=23.161386 mean=-0.047733 std=5.494731
- `latent.bin` = z 的 raw fp32 little-endian（NCHW，无 header），就是 **喂给 `post_quant_conv` 的张量**（已校验：post_quant_conv 的 hook 输入与 z 逐 bit 相等 = True）。
- `latent.npy` 存的是缩放后的 z；`latent_unnormalized.npy` 存的是缩放前的 latent_raw。

## 3. 输出图像统计

| 项 | 值 |
|---|---|
| shape | [1, 3, 512, 512] (NCHW) |
| dtype | float32 |
| min | -1.867134 |
| max | 1.653239 |
| mean | -0.089909 |
| std (ddof=0) | 0.738124 |
| abs_mean | 0.642957 |
| rms | 0.743580 |
| 超出 [-1,1] 的像素占比 | 0.238984 |
| sha256(reference_image.bin) | 5dcfd42c0a36f4727ddaff2ef14c5b01c4cbcf6e36ed852acb9648858e03475b |
| bin 格式 | raw fp32 little-endian, NCHW, C order, 无 header |
| PNG 转换 | u8 = round(clip(x,-1,1)*127.5+127.5), RGB |
| PNG sha256 | 29ed5b81e9a997f2c3a2e5abb6a64ff453eacffd66b4769bd59aac26f6c354d6 |

逐通道统计：

| channel | min | max | mean | std |
|---|---|---|---|---|
| 0 | -1.867134 | 1.573753 | -0.173761 | 0.710800 |
| 1 | -1.687167 | 1.550352 | -0.059832 | 0.725024 |
| 2 | -1.820274 | 1.653239 | -0.036134 | 0.769908 |

### 3.1 stage 边界参考张量（统计量 + raw sha256，给逐段数值对齐用）

sha256 是对 **C-order 连续 fp32 little-endian 原始字节**（不含 npy header）计算的。

| stage | module | shape | min | max | mean | std | sha256(raw fp32 LE) |
|---|---|---|---|---|---|---|---|
| mid_block_resnets1_out | decoder.mid_block.resnets.1 | 1x512x64x64 | -19.078999 | 18.860222 | 0.037577 | 1.596248 | 64323559aa14508d229831281f899fc315a54a4d3901f20a0728071f7989c672 |
| up_blocks1_upsampler_out | decoder.up_blocks.1.upsamplers.0 | 1x512x256x256 | -232.796936 | 214.554504 | -0.809622 | 18.617849 | 96de91009d01a50a0256b18ad4e24c5d197049f8d66db42a9f804412a85f33fa |
| conv_out | decoder.conv_out | 1x3x512x512 | -1.867134 | 1.653239 | -0.089909 | 0.738124 | 5dcfd42c0a36f4727ddaff2ef14c5b01c4cbcf6e36ed852acb9648858e03475b |

对应的 dump 文件：`mid_block_resnets1_out` → `dump_026_decoder_mid_block_resnets_1_out0.npy`, `up_blocks1_upsampler_out` → `dump_082_decoder_up_blocks_1_upsamplers_0_out0.npy`, `conv_out` → `dump_141_decoder_conv_out_out0.npy`。

## 4. 层表摘要

- 总事件数 **143**（模块级 + interpolate + softmax/sdpa + 合成 add 事件）
- 按 op 计数：

| op | count |
|---|---|
| conv2d | 36 |
| groupnorm | 30 |
| silu | 29 |
| add | 15 |
| resnet_block | 14 |
| linear | 4 |
| up_block | 4 |
| interpolate | 3 |
| upsample | 3 |
| attention | 1 |
| decoder | 1 |
| mid_block | 1 |
| sdpa | 1 |
| softmax | 1 |

- 按 class 计数：

| class | count |
|---|---|
| Conv2d | 36 |
| GroupNorm | 30 |
| SiLU | 29 |
| synthetic.add | 15 |
| ResnetBlock2D | 14 |
| Linear | 4 |
| UpDecoderBlock2D | 4 |
| Upsample2D | 3 |
| torch.nn.functional.interpolate | 3 |
| Attention | 1 |
| Decoder | 1 |
| UNetMidBlock2D | 1 |
| torch.nn.functional.scaled_dot_product_attention | 1 |
| torch.softmax | 1 |

- 解码主干（骨架，完整逐层参数见 `layer_table.json`）：

| order | module | op | in shape | out shape |
|---|---|---|---|---|
| 0 | post_quant_conv | conv2d | 1x4x64x64 | 1x4x64x64 |
| 1 | decoder.conv_in | conv2d | 1x4x64x64 | 1x512x64x64 |
| 2 | decoder.mid_block.attentions.0 | attention | 1x512x64x64 | 1x512x64x64 |
| 3 | decoder.mid_block | mid_block | 1x512x64x64 | 1x512x64x64 |
| 4 | decoder.up_blocks.0.upsamplers.0 | upsample | 1x512x64x64 | 1x512x128x128 |
| 5 | decoder.up_blocks.0 | up_block | 1x512x64x64 | 1x512x128x128 |
| 6 | decoder.up_blocks.1.upsamplers.0 | upsample | 1x512x128x128 | 1x512x256x256 |
| 7 | decoder.up_blocks.1 | up_block | 1x512x128x128 | 1x512x256x256 |
| 8 | decoder.up_blocks.2.upsamplers.0 | upsample | 1x256x256x256 | 1x256x512x512 |
| 9 | decoder.up_blocks.2 | up_block | 1x512x256x256 | 1x256x512x512 |
| 10 | decoder.up_blocks.3 | up_block | 1x256x512x512 | 1x128x512x512 |
| 11 | decoder.conv_norm_out | groupnorm | 1x128x512x512 | 1x128x512x512 |
| 12 | decoder.conv_act | silu | 1x128x512x512 | 1x128x512x512 |
| 13 | decoder.conv_out | conv2d | 1x128x512x512 | 1x3x512x512 |
| 14 | decoder | decoder | 1x4x64x64 | 1x3x512x512 |

分辨率链：64x64 -> mid 64x64 -> up0 128x128 -> up1 256x256 -> up2 512x512 -> up3 512x512 -> conv_out [1,3,512,512]。
VAE decoder 中 **没有 concat / 跨层 skip**（只有 ResnetBlock 和 attention 内部的加性残差）。

## 5. 语义要点（逐条实测确认）

### 5.1 解码主干
- `AutoencoderKL.decode(z)` 内部 = `post_quant_conv(z)` 然后 `decoder(...)`（已校验 decoder 输入与 post_quant_conv 输出逐 bit 相等 = True）。
- `post_quant_conv`: **Conv2d(4, 4, kernel=1x1, stride=1, padding=0, bias=True)**；`quant_conv`(8->8) 只用于 encode，**decode 不执行**（因此其权重在 `weights_index.json` 中标记 used_in_decode=false）。
- `decoder.conv_in`: Conv2d(4, 512, k=3, s=1, p=1, bias=True)，输出 [1,512,64,64]。
- 主干顺序：conv_in -> mid_block(resnet0 -> attention -> resnet1) -> up_blocks[0..3]（每块 3 个 ResnetBlock2D；第 0/1/2 块末尾各接一个 Upsample2D，第 3 块无上采样）-> conv_norm_out -> conv_act -> conv_out。
- `decoder.conv_norm_out`: **GroupNorm(num_groups=32, num_channels=128, eps=1e-6, affine=True)**。
- `decoder.conv_act`: **SiLU(inplace=True)**。
- `decoder.conv_out`: **Conv2d(128, 3, k=3, s=1, padding=1, bias=True)** —— 即最终 conv 有 padding=1、有 bias。
- 所有 Conv2d 均为 stride=1、dilation=1、groups=1、bias=True（ResnetBlock 的 1x1 shortcut conv 也是 bias=True）。
- 输出 **不做 clamp**；`reference_image.bin/.npy` 是网络原始输出。PNG 才做 clamp(-1,1)。

### 5.2 ResnetBlock2D（顺序固定）
```python
x = input_tensor
h = norm1(x)            # GroupNorm(32, C, eps=1e-6)
h = silu(h)             # SiLU
h = conv1(h)            # 3x3, pad=1, bias=True
h = norm2(h)            # GroupNorm(32, C, eps=1e-6)
h = silu(h)             # SiLU
h = dropout(h)          # p=0.0, eval 下为恒等
h = conv2(h)            # 3x3, pad=1, bias=True
if conv_shortcut is not None:
    x = conv_shortcut(x)   # 1x1 conv, bias=True
out = (x + h) / output_scale_factor   # output_scale_factor = 1.0
```
- **add 顺序：`input_tensor(或 shortcut(x)) + conv2 输出`**，即残差在前、conv2 输出在后；随后除以 `output_scale_factor=1.0`。
- 每个 ResnetBlock 内部 norm1/norm2 的 GroupNorm 均为 **num_groups=32、eps=1e-6、affine=True**；共 14 个 ResnetBlock（mid 2 个 + 4 个 up block 各 3 个）。
- 只有 2 个 ResnetBlock 带 1x1 `conv_shortcut`：`decoder.up_blocks.2.resnets.0`(512->256) 和 `decoder.up_blocks.3.resnets.0`(256->128)；其余为恒等 shortcut。
- 已对每个 ResnetBlock 用 dump 的 conv2/conv_shortcut 输出反算 add，与 block 输出逐 bit 比对：max_abs_diff = 0.000e+00（远小于 1e-6）。

### 5.3 Attention（mid block 唯一 1 个）
- 类：`diffusers.models.attention_processor.Attention`（`_from_deprecated_attn_block=True`，由旧式 AttnBlock 权重转换而来）。
- 位置：`decoder.mid_block.attentions.0`，在 mid resnet0 之后、mid resnet1 之前。
- **q/k/v 是 `nn.Linear`（不是 Conv2d）**：`to_q/to_k/to_v = Linear(512, 512, bias=True)`。
- **输出投影是 `to_out.0 = Linear(512, 512, bias=True)`**（ModuleList 里第 0 个；第 1 个是 p=0 的 Dropout）。
- **heads = 1，dim_head = 512**（`scale = 1/sqrt(512) = 0.044194173824159223`）。
- 前置 GroupNorm：**num_groups=32, num_channels=512, eps=1e-6, affine=True**，作用在 `[B, C, H*W]` 上（见下）。
- 处理器：`AttnProcessor2_0`，用 `F.scaled_dot_product_attention`（默认 scale = 1/sqrt(E)，与 attn.scale 相同）。
- 精确 forward（实测源码）：
```python
residual = x                                # [B, C, H, W] = [1,512,64,64]
x = x.view(B, C, H*W).transpose(1, 2)       # [B, H*W, C] = [1,4096,512]
x = group_norm(x.transpose(1, 2)).transpose(1, 2)   # GroupNorm 作用在 [B,C,H*W]
q, k, v = to_q(x), to_k(x), to_v(x)         # Linear(512,512,bias=True), 各自 [1,4096,512]
# reshape -> [B, heads=1, seq=4096, head_dim=512]
o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
o = o.transpose(1,2).reshape(B, H*W, C)     # [1,4096,512]
o = to_out[0](o)                            # Linear(512,512,bias=True)
o = o.transpose(-1,-2).reshape(B, C, H, W)  # [1,512,64,64]
x = (o + residual) / rescale_output_factor  # rescale_output_factor = 1.0
```
- **softmax 沿 key 序列方向（4096 个 key，dim=-1）计算，单 head；每个 head 的 key/value 维度为 512。**
- `upcast_softmax=True` 是旧字段，`AttnProcessor2_0` **忽略它**；本参考在 CPU fp32 下运行，所以 softmax 本来就是 fp32。
- attention 内部 residual add 已用 dump 反算校验：max_abs_diff = 0.000e+00。
- 另外单独 dump 了 `attn_sdpa_q/k/v/out`、`attn_softmax_logits`、`attn_softmax_probs`（由捕获的 q,k 按 `q@k^T*scale` 重算）、`attn_sdpa_manual_out`（`probs@v`）。`probs@v` 与 SDPA 输出的 max|Δ| = 4.768e-06（mean 3.495e-08）——说明逐层核对 softmax 时可用该重算路径，误差在 1e-6 量级。

### 5.4 Upsample2D（decoder 里 3 个）
- `F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")`（align_corners=None）后接 **Conv2d(C, C, k=3, s=1, padding=1, bias=True)**。
- 上采样点：`up_blocks.0`(512ch,64->128)、`up_blocks.1`(512ch,128->256)、`up_blocks.2`(256ch,256->512)；`up_blocks.3` 无上采样。
- 3 次 interpolate 调用参数已 dump（`dump_*_interpolate_*.npy`）。

### 5.5 其它
- `scaling_factor = 0.18215`，`shift_factor = null`，`latents_mean/latents_std = null`（无 shift / 无 mean-std 归一化）。
- config: latent_channels=4, block_out_channels=[128,256,512,512], layers_per_block=2（decoder 每块实际 3 个 resnet = layers_per_block+1）, norm_num_groups=32, act_fn=silu, force_upcast=true（pipeline 级，直接 `vae.decode` fp16 时不生效）, use_quant_conv=true, use_post_quant_conv=true, mid_block_add_attention=true。
- 全部 GroupNorm affine=True；全部 Conv2d/Linear bias 均存在。
- decoder 共 49,490,199 个参数（含 post_quant_conv），整个 AutoencoderKL 共 83,653,863（其中 encoder 34,163,664 不参与 decode）。

## 6. fp16 / bf16 vs fp32 差异

同一 z（fp16/bf16 模型下先 cast 输入），与 fp32 参考图像比较：

| dtype | status | max abs delta | mean abs delta | p99 abs delta | u8 max abs delta |
|---|---|---|---|---|---|
| fp16 | SKIPPED/未做 | - | - | - | - |
| bf16 | SKIPPED/未做 | - | - | - | - |

> **本次 fp16/bf16 解码未做（按 lead 指示砍掉）**，因此没有 fp16-vs-fp32 差异数值。
> 原因（实测）：本机 CPU 上 batch=1 的 fp16/bf16 Conv2d 会走单线程慢路径 —— 单层
> `Conv2d(128->128, k=3)` @ 512x512 实测 **fp16 = 176.7s、bf16 = 111.6s**（同层 fp32 仅 0.36s，2 线程），
> 整套 VAE 解码按此推算需要数小时，不现实。若真机需要 bf16 误差判据，建议在 GPU/真机上另做，
> 或采用“抽若干层单独比对”的方式；fp32 golden 的确定性已验证（同一输入两次 decode 逐 bit 相同），
> 可直接作为唯一基准。

## 7. 文件清单 + sha256

（`ref/` 下所有 dump 的 sha256 见 `dumps_index.json`；dump 共 16233.4 MB，保留在 WSL `/root/sd_vae/ref/`，未复制到 Windows）

| file | bytes | sha256 |
|---|---|---|
| model/vae/config.json | 547 | 786a7d21647ddea6a04b9675c03d3cb45e90a2f3c6da5fbda2c54ade040036de |
| model/model_index.json | 541 | e2f6f22e274374010aec30c79eb2d6e53fff4a36ac0225523a92cb2d41a83347 |
| model/vae/diffusion_pytorch_model.fp16.safetensors | 167,335,342 | 4fbcf0ebe55a0984f5a5e00d8c4521d52359af7229bb4d81890039d2aa16dd7c |
| model/vae/diffusion_pytorch_model.safetensors | 334,643,276 | a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c |
| ref/latent_unnormalized.npy | 65,664 | d1763225617cde0339212cdeab85c87131207f116459b1e5c2cc8d9ac7ca5378 |
| ref/latent.npy | 65,664 | 62552de2e1d95596e972e69bb636b19437503e214c35667c6fbc201c6ea2f357 |
| ref/latent.bin | 65,536 | 79ed341043fe4be1f398e53a3af03d190d0a189fbc23034570acba4e8330a031 |
| ref/reference_image.npy | 3,145,856 | b7cc6839afa2b5a6d0d16b73bc061216d5cc4fcab032e39f3a3f7e8442bce126 |
| ref/reference_image.bin | 3,145,728 | 5dcfd42c0a36f4727ddaff2ef14c5b01c4cbcf6e36ed852acb9648858e03475b |
| ref/reference_image.png | 576,950 | 29ed5b81e9a997f2c3a2e5abb6a64ff453eacffd66b4769bd59aac26f6c354d6 |
| ref/layer_table.json | 165,711 | 6f31d61f51f04478644772a611a904876ee10c27bfb68da310ad550edb331476 |
| ref/weights_index.json | 90,975 | 9c31972fc17d5cd9b6af5aa749df49d191c4306d68e72f1dea16aa3c0527f90c |
| ref/dumps_index.json | 161,023 | 36e4d6cbadf2c0ee5b5f5d4c10461043bf35e10771494b517ebcee5e922476fe |
| ref/stage_summary.json | 1,666 | ff5c610224af3925532a35a0e6fea2cabdbd07ce018df9a02079a891ca59e750 |

## 8. 自检（全部来自真实运行）

| check | result |
|---|---|
| fp32 模型参数与 fp32 safetensors 逐 bit 相等 | True (248/248) |
| post_quant_conv 输入 == z（逐 bit） | True |
| numpy 重算 post_quant_conv 1x1 conv 的 max abs diff | 9.537e-07 |
| decoder 输入 == post_quant_conv 输出（逐 bit） | True |
| conv_in 用 F.conv2d 重算 max abs diff | 0.000e+00 |
| decoder 输出 == reference_image（逐 bit） | True |
| 第二次干净 decode 与第一次逐 bit 相同 | True |
| 所有 add 顺序反算均 < 1e-6 | True |
| SDPA 调用次数 / interpolate 调用次数 | 1 / 3 |
| PNG roundtrip 与 u8 期望值 max 差 | 0 |

## 9. 遇到的坑与失败记录（原样记录）

1. `download.pytorch.org/whl/cpu` 实测 **HTTP 200 可用**，torch 2.15.0+cpu 209.5 MB 正常下载安装，没有退到清华源；因此**没有**拉 CUDA 大包。
2. 加载时 diffusers 提示 `accelerate` 未安装（"Cannot initialize model with low cpu memory usage..."），仅是性能提示，**不影响结果**；未安装 accelerate。
3. 第一次 introspection 脚本命名为 `inspect.py`，与 Python 标准库 `inspect` 循环导入冲突报错 `AttributeError: partially initialized module 'inspect' has no attribute 'signature'`，已改名 `recon_inspect.py`。
4. fp32 权重文件是 **旧式 CompVis 命名**（`mid_block.attentions.0.query/key/value/proj_attn`），diffusers 在 `load_state_dict` 时自动转换成 `to_q/to_k/to_v/to_out.0`。`weights_index.json` 同时给出 st_key 与 model_key。
5. 首次 `from_pretrained` 打印了 `low_cpu_mem_usage` 警告（同上，不影响）。
6. 本机 CPU fp16/bf16 **小规模** conv 可用（4->512 @64x64 测试通过），但 **batch=1 大分辨率** fp16/bf16 conv 走单线程慢路径：实测 `Conv2d(128->128,k=3,512x512)` fp16=176.7s、bf16=111.6s（fp32 同层 0.36s），因此完整 fp16/bf16 VAE 解码不可行，已放弃 fp16 对比（见第 6 节）。
7. fp16 实验曾真实启动约 5 分钟无完成迹象（进程 100% 单核），随后按 lead 指示终止；这不影响 fp32 golden 与全部 dump（它们在 fp16 阶段之前已落盘）。

## 10. 给 NPU 实现的提示

- 逐层核对时请用 `dumps_index.json` 的 seq 顺序；每个 dump 都是该模块 **实际 forward 的输入/输出**（fp32 LE）。
- attention 请按 5.3 的 forward 实现核对：q/k/v Linear（带 bias）→ reshape 成单 head 4096x512 → `q@k^T/sqrt(512)` → softmax(dim=-1) → `@v` → to_out Linear（带 bias）→ 加 residual → 不缩放。
- ResnetBlock 的 add 语义固定为 `(shortcut 分支) + (conv2 输出)`，再除以 1.0；不要漏掉 norm/act 的顺序（norm1→silu→conv1→norm2→silu→conv2）。
- 上采样是 nearest x2 + 3x3 conv，不是转置卷积。
- 输出不要 clamp；如需 PNG 再 clamp。
