"""
PRFNet (Polarization-Retinex Fusion Network)
Combines Stage 1 PRDNet and Stage 2 DPFNet (SSCF, DICB with I-Branch/HV-Branch, ICIF, CMF).
"""
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from timm.models.layers import trunc_normal_

from models.prdnet import PRDNet, remap_prdnet_key

pi = 3.141592653589793

LEGACY_SINGLE_BRANCH_PREFIXES = (
    'conv_first1.', 'conv_first2.', 'fusion_concat_conv.', 're_concat_conv.',
    'layers_Ex.', 'layers_Fusion.', 'layers_Reconstruction.',
    'conv_last1.', 'conv_last2.',
)

LEGACY_PRFNET_KEY_MAP = (
    ('shallow_spectral_fusion.', 'sscf.'),
    ('encoder_bridges_I_HV.', 'icif_enc.'),
    ('bottleneck_bridge_I_HV.', 'icif_bottleneck.'),
    ('decoder_bridges_I_HV.', 'icif_dec.'),
    ('layers_Fusion_I.', 'cmf_I.'),
    ('layers_Fusion_HV.', 'cmf_HV.'),
)


def load_prfnet_state_dict(model, checkpoint, logger=None):
    """Load PRFNet weights; remap legacy module names and filter obsolete single-branch keys."""
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    remapped = {}
    skipped = []
    remapped_count = 0

    for k, v in state_dict.items():
        if k.startswith(LEGACY_SINGLE_BRANCH_PREFIXES):
            skipped.append(k)
            continue

        if k.startswith('p_raeg.'):
            nk = 'prdnet.' + remap_prdnet_key(k[len('p_raeg.'):])
        elif k.startswith('prdnet.'):
            nk = 'prdnet.' + remap_prdnet_key(k[len('prdnet.'):])
        else:
            nk = k
            for old_prefix, new_prefix in LEGACY_PRFNET_KEY_MAP:
                if k.startswith(old_prefix):
                    nk = new_prefix + k[len(old_prefix):]
                    break

        if nk != k:
            remapped_count += 1
        remapped[nk] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"PRFNet state dict mismatch: missing={missing}, unexpected={unexpected}")
    if skipped and logger is not None:
        logger.info(f"   ✓ 已自动过滤 {len(skipped)} 个旧版单分支冗余权重键")
    if remapped_count and logger is not None:
        logger.info(f"   ✓ 已自动映射 {remapped_count} 个历史 PRFNet 权重键")


class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1], 0.2))
        self.this_k = 0

    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:, 2] == value] = 4.0 + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[img[:, 2] == value]
        hue[img[:, 1] == value] = 2.0 + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[img[:, 1] == value]
        hue[img[:, 0] == value] = (0.0 + ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[img[:, 0] == value]) % 6

        hue[img.min(1)[0] == value] = 0.0
        hue = hue / 6.0

        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k
        self.this_k = k.item()

        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        H = color_sensitive * saturation * (2.0 * pi * hue).cos()
        V = color_sensitive * saturation * (2.0 * pi * hue).sin()
        return torch.cat([H, V, value], dim=1)

    def PHVIT(self, img):
        eps = 1e-8
        H, V, I = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)

        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = torch.clamp(H / (color_sensitive + eps), -1, 1)
        V = torch.clamp(V / (color_sensitive + eps), -1, 1)
        h = (torch.atan2(V + eps, H + eps) / (2 * pi)) % 1
        s = torch.clamp(torch.sqrt(H ** 2 + V ** 2 + eps), 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))

        hi0, hi1, hi2, hi3, hi4, hi5 = hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5
        r[hi0], g[hi0], b[hi0] = v[hi0], t[hi0], p[hi0]
        r[hi1], g[hi1], b[hi1] = q[hi1], v[hi1], p[hi1]
        r[hi2], g[hi2], b[hi2] = p[hi2], v[hi2], t[hi2]
        r[hi3], g[hi3], b[hi3] = p[hi3], q[hi3], v[hi3]
        r[hi4], g[hi4], b[hi4] = t[hi4], p[hi4], v[hi4]
        r[hi5], g[hi5], b[hi5] = v[hi5], p[hi5], q[hi5]

        return torch.cat([r.unsqueeze(1), g.unsqueeze(1), b.unsqueeze(1)], dim=1)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = F.normalize(rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        k = F.normalize(rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = rearrange(attn @ v, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, _, C = x.shape
        x = self.dwconv(x.transpose(1, 2).view(B, C, H, W))
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class CMF(nn.Module):
    """Cross-Modal Fusion module."""

    def __init__(self, dim, num_heads, bias=False, LayerNorm_type='WithBias'):
        super(CMF, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm_S0 = LayerNorm(dim, LayerNorm_type)
        self.norm_P = LayerNorm(dim, LayerNorm_type)

        self.qkv_S0 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv_S0 = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.qkv_P = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv_P = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )

        self.mlp = MLP(dim * 2, dim)
        self.mlp_out = nn.Conv2d(dim * 2, 2, kernel_size=1, bias=bias)

    def forward(self, S0, P):
        b, _, h, w = S0.shape
        norm_S0 = self.norm_S0(S0)
        norm_P = self.norm_P(P)

        Q_S0, K_S0, V_S0 = self.qkv_dwconv_S0(self.qkv_S0(norm_S0)).chunk(3, dim=1)
        Q_P, K_P, V_P = self.qkv_dwconv_P(self.qkv_P(norm_P)).chunk(3, dim=1)

        Q_S0 = F.normalize(rearrange(Q_S0, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        K_S0 = F.normalize(rearrange(K_S0, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        V_S0 = rearrange(V_S0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        Q_P = F.normalize(rearrange(Q_P, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        K_P = F.normalize(rearrange(K_P, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        V_P = rearrange(V_P, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn_P_S0 = ((Q_P @ K_S0.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out_P_S0 = rearrange(attn_P_S0 @ V_S0, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        attn_S0_P = ((Q_S0 @ K_P.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out_S0_P = rearrange(attn_S0_P @ V_P, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        cat_out = torch.cat([out_P_S0, out_S0_P], dim=1).flatten(2).transpose(1, 2)
        cat_out = self.mlp(cat_out, h, w).transpose(1, 2).view(b, -1, h, w)
        weights = F.softmax(self.mlp_out(cat_out), dim=1)
        return S0 * weights[:, 0:1, :, :] + P * weights[:, 1:2, :, :]


CMFModule = CMF


class SingleChannelCrossAttention(nn.Module):
    def __init__(self, dim=32, num_heads=4, bias=False):
        super(SingleChannelCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.embed_x = nn.Conv2d(1, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.embed_y = nn.Conv2d(1, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.qkv_A = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv_A = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.qkv_B = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv_B = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )

        self.project_out_A = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out_B = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.concat_conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.output_conv = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x, y):
        _, _, h, w = x.shape
        x_emb = self.embed_x(x)
        y_emb = self.embed_y(y)

        q_a, k_a, v_a = self.qkv_dwconv_A(self.qkv_A(x_emb)).chunk(3, dim=1)
        q_b, k_b, v_b = self.qkv_dwconv_B(self.qkv_B(y_emb)).chunk(3, dim=1)

        q_a = F.normalize(rearrange(q_a, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        k_a = F.normalize(rearrange(k_a, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        v_a = rearrange(v_a, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_b = F.normalize(rearrange(q_b, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        k_b = F.normalize(rearrange(k_b, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        v_b = rearrange(v_b, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn_a = ((q_b @ k_a.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out_a = self.project_out_A(
            rearrange(attn_a @ v_a, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        ) + x_emb

        attn_b = ((q_a @ k_b.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out_b = self.project_out_B(
            rearrange(attn_b @ v_b, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        ) + y_emb

        return self.output_conv(self.concat_conv(torch.cat([out_a, out_b], dim=1)))


class SSCF(nn.Module):
    """Shallow Spectral-Channel Fusion module."""

    def __init__(self, dim=32, num_heads=4, bias=False):
        super(SSCF, self).__init__()
        self.cross_attn_R = SingleChannelCrossAttention(dim=dim, num_heads=num_heads, bias=bias)
        self.cross_attn_G = SingleChannelCrossAttention(dim=dim, num_heads=num_heads, bias=bias)
        self.cross_attn_B = SingleChannelCrossAttention(dim=dim, num_heads=num_heads, bias=bias)

    def forward(self, x0, dolp):
        out_R = self.cross_attn_R(x0[:, 0:1], dolp[:, 0:1])
        out_G = self.cross_attn_G(x0[:, 1:2], dolp[:, 1:2])
        out_B = self.cross_attn_B(x0[:, 2:3], dolp[:, 2:3])
        return torch.cat([out_R, out_G, out_B], dim=1)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_query, dim_key_value, num_heads=4, bias=False):
        super(CrossAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim_query, dim_query, 1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim_query, dim_query, 3, 1, 1, groups=dim_query, bias=bias)
        self.kv = nn.Conv2d(dim_key_value, dim_query * 2, 1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim_query * 2, dim_query * 2, 3, 1, 1, groups=dim_query * 2, bias=bias)
        self.project_out = nn.Conv2d(dim_query, dim_query, 1, bias=bias)

    def forward(self, query, key_value):
        _, _, h, w = query.shape
        q = self.q_dwconv(self.q(query))
        k, v = self.kv_dwconv(self.kv(key_value)).chunk(2, dim=1)
        q = F.normalize(rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        k = F.normalize(rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = rearrange(attn @ v, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class ICIF(nn.Module):
    """Intensity-Chrominance Interaction Fusion module."""

    def __init__(self, dim_I, dim_HV, num_heads=4, bias=False):
        super(ICIF, self).__init__()
        self.cross_attn_I_from_HV = CrossAttentionBlock(dim_query=dim_I, dim_key_value=dim_HV, num_heads=num_heads, bias=bias)
        self.cross_attn_HV_from_I = CrossAttentionBlock(dim_query=dim_HV, dim_key_value=dim_I, num_heads=num_heads, bias=bias)
        self.gate_I = nn.Sequential(nn.Conv2d(dim_I * 2, dim_I, 1, bias=bias), nn.Sigmoid())
        self.gate_HV = nn.Sequential(nn.Conv2d(dim_HV * 2, dim_HV, 1, bias=bias), nn.Sigmoid())

    def forward(self, feat_I, feat_HV):
        cross_info_to_I = self.cross_attn_I_from_HV(query=feat_I, key_value=feat_HV)
        cross_info_to_HV = self.cross_attn_HV_from_I(query=feat_HV, key_value=feat_I)
        gate_I = self.gate_I(torch.cat([feat_I, cross_info_to_I], dim=1))
        gate_HV = self.gate_HV(torch.cat([feat_HV, cross_info_to_HV], dim=1))
        return cross_info_to_I * gate_I, cross_info_to_HV * gate_HV


class HVI_Transform(nn.Module):
    def __init__(self):
        super(HVI_Transform, self).__init__()
        self.rgb_hvi = RGB_HVI()

    def to_hvi(self, rgb):
        return self.rgb_hvi.HVIT(rgb)

    def to_rgb(self, hvi):
        return self.rgb_hvi.PHVIT(hvi)

    def split_hvi(self, hvi):
        return hvi[:, 2:3, :, :], hvi[:, 0:2, :, :]

    def merge_hvi(self, HV, I):
        return torch.cat([HV, I], dim=1)


class Up_scale_pixel(nn.Module):
    def __init__(self, n_feat):
        super(Up_scale_pixel, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class EncoderTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(EncoderTransformerBlock, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.downsampled = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        x_ = x
        return self.lrelu(self.downsampled(x)), x_


class DecoderTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(DecoderTransformerBlock, self).__init__()
        self.upsampled = Up_scale_pixel(dim)
        self.concat_conv = nn.Conv2d(dim * 3 // 2, dim // 2, kernel_size=1)
        self.norm1 = LayerNorm(dim // 2, LayerNorm_type)
        self.attn = Attention(dim // 2, num_heads, bias)
        self.norm2 = LayerNorm(dim // 2, LayerNorm_type)
        self.ffn = FeedForward(dim // 2, ffn_expansion_factor, bias)

    def forward(self, x, x_inter, y_inter):
        x = self.concat_conv(torch.concat([x_inter, y_inter, self.upsampled(x)], dim=1))
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PRFNet(nn.Module):
    """Polarization-Retinex Fusion Network (Stage 1 PRDNet + Stage 2 DPFNet)."""

    def __init__(self, in_chans=1, embed_dim=96, Ex_depths=2, Fusion_depths=1, Re_depths=2,
                 Ex_num_heads=8, Fusion_num_heads=8, Re_num_heads=8, img_range=1.,
                 use_prdnet=True, use_sscf=True):
        super(PRFNet, self).__init__()
        self.img_range = img_range
        self.in_chans = in_chans
        self.use_prdnet = use_prdnet
        self.use_sscf = use_sscf
        self.Ex_num_layers = Ex_depths
        self.Fusion_num_layers = Fusion_depths
        self.Re_num_layers = Re_depths
        self.embed_dim = embed_dim
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.mean = torch.zeros(1, in_chans, 1, 1)
        embed_dim_temp = int(embed_dim / 2)

        if self.use_prdnet:
            self.prdnet = PRDNet(channels=in_chans)

        if self.use_sscf:
            self.sscf = SSCF(dim=32, num_heads=4, bias=False)

        self.hvi_transform = HVI_Transform()

        # DICB: Intensity Branch (I-Branch)
        self.conv_first1_I = nn.Conv2d(1, embed_dim_temp, 3, 1, 1)
        self.conv_first2_I = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.layers_Ex_I = nn.ModuleList([
            EncoderTransformerBlock(dim=embed_dim * (i + 1), num_heads=Ex_num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
            for i in range(self.Ex_num_layers)
        ])
        self.cmf_I = nn.ModuleList([
            CMF(dim=embed_dim * self.Ex_num_layers * 2, num_heads=Fusion_num_heads, bias=False, LayerNorm_type='WithBias')
            for _ in range(self.Fusion_num_layers)
        ])
        self.layers_Reconstruction_I = nn.ModuleList([
            DecoderTransformerBlock(dim=embed_dim * (self.Ex_num_layers - i) * 2, num_heads=Re_num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
            for i in range(self.Re_num_layers)
        ])
        self.conv_last1_I = nn.Conv2d(embed_dim, embed_dim_temp, 3, 1, 1)
        self.conv_last2_I = nn.Conv2d(embed_dim_temp, 1, 3, 1, 1)

        # DICB: Chrominance Branch (HV-Branch)
        self.conv_first1_HV = nn.Conv2d(2, embed_dim_temp, 3, 1, 1)
        self.conv_first2_HV = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.layers_Ex_HV = nn.ModuleList([
            EncoderTransformerBlock(dim=embed_dim * (i + 1), num_heads=Ex_num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
            for i in range(self.Ex_num_layers)
        ])
        self.cmf_HV = nn.ModuleList([
            CMF(dim=embed_dim * self.Ex_num_layers * 2, num_heads=Fusion_num_heads, bias=False, LayerNorm_type='WithBias')
            for _ in range(self.Fusion_num_layers)
        ])
        self.layers_Reconstruction_HV = nn.ModuleList([
            DecoderTransformerBlock(dim=embed_dim * (self.Ex_num_layers - i) * 2, num_heads=Re_num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
            for i in range(self.Re_num_layers)
        ])
        self.conv_last1_HV = nn.Conv2d(embed_dim, embed_dim_temp, 3, 1, 1)
        self.conv_last2_HV = nn.Conv2d(embed_dim_temp, 2, 3, 1, 1)

        # SSCF + HVI Spatial Attention Reconstruction Head
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        # ICIF Cross-Branch Interaction Modules
        self.icif_enc = nn.ModuleList([
            ICIF(dim_I=embed_dim * (i + 1) * 2, dim_HV=embed_dim * (i + 1) * 2, num_heads=4, bias=False)
            for i in range(self.Ex_num_layers)
        ])
        dim_bottleneck = embed_dim * self.Ex_num_layers * 2
        self.icif_bottleneck = ICIF(
            dim_I=dim_bottleneck, dim_HV=dim_bottleneck, num_heads=4, bias=False
        )
        self.icif_dec = nn.ModuleList([
            ICIF(dim_I=embed_dim * (self.Ex_num_layers - i), dim_HV=embed_dim * (self.Ex_num_layers - i), num_heads=4, bias=False)
            for i in range(self.Re_num_layers)
        ])

        self.encoder_bridges_x_y_I = nn.ModuleList([
            CrossAttentionBlock(dim_query=embed_dim * (i + 1) * 2, dim_key_value=embed_dim * (i + 1) * 2, num_heads=4, bias=False)
            for i in range(self.Ex_num_layers)
        ])
        self.encoder_bridges_x_y_HV = nn.ModuleList([
            CrossAttentionBlock(dim_query=embed_dim * (i + 1) * 2, dim_key_value=embed_dim * (i + 1) * 2, num_heads=4, bias=False)
            for i in range(self.Ex_num_layers)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, A, B, I_max=None, I_min=None, I_theta=None, I_random=None,
                return_retinex_intermediate=False, fusion_branch_input='i_opt'):
        retinex_intermediate = None
        if self.use_prdnet and I_max is not None and I_min is not None:
            if I_max.max() > 1.0:
                I_max = I_max / 255.0
            if I_min.max() > 1.0:
                I_min = I_min / 255.0

            I_opt, Rhat, L_max, L_min, L_random, R_refined = self.prdnet(
                I_max, I_min, I_theta=I_theta, I_random=I_random
            )
            if return_retinex_intermediate:
                retinex_intermediate = {
                    'I_opt': I_opt,
                    'Rhat': Rhat,
                    'L_max': L_max,
                    'L_min': L_min,
                    'L_random': L_random,
                    'R_refined': R_refined,
                    'I_max': I_max,
                    'I_min': I_min,
                    'I_theta': I_theta,
                    'I_random': I_random,
                }

            fb = (fusion_branch_input or 'i_opt').strip().lower()
            if fb == 'rhat':
                x0 = Rhat
            elif fb in ('i_opt', 'iopt'):
                x0 = I_opt
            else:
                raise ValueError(f"fusion_branch_input must be 'i_opt' or 'rhat', got {fusion_branch_input!r}")
        else:
            x0 = A
            Rhat = x0

        spectral_fused = None
        if self.use_sscf:
            spectral_fused = self.sscf(x0, B)
            if return_retinex_intermediate:
                if retinex_intermediate is None:
                    retinex_intermediate = {}
                retinex_intermediate['spectral_fused'] = spectral_fused

        # DICB: HVI Dual-Branch Deep Fusion
        x0_hvi = self.hvi_transform.to_hvi(x0)
        B_hvi = self.hvi_transform.to_hvi(B)
        x0_I, x0_HV = self.hvi_transform.split_hvi(x0_hvi)
        B_I, B_HV = self.hvi_transform.split_hvi(B_hvi)

        self.mean_A_I = self.mean[:, 2:3, :, :].type_as(x0_I)
        self.mean_B_I = self.mean[:, 2:3, :, :].type_as(B_I)
        self.mean_I = (self.mean_A_I + self.mean_B_I) / 2

        self.mean_A_HV = self.mean[:, 0:2, :, :].type_as(x0_HV)
        self.mean_B_HV = self.mean[:, 0:2, :, :].type_as(B_HV)
        self.mean_HV = (self.mean_A_HV + self.mean_B_HV) / 2

        x0_I = (x0_I - self.mean_A_I) * self.img_range
        B_I = (B_I - self.mean_B_I) * self.img_range
        x0_HV = (x0_HV - self.mean_A_HV) * self.img_range
        B_HV = (B_HV - self.mean_B_HV) * self.img_range

        x_I = self.lrelu(self.conv_first2_I(self.lrelu(self.conv_first1_I(x0_I))))
        y_I = self.lrelu(self.conv_first2_I(self.lrelu(self.conv_first1_I(B_I))))
        x_HV = self.lrelu(self.conv_first2_HV(self.lrelu(self.conv_first1_HV(x0_HV))))
        y_HV = self.lrelu(self.conv_first2_HV(self.lrelu(self.conv_first1_HV(B_HV))))

        inter_x_I, inter_y_I, inter_x_HV, inter_y_HV = [], [], [], []
        for i, (layer_Ex_I, layer_Ex_HV, icif_module) in enumerate(
            zip(self.layers_Ex_I, self.layers_Ex_HV, self.icif_enc)
        ):
            x_I, x_I_ = layer_Ex_I(x_I)
            y_I, y_I_ = layer_Ex_I(y_I)
            x_HV, x_HV_ = layer_Ex_HV(x_HV)
            y_HV, y_HV_ = layer_Ex_HV(y_HV)

            enhanced_x_I, enhanced_x_HV = icif_module(x_I, x_HV)
            x_I = x_I + enhanced_x_I
            x_HV = x_HV + enhanced_x_HV

            enhanced_y_I, enhanced_y_HV = icif_module(y_I, y_HV)
            y_I = y_I + enhanced_y_I
            y_HV = y_HV + enhanced_y_HV

            x_I = x_I + 0.1 * self.encoder_bridges_x_y_I[i](x_I, y_I)
            x_HV = x_HV + 0.1 * self.encoder_bridges_x_y_HV[i](x_HV, y_HV)

            inter_x_I.append(x_I_)
            inter_y_I.append(y_I_)
            inter_x_HV.append(x_HV_)
            inter_y_HV.append(y_HV_)

        for cmf_layer_I in self.cmf_I:
            x_I = cmf_layer_I(x_I, y_I)
        for cmf_layer_HV in self.cmf_HV:
            x_HV = cmf_layer_HV(x_HV, y_HV)

        enhanced_x_I, enhanced_x_HV = self.icif_bottleneck(x_I, x_HV)
        x_I = x_I + enhanced_x_I
        x_HV = x_HV + enhanced_x_HV

        for cnt_Rec, (layer_Rec_I, layer_Rec_HV, icif_module) in enumerate(
            zip(self.layers_Reconstruction_I, self.layers_Reconstruction_HV, self.icif_dec)
        ):
            idx = 1 - cnt_Rec
            x_I = layer_Rec_I(x_I, inter_x_I[idx], inter_y_I[idx])
            x_HV = layer_Rec_HV(x_HV, inter_x_HV[idx], inter_y_HV[idx])

            enhanced_x_I, enhanced_x_HV = icif_module(x_I, x_HV)
            x_I = x_I + enhanced_x_I
            x_HV = x_HV + enhanced_x_HV

        x_I = self.conv_last2_I(self.lrelu(self.conv_last1_I(x_I))) / self.img_range + self.mean_I
        x_HV = self.conv_last2_HV(self.lrelu(self.conv_last1_HV(x_HV))) / self.img_range + self.mean_HV

        fused_hvi = self.hvi_transform.merge_hvi(x_HV, x_I)
        hvi_fused_rgb = self.hvi_transform.to_rgb(fused_hvi)

        if spectral_fused is not None:
            alpha_map = self.fusion_conv(torch.cat([spectral_fused, hvi_fused_rgb], dim=1))
            x = alpha_map * spectral_fused + (1 - alpha_map) * hvi_fused_rgb
        else:
            x = hvi_fused_rgb

        I_final = torch.clamp(x, 0, 1)

        if return_retinex_intermediate and retinex_intermediate is not None:
            retinex_intermediate['hvi_fused_rgb'] = hvi_fused_rgb
            retinex_intermediate['I_final'] = I_final
            retinex_intermediate['hvi_components'] = {
                'x0_hvi': x0_hvi,
                'B_hvi': B_hvi,
                'fused_I': x_I,
                'fused_HV': x_HV,
                'fused_hvi': fused_hvi,
            }
            return I_final, Rhat, retinex_intermediate

        return I_final, Rhat


DPFNet = PRFNet
