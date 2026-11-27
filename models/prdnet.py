"""
PRDNet (Polarimetric Retinex Decomposition Network)
Includes PRE, AIE, PDE, RTB, AdaptiveIlluminationFusion, and legacy checkpoint compatibility.
"""
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


LEGACY_PRDNET_KEY_MAP = (
    ('L_net.L_net.', 'aie.net.'),
    ('R_net.', 'pre.'),
    ('r_refiner.', 'pde.'),
    ('fusion.', 'illum_fusion.'),
)


def remap_prdnet_key(key):
    """Map legacy P-RAEG state_dict keys to PRDNet module names."""
    for old_prefix, new_prefix in LEGACY_PRDNET_KEY_MAP:
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):]
    return key


def load_prdnet_state_dict(model, checkpoint, logger=None):
    """Load PRDNet weights from current or legacy P-RAEG checkpoints with strict verification."""
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        sd = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'prdnet' in checkpoint:
        sd = checkpoint['prdnet']
    elif isinstance(checkpoint, dict) and 'p_raeg' in checkpoint:
        sd = checkpoint['p_raeg']
    else:
        sd = checkpoint

    remapped = {}
    remapped_count = 0
    for k, v in sd.items():
        nk = remap_prdnet_key(k)
        if nk != k:
            remapped_count += 1
        remapped[nk] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"PRDNet state dict mismatch: missing={missing}, unexpected={unexpected}")
    if remapped_count and logger is not None:
        logger.info(f"   ✓ 自动映射 {remapped_count} 个历史 PRDNet 权重键")


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
    def __init__(self, dim):
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
        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        k = self.k_dwconv(self.k(x))
        v = self.v_dwconv(self.v(x))

        q = F.normalize(rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        k = F.normalize(rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = rearrange(attn @ v, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class RTB(nn.Module):
    """Restormer Transformer Block."""

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(RTB, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PRE(nn.Module):
    """Polarization Reflectance Estimator."""

    def __init__(self, out_channels=3, dim=32, heads=(8, 8, 8), ffn_expansion_factor=2, bias=False):
        super(PRE, self).__init__()
        self.patch_embed1 = nn.Conv2d(6, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.encoder = nn.Sequential(*[
            RTB(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(2)
        ])
        self.output1 = nn.Sequential(
            nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )
        self.act = nn.Sigmoid()

    def forward(self, I_max, I_min):
        R = self.encoder(self.patch_embed1(torch.cat([I_max, I_min], dim=1)))
        Rhat = self.act(self.output1(R))
        return Rhat, R


class AIE(nn.Module):
    """Adaptive Illumination Estimator."""

    def __init__(self, num=64):
        super(AIE, self).__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class PDE(nn.Module):
    """Polarimetric Degradation Estimator."""

    def __init__(self, channels=3, hidden_dim=32, num_heads=4, ffn_expansion_factor=2, bias=False):
        super(PDE, self).__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim

        self.stage1_conv = nn.Conv2d(6, hidden_dim, kernel_size=3, padding=1, bias=bias)
        self.stage1_rtb = RTB(
            dim=hidden_dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias
        )

        self.stage2_proj = nn.Conv2d(hidden_dim + 1, hidden_dim, kernel_size=1, bias=bias)
        self.stage2_rtb = RTB(
            dim=hidden_dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias
        )

        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1, bias=bias),
            nn.Tanh()
        )
        self.delta_scale = 0.2

    def forward(self, Rhat, I_theta, I_min, L_random):
        diff = I_theta - I_min
        x = self.stage1_rtb(self.stage1_conv(torch.cat([diff, Rhat], dim=1)))
        x = self.stage2_rtb(self.stage2_proj(torch.cat([x, L_random], dim=1)))
        delta_R = self.output(x) * self.delta_scale
        return torch.clamp(Rhat + delta_R, 0, 1)


class AdaptiveIlluminationFusion(nn.Module):
    def __init__(self, init_k=0.5):
        super(AdaptiveIlluminationFusion, self).__init__()
        self.register_buffer('k', torch.tensor(init_k))

    def forward(self, L_max, L_min):
        L_mean = (L_max + L_min) / 2.0
        k_val = torch.clamp(self.k, 0.1, 0.9)
        return (k_val * L_mean) / (k_val * L_mean + (1 - k_val) * (1 - L_mean) + 1e-6)


class PRDNet(nn.Module):
    """Polarimetric Retinex Decomposition Network (Stage 1)."""

    def __init__(self, channels=3, use_pde=True):
        super(PRDNet, self).__init__()
        self.pre = PRE(out_channels=channels, dim=32)
        self.aie = AIE(num=64)
        self.illum_fusion = AdaptiveIlluminationFusion(init_k=0.5)

        self.use_pde = use_pde
        if use_pde:
            self.pde = PDE(
                channels=3, hidden_dim=32, num_heads=4, ffn_expansion_factor=2, bias=False
            )

    def forward(self, I_max, I_min, I_theta=None, I_random=None):
        Rhat, _ = self.pre(I_max, I_min)
        L_max = self.aie(I_max)
        L_min = self.aie(I_min)
        L_random = self.aie(I_random) if I_random is not None else (L_max + L_min) / 2.0

        if self.use_pde and I_theta is not None:
            R_refined = self.pde(Rhat, I_theta, I_min, L_random)
        else:
            R_refined = None

        L_opt = self.illum_fusion(L_max, L_min)
        I_opt = L_opt * Rhat
        return I_opt, Rhat, L_max, L_min, L_random, R_refined
