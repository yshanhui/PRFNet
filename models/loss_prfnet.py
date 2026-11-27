"""
loss_prfnet.py

PRFNet Loss Functions
Stage 1 (PRDNet): L_rec (recon), L_Rhat (consist + grad_polar), L_reg (suppress + initialize)
Stage 2 (DPFNet): L_grad (gradient), L_color (color), L_per (perceptual)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiGrad(nn.Module):
    def __init__(self):
        super(MultiGrad, self).__init__()
        dx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        dy = torch.tensor([[-1, 2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        dxx = torch.tensor([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=torch.float32)
        dyy = torch.tensor([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=torch.float32)
        dxy = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)

        self.register_buffer('dx', dx.unsqueeze(0).unsqueeze(0))
        self.register_buffer('dy', dy.unsqueeze(0).unsqueeze(0))
        self.register_buffer('dxx', dxx.unsqueeze(0).unsqueeze(0))
        self.register_buffer('dyy', dyy.unsqueeze(0).unsqueeze(0))
        self.register_buffer('dxy', dxy.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        _, C, _, _ = x.shape
        grad_list = []
        for c in range(C):
            xc = x[:, c:c + 1, :, :]
            Fx = F.conv2d(xc, self.dx, padding=1)
            Fy = F.conv2d(xc, self.dy, padding=1)
            Fxx = F.conv2d(xc, self.dxx, padding=1)
            Fyy = F.conv2d(xc, self.dyy, padding=1)
            Fxy = F.conv2d(xc, self.dxy, padding=1)
            grad_list.append(torch.cat((Fx, Fy, Fxx, Fxy, Fyy), dim=1))
        return torch.cat(grad_list, dim=1)


def color_angle_loss(img1, img2, eps=1e-6):
    B, C, _, _ = img1.shape
    img1_flat = img1.view(B, C, -1)
    img2_flat = img2.view(B, C, -1)
    total_angle_loss = 0.0

    for c in range(C):
        c1 = img1_flat[:, c, :]
        c2 = img2_flat[:, c, :]
        dot_product = torch.sum(c1 * c2, dim=1)
        norm1 = torch.sqrt(torch.sum(c1 ** 2, dim=1) + eps)
        norm2 = torch.sqrt(torch.sum(c2 ** 2, dim=1) + eps)
        cos_sim = torch.clamp(dot_product / (norm1 * norm2 + eps), -0.99999, 0.99999)
        angle = torch.acos(cos_sim)
        if torch.isnan(angle).any():
            angle = torch.where(torch.isnan(angle), torch.zeros_like(angle), angle)
        total_angle_loss += torch.mean(angle)

    return total_angle_loss / 3.0


def resolve_iopt_rhat_reference(mode, I_opt, Rhat):
    m = (mode or 'auto').strip().lower()
    if m in ('iopt', 'i_opt'):
        m = 'i_opt'
    if m == 'rhat':
        return Rhat
    if m in ('i_opt', 'auto'):
        return I_opt if I_opt is not None else Rhat
    raise ValueError(f"ref mode must be 'auto', 'i_opt' (or 'iopt'), or 'rhat', got {mode!r}")


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.conv3 = nn.Sequential(*list(vgg16.features.children())[:16])
        self.conv4 = nn.Sequential(*list(vgg16.features.children())[:23])
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_norm = (x - mean) / std
        return self.conv3(x_norm), self.conv4(x_norm)


def perceptual_loss(original_image, generated_image, vgg_extractor):
    with torch.no_grad() if not vgg_extractor.training else torch.enable_grad():
        orig_conv3, orig_conv4 = vgg_extractor(original_image)
        gen_conv3, gen_conv4 = vgg_extractor(generated_image)
    return F.l1_loss(gen_conv3, orig_conv3) + F.l1_loss(gen_conv4, orig_conv4)


def _sanitize_loss_terms(loss_terms, dev, dtype):
    for name, loss_val in loss_terms.items():
        if torch.isnan(loss_val).any() or torch.isinf(loss_val).any():
            print(f"⚠️  WARNING: {name} loss contains NaN or Inf!")
            loss_terms[name] = torch.tensor(0.0, device=dev, dtype=dtype)


def _sanitize_total_loss(total_loss, dev, dtype):
    if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
        print("⚠️  WARNING: Total loss contains NaN or Inf!")
        return torch.where(
            torch.isnan(total_loss) | torch.isinf(total_loss),
            torch.tensor(1.0, device=dev, dtype=dtype),
            total_loss,
        )
    return total_loss


class PRFNetLoss(nn.Module):
    def __init__(self,
                 w_recon=1.0,
                 w_initialize=1.0,
                 w_suppress=1.0,
                 w_gradient=1.0,
                 w_color=1.0,
                 color_loss_ref='auto',
                 w_consist=0.0,
                 w_perceptual=0.0,
                 perceptual_ref='i_opt',
                 w_grad_polar=0.0,
                 grad_polar_k=10.0):
        super(PRFNetLoss, self).__init__()
        self.w_recon = w_recon
        self.w_initialize = w_initialize
        self.w_suppress = w_suppress
        self.w_gradient = w_gradient
        self.w_color = w_color

        clr = (color_loss_ref or 'auto').strip().lower()
        if clr == 'iopt':
            clr = 'i_opt'
        if clr not in ('auto', 'i_opt', 'rhat'):
            raise ValueError(f"color_loss_ref must be 'auto', 'i_opt', or 'rhat', got {color_loss_ref!r}")
        self.color_loss_ref = clr

        self.w_consist = w_consist
        self.w_perceptual = w_perceptual
        pref = (perceptual_ref or 'i_opt').strip().lower()
        if pref == 'iopt':
            pref = 'i_opt'
        if pref not in ('i_opt', 'rhat'):
            raise ValueError(f"perceptual_ref must be 'i_opt' (or 'iopt') or 'rhat', got {perceptual_ref!r}")
        self.perceptual_ref = pref

        self.w_grad_polar = w_grad_polar
        self.grad_polar_k = grad_polar_k

        self.gradconv = MultiGrad()
        self.vgg_extractor = VGG16FeatureExtractor() if self.w_perceptual > 0 else None

    def _compute_gradient_loss(self, I_final, I_opt, DOLP, I_random, z):
        if self.w_gradient <= 0:
            return z
        if I_opt is not None and DOLP is not None:
            gradient_I_opt = self.gradconv(I_opt)
            gradient_DoLP = self.gradconv(DOLP)
            gradient_target = torch.max(gradient_I_opt.abs(), gradient_DoLP.abs()) * \
                              torch.sign(gradient_I_opt + gradient_DoLP)
            return F.huber_loss(self.gradconv(I_final), gradient_target, reduction='mean', delta=0.1)
        return F.l1_loss(self.gradconv(I_final), self.gradconv(I_random))

    def _compute_color_loss(self, I_final, I_opt, Rhat, z):
        if self.w_color <= 0:
            return z
        return color_angle_loss(I_final, resolve_iopt_rhat_reference(self.color_loss_ref, I_opt, Rhat))

    def _compute_perceptual_loss(self, I_final, I_opt, Rhat, z):
        if self.w_perceptual <= 0 or self.vgg_extractor is None:
            return z
        ref_perc = resolve_iopt_rhat_reference(self.perceptual_ref, I_opt, Rhat)
        if ref_perc is None:
            return z
        return perceptual_loss(ref_perc, I_final, self.vgg_extractor)

    def forward(self,
                I_max, I_min,
                L_max, L_min,
                I_random,
                L_random,
                I_final,
                Rhat,
                DOLP=None,
                I_opt=None,
                R_refined=None,
                fusion_only=False):
        dev = I_final.device
        dtype = I_final.dtype
        z = torch.tensor(0.0, device=dev, dtype=dtype)

        if fusion_only:
            loss_terms = {
                'gradient': self._compute_gradient_loss(I_final, I_opt, DOLP, I_random, z),
                'color': self._compute_color_loss(I_final, I_opt, Rhat, z),
                'perceptual': self._compute_perceptual_loss(I_final, I_opt, Rhat, z),
            }
            _sanitize_loss_terms(loss_terms, dev, dtype)
            total_loss = (
                self.w_gradient * loss_terms['gradient'] +
                self.w_color * loss_terms['color'] +
                self.w_perceptual * loss_terms['perceptual']
            )
            total_loss = _sanitize_total_loss(total_loss, dev, dtype)
            loss_terms['total'] = total_loss
            return total_loss, loss_terms

        if R_refined is None:
            raise ValueError("R_refined is None! Make sure use_pde=True in PRDNet")

        L_random_expanded = L_random.repeat(1, 3, 1, 1)
        loss_recon = F.l1_loss(L_random_expanded * R_refined, I_random)
        loss_initialize = F.l1_loss(L_random, torch.max(I_random, 1, keepdim=True)[0])

        L_max_expanded = L_max.repeat(1, 3, 1, 1)
        L_min_expanded = L_min.repeat(1, 3, 1, 1)
        loss_suppress = torch.mean(
            F.relu(Rhat * L_max_expanded.detach() - I_max) +
            F.relu(Rhat * L_min_expanded.detach() - I_min) +
            F.relu(Rhat * L_random_expanded.detach() - I_random)
        )

        loss_gradient = self._compute_gradient_loss(I_final, I_opt, DOLP, I_random, z)

        if self.w_grad_polar > 0 and DOLP is not None:
            grad_I_min = self.gradconv(I_min).detach()
            grad_I_max = self.gradconv(I_max).detach()
            dolp_w = DOLP.mean(dim=1, keepdim=True) if DOLP.size(1) > 1 else DOLP
            Wmap = torch.sigmoid(self.grad_polar_k * dolp_w)
            W_exp = Wmap.expand(-1, grad_I_min.shape[1], -1, -1)
            grad_ideal = W_exp * grad_I_min + (1.0 - W_exp) * grad_I_max
            loss_grad_polar = F.l1_loss(self.gradconv(Rhat), grad_ideal)
        else:
            loss_grad_polar = z

        loss_color = self._compute_color_loss(I_final, I_opt, Rhat, z)
        loss_consist = F.l1_loss(Rhat, R_refined) if (self.w_consist > 0 and R_refined is not None) else z
        loss_perceptual = self._compute_perceptual_loss(I_final, I_opt, Rhat, z)

        loss_terms = {
            'recon': loss_recon,
            'initialize': loss_initialize,
            'suppress': loss_suppress,
            'gradient': loss_gradient,
            'grad_polar': loss_grad_polar,
            'color': loss_color,
            'consist': loss_consist,
            'perceptual': loss_perceptual,
        }
        _sanitize_loss_terms(loss_terms, dev, dtype)

        total_loss = (
            self.w_recon * loss_terms['recon'] +
            self.w_initialize * loss_terms['initialize'] +
            self.w_suppress * loss_terms['suppress'] +
            self.w_gradient * loss_terms['gradient'] +
            self.w_grad_polar * loss_terms['grad_polar'] +
            self.w_color * loss_terms['color'] +
            self.w_consist * loss_terms['consist'] +
            self.w_perceptual * loss_terms['perceptual']
        )
        total_loss = _sanitize_total_loss(total_loss, dev, dtype)
        loss_terms['total'] = total_loss
        return total_loss, loss_terms


PolarizationRetinexLoss = PRFNetLoss
