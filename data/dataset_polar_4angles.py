"""
四角度偏振数据集 (0°, 45°, 90°, 135°)
计算 Stokes 参数 (S0, DoLP, AoP)、I_max、I_min 及随机角度 Malus 重建 I_theta / I_random。
"""
import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util


def calculate_polarization(img0, img45, img90, img135):
    eps = 1e-6
    if img0.max() > 1.5:
        img0 = img0 / 255.0
        img45 = img45 / 255.0
        img90 = img90 / 255.0
        img135 = img135 / 255.0

    S0 = img0 + img90
    S1 = img0 - img90
    S2 = img45 - img135
    Ip = np.sqrt(S1 ** 2 + S2 ** 2)
    DoLP = np.clip(Ip / (S0 + eps), 0, 1)
    AoP = 0.5 * np.arctan2(S2, S1)
    return S0, Ip, DoLP, AoP


def reconstruct_intensity(S0, Ip, AoP, theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    intensity = 0.5 * (S0 + Ip * np.cos(2 * (AoP - theta_rad)))
    return np.clip(intensity, 0, 1)


def calculate_I_max_I_min(S0, Ip):
    I_max = np.clip(0.5 * (S0 + Ip), 0, 1)
    I_min = np.clip(0.5 * (S0 - Ip), 0, 1)
    return I_max, I_min


class DatasetPolar4Angles(data.Dataset):
    def __init__(self, opt):
        super(DatasetPolar4Angles, self).__init__()
        self.opt = opt
        self.n_channels = opt.get('n_channels', 3)
        self.patch_size = opt.get('H_size', 128)

        self.dataroot_0 = opt['dataroot_0deg']
        self.dataroot_45 = opt['dataroot_45deg']
        self.dataroot_90 = opt['dataroot_90deg']
        self.dataroot_135 = opt['dataroot_135deg']

        self.paths_0 = util.get_image_paths(self.dataroot_0)
        self.paths_45 = util.get_image_paths(self.dataroot_45)
        self.paths_90 = util.get_image_paths(self.dataroot_90)
        self.paths_135 = util.get_image_paths(self.dataroot_135)

        assert len(self.paths_0) == len(self.paths_45) == len(self.paths_90) == len(self.paths_135), \
            f"数据集数量不一致！0°: {len(self.paths_0)}, 45°: {len(self.paths_45)}, " \
            f"90°: {len(self.paths_90)}, 135°: {len(self.paths_135)}"

        self.theta_min = opt.get('theta_min', 0)
        self.theta_max = opt.get('theta_max', 180)

    def __getitem__(self, index):
        path_0 = self.paths_0[index]
        path_45 = self.paths_45[index]
        path_90 = self.paths_90[index]
        path_135 = self.paths_135[index]

        img_0_uint = util.imread_uint(path_0, self.n_channels)
        img_45_uint = util.imread_uint(path_45, self.n_channels)
        img_90_uint = util.imread_uint(path_90, self.n_channels)
        img_135_uint = util.imread_uint(path_135, self.n_channels)

        if self.opt['phase'] == 'train':
            H, W, _ = img_0_uint.shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            ps = self.patch_size
            mode = random.randint(0, 7)

            img_0_uint = util.augment_img(img_0_uint[rnd_h:rnd_h + ps, rnd_w:rnd_w + ps, :], mode=mode)
            img_45_uint = util.augment_img(img_45_uint[rnd_h:rnd_h + ps, rnd_w:rnd_w + ps, :], mode=mode)
            img_90_uint = util.augment_img(img_90_uint[rnd_h:rnd_h + ps, rnd_w:rnd_w + ps, :], mode=mode)
            img_135_uint = util.augment_img(img_135_uint[rnd_h:rnd_h + ps, rnd_w:rnd_w + ps, :], mode=mode)

        S0, Ip, DoLP_norm, AoP = calculate_polarization(
            img_0_uint.astype(np.float32),
            img_45_uint.astype(np.float32),
            img_90_uint.astype(np.float32),
            img_135_uint.astype(np.float32),
        )
        I_max_norm, I_min_norm = calculate_I_max_I_min(S0, Ip)
        random_theta = random.uniform(self.theta_min, self.theta_max)
        I_theta_norm = reconstruct_intensity(S0, Ip, AoP, random_theta)

        img_I_max = util.uint2tensor3((I_max_norm * 255).clip(0, 255).astype(np.uint8))
        img_I_min = util.uint2tensor3((I_min_norm * 255).clip(0, 255).astype(np.uint8))
        img_I_theta = util.uint2tensor3((I_theta_norm * 255).clip(0, 255).astype(np.uint8))
        img_S0 = util.uint2tensor3((np.clip(S0, 0, 1) * 255).astype(np.uint8))
        img_DoLP = util.uint2tensor3((DoLP_norm * 255).clip(0, 255).astype(np.uint8))

        choice = random.randint(0, 2)
        if choice == 0:
            img_I_random = img_I_max
        elif choice == 1:
            img_I_random = img_I_min
        else:
            img_I_random = img_I_theta

        return {
            'I_max': img_I_max,
            'I_min': img_I_min,
            'I_theta': img_I_theta,
            'I_random': img_I_random,
            'S0': img_S0,
            'DoLP': img_DoLP,
            'I_max_path': path_0,
            'I_min_path': path_90,
        }

    def __len__(self):
        return len(self.paths_0)
