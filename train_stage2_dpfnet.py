"""
train_stage2_dpfnet.py
Stage 2: Train DPFNet within PRFNet with Frozen PRDNet (Fusion-Only Backprop)
"""
import os
import sys
import time
import random
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.prfnet import PRFNet
from models.prdnet import load_prdnet_state_dict
from models.loss_prfnet import PRFNetLoss
from data.dataset_polar_4angles import DatasetPolar4Angles
import utils.utils_option as option
import utils.utils_image as util
from utils.utils_logger import logger_info


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_module(module, freeze=True):
    for param in module.parameters():
        param.requires_grad = not freeze


def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def _basename_from_batch_path(batch_data, b):
    path_key = batch_data['I_max_path']
    p = path_key[b] if isinstance(path_key, (list, tuple)) else path_key
    return os.path.splitext(os.path.basename(p))[0]


def load_i_opt_batch_aligned(batch_data, i_opt_dir, ref_tensor, device):
    B, _, H, W = ref_tensor.shape
    if B == 0:
        raise ValueError('empty batch')
    chunks = []
    for b in range(B):
        base = _basename_from_batch_path(batch_data, b)
        found = None
        for ext in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'):
            cand = os.path.join(i_opt_dir, base + ext)
            if os.path.isfile(cand):
                found = cand
                break
        if found is None:
            raise FileNotFoundError(f"未在 {i_opt_dir} 找到与 {base} 同名的 I_opt 图像（尝试 .png/.jpg）")
        t = util.uint2tensor3(util.imread_uint(found, 3)).to(device).unsqueeze(0)
        if t.shape[2] != H or t.shape[3] != W:
            t = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)
        chunks.append(t)
    return torch.cat(chunks, dim=0)


def _sample_to_batch_b1(sample):
    out = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)
        elif k in ('I_max_path', 'I_min_path'):
            out[k] = [v]
        else:
            out[k] = v
    return out


def _resolve_test_batch(test_set, ref_path, logger):
    if not ref_path or not str(ref_path).strip():
        sample = test_set[0]
        return _sample_to_batch_b1(sample), os.path.splitext(os.path.basename(sample['I_max_path']))[0]

    ref_path = os.path.abspath(os.path.expanduser(str(ref_path).strip()))
    if not os.path.isfile(ref_path):
        logger.warning(f"test_reference_image_path 不存在: {ref_path}，改用 test_set[0]")
        sample = test_set[0]
        return _sample_to_batch_b1(sample), os.path.splitext(os.path.basename(sample['I_max_path']))[0]

    base = os.path.basename(ref_path)
    idx = next(
        (i for i, p in enumerate(test_set.paths_0)
         if os.path.abspath(p) == ref_path or os.path.basename(p).lower() == base.lower()),
        None
    )
    if idx is None:
        logger.warning(f"未在 test 集 0° 列表中找到与 {base} 匹配的条目，改用 test_set[0]")
        idx = 0
    sample = test_set[idx]
    stem = os.path.splitext(os.path.basename(sample['I_max_path']))[0]
    logger.info(f"   验证样本: {base} -> index {idx}")
    return _sample_to_batch_b1(sample), stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='options/train_stage2_dpfnet.json',
                        help='Path to option JSON file.')
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    tr = opt['train']

    use_disk_i_opt = tr.get('use_disk_i_opt', False)
    disk_i_opt_dir = tr.get('disk_i_opt_dir') or opt['path'].get('disk_i_opt_dir', '')
    if use_disk_i_opt:
        disk_i_opt_dir = os.path.expanduser(disk_i_opt_dir)
    enable_test_phase = tr.get('enable_test_phase', True)

    fusion_branch_input = (tr.get('fusion_branch_input') or 'i_opt').strip().lower()
    if fusion_branch_input == 'iopt':
        fusion_branch_input = 'i_opt'
    if fusion_branch_input not in ('i_opt', 'rhat'):
        raise ValueError(f"train.fusion_branch_input 须为 'i_opt' 或 'rhat'，当前: {tr.get('fusion_branch_input')!r}")

    setup_seed(tr['manual_seed'])
    for key in ['models', 'images', 'log']:
        os.makedirs(opt['path'][key], exist_ok=True)

    logger_name = 'train'
    logger_info(logger_name, log_path=os.path.join(opt['path']['log'], f'{logger_name}.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    use_sscf = opt['netG'].get('use_sscf', opt['netG'].get('use_shallow_spectral_fusion', True))
    pretrained_prdnet = opt['path'].get('pretrained_prdnet') or opt['path'].get('pretrained_praeg')
    freeze_prdnet = tr.get('freeze_prdnet', tr.get('freeze_praeg', True))

    logger.info(">>> Creating PRFNet model (PRDNet + DPFNet)...")
    model = PRFNet(
        in_chans=3,
        embed_dim=opt['netG']['embed_dim'],
        Ex_depths=opt['netG']['Ex_depths'],
        Fusion_depths=opt['netG']['Fusion_depths'],
        Re_depths=opt['netG']['Re_depths'],
        use_prdnet=True,
        use_sscf=use_sscf,
    )

    if pretrained_prdnet:
        logger.info(f"\n>>> Loading Stage 1 (PRDNet) checkpoint: {pretrained_prdnet}")
        checkpoint = torch.load(pretrained_prdnet)
        load_prdnet_state_dict(model.prdnet, checkpoint, logger)
        step_str = checkpoint.get('step', 'unknown') if isinstance(checkpoint, dict) else 'raw'
        logger.info(f"   ✓ Loaded PRDNet from Step {step_str}")
    else:
        logger.warning("⚠️  No Stage 1 (PRDNet) checkpoint specified! PRDNet will be randomly initialized.")

    if freeze_prdnet:
        freeze_module(model.prdnet, freeze=True)
        logger.info("   ✓ PRDNet frozen (requires_grad=False)")
    else:
        logger.info("   PRDNet NOT frozen (end-to-end training)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    num_total = count_parameters(model, only_trainable=False)
    num_trainable = count_parameters(model, only_trainable=True)
    num_prdnet = count_parameters(model.prdnet, only_trainable=False)
    logger.info(
        f"[OK] Model: total={num_total/1e6:.2f}M, trainable={num_trainable/1e6:.2f}M, "
        f"PRDNet={num_prdnet/1e6:.2f}M (frozen={freeze_prdnet})"
    )

    if use_disk_i_opt:
        if not disk_i_opt_dir or not os.path.isdir(disk_i_opt_dir):
            logger.error("use_disk_i_opt=true 但 disk_i_opt_dir 无效或目录不存在，请检查配置。")
            return
        if fusion_branch_input == 'rhat':
            logger.warning("⚠️  磁盘模式未运行 PRDNet，fusion_branch_input='rhat' 无效；仍使用磁盘加载的 A。")

    loss_fn = PRFNetLoss(
        w_recon=0.0,
        w_initialize=0.0,
        w_suppress=0.0,
        w_gradient=tr['w_gradient'],
        w_color=tr['w_color'],
        color_loss_ref=tr.get('color_loss_ref', 'auto'),
        w_consist=0.0,
        w_perceptual=tr.get('w_perceptual', 0.0),
        perceptual_ref=tr.get('perceptual_ref', 'i_opt'),
    ).to(device)

    logger.info(
        f"[OK] Loss (Stage 2 - DPFNet, fusion_only=True): w_gradient={tr['w_gradient']}, "
        f"w_color={tr['w_color']} (ref={tr.get('color_loss_ref', 'auto')}), "
        f"w_perceptual={tr.get('w_perceptual', 0.0)} (ref={tr.get('perceptual_ref', 'i_opt')}), "
        f"fusion_branch_input={fusion_branch_input}"
    )

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tr['G_optimizer_lr'],
        weight_decay=tr['G_optimizer_wd'],
        betas=tr['G_optimizer_betas']
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=tr['G_scheduler_milestones'],
        gamma=tr['G_scheduler_gamma']
    )

    train_set = DatasetPolar4Angles(opt['datasets']['train'])
    train_loader = DataLoader(
        train_set,
        batch_size=opt['datasets']['train']['dataloader_batch_size'],
        shuffle=opt['datasets']['train']['dataloader_shuffle'],
        num_workers=opt['datasets']['train']['dataloader_num_workers'],
        pin_memory=True
    )
    test_set = DatasetPolar4Angles(opt['datasets']['test']) if enable_test_phase else None
    logger.info(f"[OK] Dataset loaded: train={len(train_set)}, test={len(test_set) if test_set else 0}")

    logger.info("\n" + "=" * 80)
    logger.info("Starting Stage 2 Training (DPFNet in PRFNet with Frozen PRDNet)")
    logger.info("=" * 80)

    current_step = 0
    start_time = time.time()

    while current_step < tr['total_iters']:
        model.train()

        for batch_data in train_loader:
            current_step += 1
            if current_step > tr['total_iters']:
                break

            I_max = batch_data['I_max'].to(device)
            I_min = batch_data['I_min'].to(device)
            I_theta = batch_data['I_theta'].to(device)
            I_random = batch_data['I_random'].to(device)
            S0 = batch_data['S0'].to(device)
            DoLP = batch_data['DoLP'].to(device)

            if use_disk_i_opt:
                I_opt_in = load_i_opt_batch_aligned(batch_data, disk_i_opt_dir, I_max, device)
                I_final, Rhat, _ = model(
                    A=I_opt_in, B=DoLP, I_max=None, I_min=None, I_theta=None, I_random=None,
                    return_retinex_intermediate=True, fusion_branch_input=fusion_branch_input,
                )
                I_opt = I_opt_in
                L_max = L_min = L_random = R_refined = None
            else:
                I_final, Rhat, retinex_intermediate = model(
                    A=S0, B=DoLP, I_max=I_max, I_min=I_min, I_theta=I_theta, I_random=I_random,
                    return_retinex_intermediate=True, fusion_branch_input=fusion_branch_input,
                )
                L_max = retinex_intermediate['L_max']
                L_min = retinex_intermediate['L_min']
                L_random = retinex_intermediate['L_random']
                I_opt = retinex_intermediate.get('I_opt', None)
                R_refined = retinex_intermediate.get('R_refined', None)

            loss_total, loss_dict = loss_fn(
                I_max=I_max,
                I_min=I_min,
                L_max=L_max,
                L_min=L_min,
                I_random=I_random,
                L_random=L_random,
                I_final=I_final,
                Rhat=Rhat,
                DOLP=DoLP,
                I_opt=I_opt,
                R_refined=R_refined,
                fusion_only=True,
            )

            optimizer.zero_grad()
            loss_total.backward()
            if tr.get('G_regularizer_clipgrad', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    tr['G_regularizer_clipgrad']
                )
            optimizer.step()
            scheduler.step()

            if current_step % tr['checkpoint_print'] == 0:
                elapsed = time.time() - start_time
                eta = elapsed / current_step * (tr['total_iters'] - current_step)
                log_msg = (
                    f"[Step {current_step:06d}/{tr['total_iters']:06d}] "
                    f"[Time: {elapsed/60:.1f}m, ETA: {eta/60:.1f}m] "
                    f"[LR: {scheduler.get_last_lr()[0]:.6f}]\n"
                    f"  Loss: {loss_total.item():.4f} | Gradient: {loss_dict['gradient'].item():.4f} "
                    f"| Color: {loss_dict['color'].item():.4f}"
                )
                if tr.get('w_perceptual', 0.0) > 0:
                    log_msg += f" | Perceptual: {loss_dict['perceptual'].item():.4f}"
                logger.info(log_msg)

            if enable_test_phase and test_set is not None and current_step % tr['checkpoint_test'] == 0:
                logger.info(f"\n>>> Testing at step {current_step}...")
                model.eval()
                with torch.no_grad():
                    ref_img = tr.get('test_reference_image_path') or ''
                    test_data, img_stem = _resolve_test_batch(test_set, ref_img, logger)

                    test_I_max = test_data['I_max'].to(device)
                    test_I_min = test_data['I_min'].to(device)
                    test_I_theta = test_data['I_theta'].to(device)
                    test_I_random = test_data['I_random'].to(device)
                    test_S0 = test_data['S0'].to(device)
                    test_DoLP = test_data['DoLP'].to(device)

                    if use_disk_i_opt:
                        test_I_opt_in = load_i_opt_batch_aligned(test_data, disk_i_opt_dir, test_I_max, device)
                        test_I_final, _, _ = model(
                            A=test_I_opt_in, B=test_DoLP, I_max=None, I_min=None, I_theta=None, I_random=None,
                            return_retinex_intermediate=True, fusion_branch_input=fusion_branch_input,
                        )
                    else:
                        test_I_final, _, _ = model(
                            A=test_S0, B=test_DoLP, I_max=test_I_max, I_min=test_I_min,
                            I_theta=test_I_theta, I_random=test_I_random,
                            return_retinex_intermediate=True, fusion_branch_input=fusion_branch_input,
                        )

                    save_dir = os.path.join(opt['path']['images'], f'step_{current_step:06d}')
                    os.makedirs(save_dir, exist_ok=True)
                    out_final = os.path.join(save_dir, 'I_final.png')
                    util.imsave(util.tensor2uint(test_I_final[0]), out_final)
                    logger.info(f"   ✓ 已保存 I_final: {out_final}（样本 stem: {img_stem}）")
                model.train()

            if current_step % tr['checkpoint_save'] == 0:
                save_path = os.path.join(opt['path']['models'], f'prfnet_step_{current_step:06d}.pth')
                torch.save({
                    'step': current_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, save_path)
                logger.info(f"   ✓ Checkpoint saved: {save_path}")

    logger.info(f"Stage 2 Training Complete! Total time: {(time.time() - start_time)/3600:.2f} hours")


if __name__ == '__main__':
    main()
