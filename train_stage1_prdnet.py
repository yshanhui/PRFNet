"""
train_stage1_prdnet.py
Stage 1: Pre-train PRDNet (Polarimetric Retinex Decomposition Network)
"""
import os
import sys
import time
import random
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.prdnet import PRDNet, load_prdnet_state_dict
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='options/train_stage1_prdnet.json',
                        help='Path to option JSON file.')
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    setup_seed(opt['train']['manual_seed'])

    for key in ['models', 'images', 'log']:
        os.makedirs(opt['path'][key], exist_ok=True)

    logger_name = 'train'
    logger_info(logger_name, log_path=os.path.join(opt['path']['log'], f'{logger_name}.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    logger.info(">>> Creating PRDNet model...")
    model = PRDNet(channels=3, use_pde=True)

    if opt['path'].get('pretrained_netG', None):
        logger.info(f"Loading pretrained weights from: {opt['path']['pretrained_netG']}")
        checkpoint = torch.load(opt['path']['pretrained_netG'])
        load_prdnet_state_dict(model, checkpoint, logger)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[OK] Model created: total={num_params/1e6:.2f}M, trainable={num_trainable/1e6:.2f}M")

    tr = opt['train']
    loss_fn = PRFNetLoss(
        w_recon=tr['w_recon'],
        w_initialize=tr['w_initialize'],
        w_suppress=tr['w_suppress'],
        w_gradient=0.0,
        w_color=0.0,
        w_consist=tr['w_consist'],
        w_grad_polar=tr.get('w_grad_polar', 0.1),
        grad_polar_k=tr.get('grad_polar_k', 10.0),
    ).to(device)

    logger.info(
        f"[OK] Loss (Stage 1 - PRDNet): w_recon={tr['w_recon']}, w_initialize={tr['w_initialize']}, "
        f"w_suppress={tr['w_suppress']}, w_consist={tr['w_consist']}, "
        f"w_grad_polar={tr.get('w_grad_polar', 0.1)}, grad_polar_k={tr.get('grad_polar_k', 10.0)}"
    )

    optimizer = Adam(
        model.parameters(),
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
    test_set = DatasetPolar4Angles(opt['datasets']['test'])
    test_loader = DataLoader(
        test_set,
        batch_size=opt['datasets']['test']['dataloader_batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    logger.info(f"[OK] Dataset loaded: train={len(train_set)}, test={len(test_set)}")

    logger.info("\n" + "=" * 80)
    logger.info("Starting Stage 1 Training (PRDNet)")
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
            DoLP = batch_data['DoLP'].to(device)

            I_opt, Rhat, L_max, L_min, L_random, R_refined = model(
                I_max, I_min, I_theta=I_theta, I_random=I_random
            )

            loss, loss_dict = loss_fn(
                I_max=I_max,
                I_min=I_min,
                L_max=L_max,
                L_min=L_min,
                I_random=I_random,
                L_random=L_random,
                I_final=I_opt,
                Rhat=Rhat,
                DOLP=DoLP,
                I_opt=I_opt,
                R_refined=R_refined,
            )

            optimizer.zero_grad()
            loss.backward()
            if tr.get('G_regularizer_clipgrad', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr['G_regularizer_clipgrad'])
            optimizer.step()
            scheduler.step()

            if current_step % tr['checkpoint_print'] == 0:
                elapsed = time.time() - start_time
                eta = elapsed / current_step * (tr['total_iters'] - current_step)
                logger.info(
                    f"[Step {current_step:06d}/{tr['total_iters']:06d}] "
                    f"[Time: {elapsed/60:.1f}m, ETA: {eta/60:.1f}m] "
                    f"[LR: {scheduler.get_last_lr()[0]:.6f}]\n"
                    f"  Loss: {loss.item():.4f} | Recon: {loss_dict['recon'].item():.4f} "
                    f"| Init: {loss_dict['initialize'].item():.4f} | Suppress: {loss_dict['suppress'].item():.4f} "
                    f"| Consist: {loss_dict['consist'].item():.4f} | GradPolar: {loss_dict['grad_polar'].item():.4f}"
                )

            if current_step % tr['checkpoint_test'] == 0:
                logger.info(f"\n>>> Testing at step {current_step}...")
                model.eval()
                with torch.no_grad():
                    test_data = None
                    target_image_name = '3.png'
                    for batch in test_loader:
                        if target_image_name in batch['I_max_path'][0]:
                            test_data = batch
                            break
                    if test_data is None:
                        test_data = next(iter(test_loader))

                    test_I_max = test_data['I_max'].to(device)
                    test_I_min = test_data['I_min'].to(device)
                    test_I_theta = test_data['I_theta'].to(device)
                    test_I_random = test_data['I_random'].to(device)

                    test_I_opt, test_Rhat, test_L_max, test_L_min, _, test_R_refined = model(
                        test_I_max, test_I_min, I_theta=test_I_theta, I_random=test_I_random
                    )
                    test_L_opt = model.illum_fusion(test_L_max, test_L_min)

                    save_dir = os.path.join(opt['path']['images'], f'step_{current_step:06d}')
                    os.makedirs(save_dir, exist_ok=True)
                    util.imsave(util.tensor2uint(test_I_max[0]), os.path.join(save_dir, 'I_max.png'))
                    util.imsave(util.tensor2uint(test_I_min[0]), os.path.join(save_dir, 'I_min.png'))
                    util.imsave(util.tensor2uint(test_I_theta[0]), os.path.join(save_dir, 'I_theta.png'))
                    util.imsave(util.tensor2uint(test_I_random[0]), os.path.join(save_dir, 'I_random.png'))
                    util.imsave(util.tensor2uint(test_Rhat[0]), os.path.join(save_dir, 'Rhat.png'))
                    util.imsave(util.tensor2uint(test_R_refined[0]), os.path.join(save_dir, 'R_refined.png'))
                    util.imsave(util.tensor2uint(test_I_opt[0]), os.path.join(save_dir, 'I_opt.png'))
                    util.imsave(util.tensor2uint(test_L_opt[0]), os.path.join(save_dir, 'L_opt.png'))
                    logger.info(f"   ✓ Saved test images ({os.path.basename(test_data['I_max_path'][0])}) to: {save_dir}")
                model.train()

            if current_step % tr['checkpoint_save'] == 0:
                save_path = os.path.join(opt['path']['models'], f'prdnet_step_{current_step:06d}.pth')
                torch.save({
                    'step': current_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, save_path)
                logger.info(f"   ✓ Checkpoint saved: {save_path}")

    logger.info(f"Stage 1 Training Complete! Total time: {(time.time() - start_time)/3600:.2f} hours")


if __name__ == '__main__':
    main()
