"""
test_prfnet.py
Stage 2 (DPFNet / PRFNet) Inference: test all images and output fused image I_final.
"""
import os
import sys
import argparse
import logging
import traceback
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.prfnet import PRFNet, load_prfnet_state_dict
from data.dataset_polar_4angles import DatasetPolar4Angles
import utils.utils_option as option
import utils.utils_image as util
from utils.utils_logger import logger_info


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


def main():
    parser = argparse.ArgumentParser(description='Test PRFNet on all images and output I_final')
    parser.add_argument('--opt', type=str, default='options/train_stage2_dpfnet.json',
                        help='Path to option JSON file.')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/prfnet_podb.pth',
                        help='Path to trained PRFNet checkpoint (supports legacy checkpoints).')
    parser.add_argument('--output_dir', type=str, default='results/test_all',
                        help='Output directory for test results.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use.')
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=False)

    use_disk_i_opt = opt['train'].get('use_disk_i_opt', False)
    disk_i_opt_dir = opt['train'].get('disk_i_opt_dir') or opt['path'].get('disk_i_opt_dir', '')
    if use_disk_i_opt:
        disk_i_opt_dir = os.path.expanduser(disk_i_opt_dir)

    fusion_branch_input = (opt['train'].get('fusion_branch_input') or 'i_opt').strip().lower()
    if fusion_branch_input == 'iopt':
        fusion_branch_input = 'i_opt'
    if fusion_branch_input not in ('i_opt', 'rhat'):
        raise ValueError(f"train.fusion_branch_input 须为 'i_opt' 或 'rhat'，当前: {opt['train'].get('fusion_branch_input')!r}")

    if use_disk_i_opt:
        if fusion_branch_input == 'rhat':
            print("⚠️  警告: use_disk_i_opt=true 时 fusion_branch_input='rhat' 无效（磁盘模式不运行 PRDNet）")
        if not disk_i_opt_dir or not os.path.isdir(disk_i_opt_dir):
            raise FileNotFoundError(f"use_disk_i_opt=true 但 disk_i_opt_dir 无效或目录不存在: {disk_i_opt_dir}")

    device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger_name = 'test_prfnet'
    logger_info(logger_name, log_path=os.path.join(output_dir, 'test.log'))
    logger = logging.getLogger(logger_name)
    logger.info(f"opt={args.opt} | checkpoint={args.checkpoint} | output_dir={output_dir} | device={device}")
    logger.info(f"use_disk_i_opt={use_disk_i_opt} | fusion_branch_input={fusion_branch_input}")

    use_sscf = opt['netG'].get('use_sscf', opt['netG'].get('use_shallow_spectral_fusion', True))
    model = PRFNet(
        in_chans=3,
        embed_dim=opt['netG']['embed_dim'],
        Ex_depths=opt['netG']['Ex_depths'],
        Fusion_depths=opt['netG']['Fusion_depths'],
        Re_depths=opt['netG']['Re_depths'],
        use_prdnet=True,
        use_sscf=use_sscf,
    ).to(device)

    if not os.path.exists(args.checkpoint):
        logger.error(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    load_prfnet_state_dict(model, checkpoint, logger)
    step_str = checkpoint.get('step', 'unknown') if isinstance(checkpoint, dict) else 'raw'
    logger.info(f"   ✓ Loaded PRFNet weights (Step {step_str})")
    model.eval()

    if 'test' not in opt['datasets']:
        logger.error("❌ Test dataset not found in configuration!")
        return

    test_set = DatasetPolar4Angles(opt['datasets']['test'])
    test_loader = DataLoader(
        test_set,
        batch_size=3,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    logger.info(f"[OK] Test dataset loaded: {len(test_set)} images")

    I_final_dir = os.path.join(output_dir, 'I_final')
    os.makedirs(I_final_dir, exist_ok=True)

    total_images = len(test_set)
    success_count = 0
    failed_images = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Testing PRFNet")):
            try:
                I_max = batch_data['I_max'].to(device)
                I_min = batch_data['I_min'].to(device)
                I_theta = batch_data['I_theta'].to(device)
                I_random = batch_data['I_random'].to(device)
                S0 = batch_data['S0'].to(device)
                DoLP = batch_data['DoLP'].to(device)

                if use_disk_i_opt:
                    I_opt_in = load_i_opt_batch_aligned(batch_data, disk_i_opt_dir, I_max, device)
                    I_final, _, _ = model(
                        A=I_opt_in, B=DoLP, I_max=None, I_min=None, I_theta=None, I_random=None,
                        return_retinex_intermediate=True, fusion_branch_input='i_opt'
                    )
                else:
                    I_final, _, _ = model(
                        A=S0, B=DoLP, I_max=I_max, I_min=I_min, I_theta=I_theta, I_random=I_random,
                        return_retinex_intermediate=True, fusion_branch_input=fusion_branch_input
                    )

                for b in range(I_max.shape[0]):
                    base_name = _basename_from_batch_path(batch_data, b)
                    util.imsave(util.tensor2uint(I_final[b]), os.path.join(I_final_dir, f"{base_name}.png"))
                    success_count += 1

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"   Processed {success_count}/{total_images} images...")
            except Exception as e:
                logger.error(f"   ❌ Error processing batch {batch_idx}: {e}")
                logger.error(traceback.format_exc())
                failed_images.append(f"batch_{batch_idx}")

    logger.info(f"Testing Complete! Total={total_images}, Success={success_count}, Failed={len(failed_images)}")
    summary_path = os.path.join(output_dir, 'test_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(
            f"Checkpoint: {args.checkpoint}\n"
            f"Total images: {total_images}\n"
            f"Successfully processed: {success_count}\n"
            f"Failed: {len(failed_images)}\n"
            f"use_disk_i_opt: {use_disk_i_opt}\n"
            f"fusion_branch_input: {fusion_branch_input}\n"
        )
        if failed_images:
            f.write("Failed batches:\n" + "\n".join(failed_images) + "\n")


if __name__ == '__main__':
    main()
