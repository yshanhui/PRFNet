"""
infer_stage1_prdnet.py
Stage 1 (PRDNet) Inference: save I_opt, Rhat, R_refined, L_opt, and L_max / L_min / L_random.
"""
import os
import sys
import random
import logging
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.prdnet import PRDNet, load_prdnet_state_dict
from data.dataset_polar_4angles import DatasetPolar4Angles
import utils.utils_option as option
import utils.utils_image as util
from utils.utils_logger import logger_info

# ==================== 配置区（按需修改；支持直接加载历史 .pth 权重）====================
OPT_PATH = 'options/train_stage1_prdnet.json'
CHECKPOINT = 'checkpoints/prdnet_podb.pth'
OUTPUT_DIR = 'results/stage1'
GPU_ID = 0
SEED = 10
SAVE_INPUTS = False
# ======================================================================================


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    opt = option.parse(OPT_PATH, is_train=False)
    setup_seed(SEED)

    output_dir = os.path.expanduser(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    logger_name = 'infer_stage1'
    logger_info(logger_name, log_path=os.path.join(output_dir, 'infer_stage1.log'))
    logger = logging.getLogger(logger_name)
    logger.info(f'opt: {OPT_PATH}\ncheckpoint: {CHECKPOINT}\noutput_dir: {output_dir}')

    if not os.path.isfile(CHECKPOINT):
        logger.error(f'未找到权重文件: {CHECKPOINT}')
        return

    device = torch.device(f'cuda:{GPU_ID}') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f'使用设备: {device}')

    model = PRDNet(channels=3, use_pde=True).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    load_prdnet_state_dict(model, ckpt, logger)
    logger.info(f"   ✓ 已加载 PRDNet 权重（step={ckpt.get('step', 'unknown') if isinstance(ckpt, dict) else 'raw'}）")
    model.eval()

    if 'test' not in opt['datasets']:
        logger.error('配置中缺少 datasets.test')
        return

    test_set = DatasetPolar4Angles(opt['datasets']['test'])
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=opt['datasets']['test'].get('dataloader_num_workers', 0),
        pin_memory=torch.cuda.is_available(),
    )
    logger.info(f'测试集样本数: {len(test_set)}')

    subdirs = {
        name: os.path.join(output_dir, name)
        for name in ('I_opt', 'Rhat', 'R_refined', 'L_opt', 'L_max', 'L_min', 'L_random')
    }
    if SAVE_INPUTS:
        for name in ('I_max', 'I_min', 'I_theta', 'I_random'):
            subdirs[name] = os.path.join(output_dir, name)
    for p in subdirs.values():
        os.makedirs(p, exist_ok=True)

    success, failed = 0, []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc='Stage1 infer')):
            img_filename = f'image_{batch_idx:04d}.png'
            try:
                I_max = batch_data['I_max'].to(device)
                I_min = batch_data['I_min'].to(device)
                I_theta = batch_data['I_theta'].to(device)
                I_random = batch_data['I_random'].to(device)

                if 'I_max_path' in batch_data and batch_data['I_max_path']:
                    ip = batch_data['I_max_path']
                    img_path = ip[0] if isinstance(ip, (list, tuple)) else ip
                    img_filename = os.path.basename(img_path)
                out_name = os.path.splitext(img_filename)[0] + '.png'

                I_opt, Rhat, L_max, L_min, L_random, R_refined = model(
                    I_max, I_min, I_theta=I_theta, I_random=I_random
                )
                L_opt = model.illum_fusion(L_max, L_min)

                outputs = {
                    'I_opt': I_opt, 'Rhat': Rhat, 'R_refined': R_refined,
                    'L_opt': L_opt, 'L_max': L_max, 'L_min': L_min, 'L_random': L_random,
                }
                if SAVE_INPUTS:
                    outputs.update({'I_max': I_max, 'I_min': I_min, 'I_theta': I_theta, 'I_random': I_random})

                for key, tensor in outputs.items():
                    util.imsave(util.tensor2uint(tensor[0]), os.path.join(subdirs[key], out_name))

                success += 1
            except Exception as e:
                name = os.path.splitext(img_filename)[0]
                logger.error(f'处理失败 {name}: {e}')
                failed.append(name)
                logger.error(traceback.format_exc())

    logger.info(f'完成。成功: {success}，失败: {len(failed)}，结果目录: {output_dir}')
    summary = os.path.join(output_dir, 'infer_summary.txt')
    with open(summary, 'w', encoding='utf-8') as f:
        f.write(f'checkpoint: {CHECKPOINT}\nopt: {OPT_PATH}\nsuccess: {success}\nfailed: {len(failed)}\n')
        if failed:
            f.write('failed_list:\n' + '\n'.join(failed) + '\n')


if __name__ == '__main__':
    main()
