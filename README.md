# PRFNet: Polarization-Aware Retinex Decomposition and Fusion for Object Detection under Adverse Illumination

Official PyTorch implementation of **PRFNet** (*Polarization-Retinex Fusion Network*), a two-stage polarization-intensity image fusion framework consisting of:
- **Stage 1 — PRDNet** (*Polarimetric Retinex Decomposition Network*): estimates shared reflectance and illumination from complementary polarization observations (`PRE`, `AIE`, `PDE`) to reconstruct a polarization-stabilized intermediate image $I_{\text{opt}}$.
- **Stage 2 — DPFNet** (*Decoupled Polarization Fusion Network*): fuses $I_{\text{opt}}$ and $\text{DoLP}$ via Shallow Spectral-Channel Fusion (`SSCF`) and a Decoupled Intensity-Chrominance Backbone (`DICB`) with `ICIF` and `CMF` modules.

## Repository Structure

```text
PRFNet/
├── models/
│   ├── prdnet.py                 # Stage 1: PRDNet (PRE, AIE, PDE, RTB)
│   ├── prfnet.py                 # Stage 2: PRFNet / DPFNet (SSCF, DICB, ICIF, CMF)
│   └── loss_prfnet.py            # Training loss functions
├── options/
│   ├── train_stage1_prdnet.json  # Stage 1 config
│   └── train_stage2_dpfnet.json  # Stage 2 config
├── data/
│   └── dataset_polar_4angles.py  # 4-angle polarimetric dataset loader
├── utils/                        # Image, config, and logging utilities
├── figures/                      # Sample inputs (S0, DoLP) and fused results
├── train_stage1_prdnet.py        # Stage 1 training
├── infer_stage1_prdnet.py        # Stage 1 inference
├── train_stage2_dpfnet.py        # Stage 2 training
├── test_prfnet.py                # Stage 2 testing / inference
└── requirements.txt
```

## Environment Setup

Recommended environment: Python 3.10 and PyTorch with CUDA support.

```bash
conda create -n prfnet python=3.10 -y
conda activate prfnet

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

## Datasets & Pretrained Weights

- **Datasets**: Experiments are conducted on [PoTATO](https://github.com/luisfelipewb/PoTATO) and [PODB](https://github.com/zhuz-bit/PODB). Organize the four polarization angles under a common root directory and update the paths in `options/*.json`:
  ```text
  <DATASET_ROOT>/
  ├── RGB00/
  ├── RGB45/
  ├── RGB90/
  └── RGB135/
  ```
- **Pretrained Weights**: *(Download link coming soon — place downloaded `.pth` checkpoints under `checkpoints/`.)*

## Training & Inference

### Stage 1: PRDNet

```bash
# Train
python train_stage1_prdnet.py --opt options/train_stage1_prdnet.json

# Inference (edit paths at the top of infer_stage1_prdnet.py)
python infer_stage1_prdnet.py
```

### Stage 2: DPFNet (PRFNet)

Set `path.pretrained_prdnet` in `options/train_stage2_dpfnet.json` to your Stage 1 checkpoint, then run:

```bash
# Train (with PRDNet frozen)
python train_stage2_dpfnet.py --opt options/train_stage2_dpfnet.json

# Test / Inference
python test_prfnet.py \
    --opt options/train_stage2_dpfnet.json \
    --checkpoint path/to/prfnet_checkpoint.pth \
    --output_dir results/test_all \
    --gpu_id 0
```
