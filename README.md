# PRFNet: Polarization-Aware Retinex Decomposition and Fusion for Object Detection under Adverse Illumination

本项目为论文 **PRFNet** (*Polarization-Retinex Fusion Network*) 的官方实现，包含两阶段极化–Retinex 图像融合框架 **Stage 1（PRDNet: Polarimetric Retinex Decomposition Network）** 与 **Stage 2（DPFNet: Decoupled Polarization Fusion Network）** 的完整训练、推理代码、配置文件与示例图像。

---

## 1. 两阶段网络架构简介

1. **Stage 1 — `PRDNet`（偏振感知 Retinex 分解网络）**：
   - 输入四角度偏振图像（`0° / 45° / 90° / 135°`）经 Stokes 计算得到的偏振极值图 `I_max`、`I_min`、Malus 随机角重建图 `I_theta` 及随机采样观测 `I_random`（即 $I_s$）。
   - **PRE（Polarization Reflectance Estimator）**：基于 `I_max` 与 `I_min` 及级联的 **RTB（Restormer Transformer Block）** 估计跨偏振态共享的场景反射率 $\hat{R}$（`Rhat`）。
   - **AIE（Adaptive Illumination Estimator）**：采用随机偏振采样策略估计统一光照图 $L_s$（`L_max / L_min / L_random`），并与 `AdaptiveIlluminationFusion` 联合生成融合光照 `L_opt`。
   - **PDE（Polarimetric Degradation Estimator）**：在 $\hat{R}$、$\text{DoLP}$ 与光照图 $L_s$ 的引导下预测表观反射率 $R_s$（`R_refined`），以观测重构约束辅助共享反射率学习。
   - 最终由共享反射率与估计光照组合得到抑制偏振退化的中间强度图像 $I_{\text{opt}} = L_s \odot \hat{R}$（`I_opt`）。

2. **Stage 2 — `DPFNet`（解耦偏振融合网络，集成于 `PRFNet`）**：
   - 冻结预训练好的 Stage 1 `PRDNet` 参数（`freeze_prdnet: true`），仅反向传播优化 `DPFNet` 融合网络。
   - **SSCF（Shallow Spectral-Channel Fusion）**：在 RGB 空间中对 `I_opt` 与 `DoLP` 按 `R / G / B` 三通道分别执行跨通道交叉注意力交互，保留光谱通道对应关系。
   - **DICB（Decoupled Intensity-Chrominance Backbone）**：将 `I_opt` 与 `DoLP` 转换至 HVI 色彩空间，分别进入对称的 U 型 Transformer 亮度分支（`I-Branch`）与色度分支（`HV-Branch`）；编码器/瓶颈/解码器各阶段通过 **ICIF（Intensity-Chrominance Interaction Fusion）** 实现跨分支双向特征交互，在瓶颈处利用 **CMF（Cross-Modal Fusion）** 模块生成自适应权重完成跨模态融合。
   - **联合重建输出**：将 HVI 重建结果转换回 RGB 后与 `SSCF` 浅层特征拼接融合，输出最终融合图像 $I_f$（`I_final`）。

---

## 2. 目录结构

```text
PRFNet/
├── train_stage1_prdnet.py         # Stage 1 训练入口（PRDNet 预训练）
├── infer_stage1_prdnet.py         # Stage 1 推理入口（导出 I_opt / Rhat / R_refined / L_*）
├── train_stage2_dpfnet.py         # Stage 2 训练入口（冻结 PRDNet，训练 DPFNet）
├── test_prfnet.py                 # Stage 2 测试推理入口（批量输出融合图像 I_final）
├── requirements.txt               # Python 依赖列表
├── README.md                      # 项目说明与运行文档
├── options/
│   ├── train_stage1_prdnet.json   # Stage 1 训练/测试配置文件
│   └── train_stage2_dpfnet.json   # Stage 2 训练/测试配置文件
├── models/
│   ├── prdnet.py                  # PRDNet（PRE, AIE, PDE, RTB）及权重加载函数
│   ├── prfnet.py                  # PRFNet / DPFNet（SSCF, DICB, ICIF, CMF）及权重加载函数
│   └── loss_prfnet.py             # PRFNetLoss（Stage 1 PRDNet 损失与 Stage 2 DPFNet 损失）
├── data/
│   └── dataset_polar_4angles.py   # 四角度偏振数据集与 Stokes / Malus 参数计算
├── utils/
│   ├── utils_image.py             # 图像读写、Tensor 转换与数据增强
│   ├── utils_option.py            # JSON 配置解析与路径/GPU 初始化
│   └── utils_logger.py            # 日志初始化
└── figures/                       # 示例偏振图像（S0 与 DoLP，1224×1024）及融合输出
    ├── podb/
    │   ├── DOLP/   (376.png, 1005.png, 1397.png)
    │   ├── S0/     (376.png, 1005.png, 1397.png)
    │   └── fusion/ (376.png, 1005.png, 1397.png)  # PODB 预训练权重推理结果
    └── potato/
        ├── DOLP/   (00002.png, 00430.png)
        ├── S0/     (00002.png, 00430.png)
        └── fusion/ (00002.png, 00430.png)         # PoTATO 预训练权重推理结果
```

> 注：预训练权重文件（`.pth`）体积较大，将通过网盘链接单独提供（下载后请放置于 `checkpoints/` 或对应配置指定路径）。

---

## 3. 环境配置

### 3.1 创建 Conda 环境与安装依赖

推荐使用 Python 3.10 与带 CUDA 支持的 PyTorch：

```bash
# 1. 创建并激活环境
conda create -n prfnet python=3.10 -y
conda activate prfnet

# 2. 安装 CUDA 版 PyTorch（以 CUDA 12.4 为例；如对应 CUDA 12.1 可改用 cu121）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. 安装其余依赖
pip install -r requirements.txt
```

> 若本地网络访问 Conda 镜像异常，且本地 pkgs 缓存已有 Python 3.10，可先执行 `conda create -n prfnet python=3.10 --offline -y` 创建基础环境，再用 `pip` 安装依赖。

### 3.2 验证环境与实测版本

在 `NVIDIA GeForce RTX 3090 Ti`（24 GB 显存）上验证通过的软件版本：

| 组件 | 实测版本 |
| :--- | :--- |
| Python | `3.10.22` |
| PyTorch / CUDA | `2.6.0+cu124` (CUDA `12.4`) |
| torchvision | `0.21.0+cu124` |
| numpy | `2.2.6` |
| opencv-python (`cv2`) | `5.1.0` |
| einops | `0.8.2` |
| timm | `1.0.31` |
| tqdm | `4.70.1` |

> 注：Stage 2 启用感知损失（`w_perceptual > 0`）时，首次运行会自动下载 PyTorch 官方 `vgg16-397923af.pth`（约 528 MB）并缓存至用户目录 `~/.cache/torch/hub/checkpoints/`。

---

## 4. 数据集与示例图像说明

### 4.1 数据集获取与目录格式

本论文实验基于两个公开偏振目标检测数据集：
- **PoTATO**：https://github.com/luisfelipewb/PoTATO
- **PODB**：https://github.com/zhuz-bit/PODB

`DatasetPolar4Angles` 读取同一场景下 4 个偏振角度（`0° / 45° / 90° / 135°`）的同名 RGB 图像目录，并在内存中实时计算 `S0`、`DoLP`、`AoP`、`I_max`、`I_min` 与随机角度 `I_theta`：

```text
<DATASET_ROOT>/
├── RGB00/     # 0° 偏振图像
├── RGB45/     # 45° 偏振图像
├── RGB90/     # 90° 偏振图像
└── RGB135/    # 135° 偏振图像
```

运行前请在 `options/train_stage1_prdnet.json` 与 `options/train_stage2_dpfnet.json` 中修改 `datasets.train` 和 `datasets.test` 的 `dataroot_0deg / 45deg / 90deg / 135deg`（以及 Stage 2 的 `pretrained_prdnet`、`test_reference_image_path`）为实际本地路径。

### 4.2 内置 `figures/` 示例图像与显存实测

项目已内置 PoTATO 与 PODB 两套数据的 `1224×1024` 示例图及对应融合输出（`figures/podb/fusion/` 与 `figures/potato/fusion/`）：

- **推理输入推导说明**：
  - `PRFNet.forward(A=S0, B=DoLP, I_max=None, ...)` 可在直通模式下直接接收 `S0` 与 `DoLP`。
  - 在线 `PRDNet` 模式下，若仅有 `S0` 与 `DoLP`，可由 Stokes 强度关系近似构造 `I_max = clip(S0*(1+DoLP), 0, 1)`、`I_min = clip(S0*(1-DoLP), 0, 1)`、`I_theta = I_random = S0` 送入模型。
- **RTX 3090 Ti 实测显存与裁剪建议**：
  - 整图 `1224×1024` 推理：Stage 1 峰值显存约 **2.73 GB**；Stage 2 峰值显存约 **11.4–11.6 GB**（24 GB 显存可直接整图运行）。
  - 训练单步（`128×128` 裁剪 patch）：Stage 1 峰值显存约 **0.60 GB**；Stage 2（含 VGG16 感知损失）峰值显存约 **2.99 GB**。
  - 若在显存较小（如 8 GB / 12 GB）的 GPU 上对 Stage 2 整图推理发生 OOM，建议先将输入图像中心裁剪或分块为 `512×512` 或 `256×256` 后再推理。

---

## 5. 训练与推理运行命令

请先激活环境并进入本项目根目录 `PRFNet`：

```bash
conda activate prfnet
cd PRFNet
```

### 5.1 Stage 1 训练（PRDNet 偏振 Retinex 分解）

```bash
python train_stage1_prdnet.py --opt options/train_stage1_prdnet.json
```

- 默认每 `100` 步打印日志，每 `1000` 步保存测试可视化图（`I_max, I_min, I_theta, I_random, Rhat, R_refined, I_opt, L_opt`），每 `5000` 步保存权重至 `Model/stage1_prdnet/models/prdnet_step_XXXXXX.pth`。

### 5.2 Stage 1 推理

编辑 `infer_stage1_prdnet.py` 顶部的配置区（`OPT_PATH`、`CHECKPOINT`、`OUTPUT_DIR`），然后运行：

```bash
python infer_stage1_prdnet.py
```

- 输出目录中会生成 `I_opt/`、`Rhat/`、`R_refined/`、`L_opt/`、`L_max/`、`L_min/`、`L_random/` 及 `infer_summary.txt`。

### 5.3 Stage 2 训练（DPFNet 融合网络）

在 `options/train_stage2_dpfnet.json` 中设置好 `path.pretrained_prdnet`（指向 Stage 1 训练好的 `.pth`），然后运行：

```bash
python train_stage2_dpfnet.py --opt options/train_stage2_dpfnet.json
```

- 默认 `freeze_prdnet: true` 冻结 `PRDNet`，仅优化 `DPFNet` 融合模块参数，权重保存至 `Model/stage2_dpfnet/models/prfnet_step_XXXXXX.pth`。
- 若希望跳过在线 `PRDNet` 前向、直接读取阶段一导出到磁盘的 `I_opt` 图像，可在 JSON 中设置 `"use_disk_i_opt": true` 并指定 `disk_i_opt_dir`。

### 5.4 Stage 2 测试推理（批量输出 `I_final`）

```bash
python test_prfnet.py \
    --opt options/train_stage2_dpfnet.json \
    --checkpoint path/to/prfnet_step_XXXXXX.pth \
    --output_dir results/test_all \
    --gpu_id 0
```

- **权重兼容说明**：`models/prdnet.py` 与 `models/prfnet.py` 分别内置了 `load_prdnet_state_dict()` 和 `load_prfnet_state_dict()`，可自动完成历史权重键名映射与冗余单分支键过滤，同时对所有有效层做严格校验，支持直接加载已发布的历史预训练权重。
