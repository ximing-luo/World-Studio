# World Model Studio

## 项目概述

这是一个基于 **PyTorch** 的深度学习项目，实现了多种 **世界模型 (World Model)** 架构。该项目旨在学习理解和预测环境状态的动态变化。

### 核心功能

- **多种模型架构**:
  - **VAE (变分自编码器)**: 通用编码器 - 解码器架构。支持 **FC** (全连接)、**Conv** (卷积)、**ResNet** (残差) 三种骨干网络。
  - **VQ-VAE (矢量量化 VAE)**: 离散隐空间表示。支持 **FC**、**Conv**、**ResNet** 三种骨干网络。
  - **JEPA (联合嵌入预测架构)**: 基于表征的预测模型。支持 **FC**、**Conv**、**ResNet** 三种骨干网络。
  - **RSSM (循环状态空间模型)**: 结合 RNN 与状态空间建模。支持 **FC**、**Conv**、**ResNet** 三种骨干网络。
- **任务模式**:
  - **MNIST 旋转预测**: 输入当前帧 (t)，预测旋转后的下一帧 (t+1)。支持固定角度或随机角度旋转。
  - **Sekiro (只狼) 画面重建**: 输入只狼游戏的原始画面 `(3, 136, 240)`。使用 **卷积 VAE (ConvVAE)** 模型，通过局部感受野和权重共享机制，显著降低参数量，从而能够轻松处理全分辨率彩色图像。
- **训练可视化**: 每个 epoch 自动保存重建对比图（Input / Target / Predict）和损失曲线。
- **ELBO 损失**: 结合重建损失 (BCE/MSE) 和 KL 散度正则化。

## 项目结构

```
World-Studio/
├── data/
│   └── demos/                # 只狼轨迹数据 (.pt 文件)
├── scripts/
│   ├── mnist/
│   │   ├── train_vae.py      # MNIST VAE 训练脚本
│   │   ├── train_vqvae.py    # MNIST VQ-VAE 训练脚本
│   │   ├── train_jepa.py     # MNIST JEPA 训练脚本
│   │   └── train_rssm.py     # MNIST RSSM 训练脚本
│   └── sekiro/
│       ├── train_vae.py      # 只狼 VAE 训练脚本
│       ├── train_vqvae.py    # 只狼 VQ-VAE 训练脚本
│       ├── train_jepa.py     # 只狼 JEPA 训练脚本
│       └── train_rssm.py     # 只狼 RSSM 训练脚本
├── src/
│   ├── datasets/
│   │   ├── mnist.py          # MNIST 数据集
│   │   └── sekiro.py         # 只狼数据集
│   ├── model/
│   │   ├── mnist/            # MNIST 任务模型 (支持 FC/Conv/ResNet)
│   │   │   ├── vae.py
│   │   │   ├── vq_vae.py
│   │   │   ├── jepa.py
│   │   │   └── rssm.py
│   │   ├── sekiro/           # 只狼任务模型 (支持 Conv/ResNet)
│   │   │   ├── vae.py
│   │   │   ├── vq_vae.py
│   │   │   ├── jepa.py
│   │   │   └── rssm.py
│   │   └── components/       # 模型通用组件
│   │       ├── attention.py  # 注意力机制
│   │       ├── resnet.py     # ResNet 残差块
│   │       └── rms.py        # RMS 归一化
│   └── train/
│       ├── trainer.py        # 训练核心逻辑
│       └── untils.py         # 可视化工具
├── outputs/
│   ├── results/              # 训练结果：重建图、损失曲线
│   └── models/               # 保存的模型权重 (.pth)
├── tests/                    # 测试文件
├── .gitignore
└── QWEN.md                   # 项目文档
```

## 运行与训练

### 前置条件

需要安装 `torch`, `torchvision`, `matplotlib`, `numpy` 等库。

### 启动训练

请确保在 **conda 虚拟环境** 中运行以下命令：

```powershell
# MNIST 任务
& D:/Software/miniconda3/python.exe scripts/mnist/train_vae.py
& D:/Software/miniconda3/python.exe scripts/mnist/train_vqvae.py
& D:/Software/miniconda3/python.exe scripts/mnist/train_jepa.py
& D:/Software/miniconda3/python.exe scripts/mnist/train_rssm.py

# 只狼任务
& D:/Software/miniconda3/python.exe scripts/sekiro/train_vae.py
& D:/Software/miniconda3/python.exe scripts/sekiro/train_vqvae.py
& D:/Software/miniconda3/python.exe scripts/sekiro/train_jepa.py
& D:/Software/miniconda3/python.exe scripts/sekiro/train_rssm.py
```

训练过程将自动：
1. 根据脚本中的配置加载数据集。
2. 清空旧的 `outputs/results/` 目录。
3. 每个 epoch 保存一次重建对比图到 `outputs/results/`。
4. 训练结束后保存损失曲线和模型权重到 `outputs/models/`。

## 扩展方向

1. **调整隐空间**: 在训练脚本中修改 `latent_dim` 以观察压缩效果。
2. **多帧预测**: 扩展 `SekiroDataset` 以支持输入 $t$ 预测 $t+1$。
3. **β-VAE**: 调整 `beta` 参数控制 KL 散度的权重。
4. **注意力机制**: 集成 `components/attention.py` 增强特征提取能力。
5. **残差结构**: 使用 `components/resnet.py` 构建更深的网络。
6. **新模型架构**: 参考 `scripts/mnist/` 下的脚本模板，实现新的世界模型变体。
