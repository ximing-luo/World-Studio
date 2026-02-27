# World Model - VAE Project

## 项目概述

这是一个基于 **PyTorch** 的深度学习项目，实现了 **变分自编码器 (Variational Autoencoder, VAE)** 模型。该项目模拟了"世界模型"(World Model)的核心思想：学习理解和重建环境状态。

### 核心功能

- **VAE 模型**: 通用编码器 - 解码器架构，支持动态输入维度。
- **任务模式**:
  - **MNIST 旋转预测**: 输入当前帧 (t)，预测旋转后的下一帧 (t+1)。支持固定角度或根据数字类别动态旋转。
  - **Sekiro (只狼) 画面重建**: 输入只狼游戏的原始画面 `(3, 136, 240)`。使用 **卷积 VAE (ConvVAE)** 模型，通过局部感受野和权重共享机制，显著降低参数量（约 200MB 显存占用），从而能够轻松处理全分辨率彩色图像。
- **训练可视化**: 每个 epoch 自动保存重建对比图（Input / Target / Predict）和损失曲线。
- **ELBO 损失**: 结合重建损失 (BCE) 和 KL 散度正则化。

## 项目结构

```
world/
├── data/
│   └── demos/                # 只狼轨迹数据 (.pt 文件)
├── scripts/
│   ├── train_mnist.py       # MNIST 旋转预测任务启动脚本
│   └── train_sekiro.py      # 只狼 画面重建任务启动脚本 (卷积 VAE 版)
├── src/
│   ├── datasets/
│   │   └── dataset.py        # RotatedMNIST & SekiroDataset
│   ├── model/
│   │   ├── vae.py            # 全连接 VAE 模型 (MNIST)
│   │   ├── conv_vae.py       # 卷积 VAE 模型 (只狼)
│   │   ├── spatial.py        # 空间变换模块
│   │   ├── loss.py           # ELBO 损失函数
│   │   └── components/       # 模型组件
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
└── .gitignore
```

## 运行与训练

### 前置条件

需要安装 `torch`, `torchvision`, `matplotlib`, `numpy` 等库。

### 启动训练

请确保在 **conda 虚拟环境** 中运行以下命令：

```powershell
# 运行 MNIST 旋转预测任务
& D:/Software/miniconda3/python.exe scripts/train_mnist.py

# 运行只狼画面重建任务 (卷积版)
& D:/Software/miniconda3/python.exe scripts/train_sekiro.py
```

训练过程将自动：
1. 根据 `scripts/train.py` 中的 `task` 配置加载数据集。
2. 清空旧的 `outputs/results/` 目录。
3. 每个 epoch 保存一次重建对比图到 `outputs/results/`。
4. 训练结束后保存损失曲线和模型权重到 `outputs/models/`。

## 扩展方向

1. **调整隐空间**: 在 `train.py` 中修改 `latent_dim` 以观察压缩效果。
2. **多帧预测**: 扩展 `SekiroDataset` 以支持输入 $t$ 预测 $t+1$。
3. **β-VAE**: 调整 `beta` 参数控制 KL 散度的权重。
4. **注意力机制**: 集成 `components/attention.py` 增强特征提取能力。
5. **残差结构**: 使用 `components/resnet.py` 构建更深的网络。
