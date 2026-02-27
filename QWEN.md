# World Model - VAE Project

## 项目概述

这是一个基于 **PyTorch** 的深度学习项目，实现了 **变分自编码器 (Variational Autoencoder, VAE)** 模型，用于 MNIST 手写数字的旋转预测任务。该项目模拟了"世界模型"(World Model)的核心思想：学习预测下一帧/下一状态。

### 核心功能

- **VAE 模型**: 编码器 - 解码器架构，隐空间维度为 15
- **旋转预测任务**: 输入当前帧 (t)，预测旋转 45° 后的下一帧 (t+1)
- **训练可视化**: 每个 epoch 自动保存重建对比图和损失曲线
- **ELBO 损失**: 结合重建损失 (BCE) 和 KL 散度正则化

## 项目结构

```
world/
├── configs/
│   └── vae_mnist.py          # 超参数配置文件
├── scripts/
│   └── train.py              # 训练启动脚本（精简版）
├── src/
│   ├── datasets/
│   │   └── dataset.py        # RotatedMNIST 数据集包装器
│   ├── model/
│   │   ├── vae.py            # VAE 模型定义
│   │   └── loss.py           # ELBO 损失函数
│   └── train/
│       ├── trainer.py        # Trainer 训练类（核心逻辑封装）
│       └── untils.py         # 可视化工具（重建图、损失曲线）
├── data/                     # MNIST 数据下载目录（git 忽略）
├── outputs/
│   ├── results/              # 训练结果：重建图、损失曲线
│   └── models/               # 保存的模型权重
└── .gitignore
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 框架 | PyTorch |
| 数据集 | torchvision.datasets.MNIST |
| 优化器 | Adam (lr=1e-3) |
| 设备 | CUDA / CPU 自动检测 |

## 模型架构

```
输入 (784) → Encoder → [μ(15), logσ²(15)] → 重参数化 → z(15) → Decoder → 输出 (784)
                    ↓
              KL 散度正则化
```

**超参数** (在 `scripts/train.py` 中配置):
- `batch_size`: 128
- `epochs`: 6
- `learning_rate`: 1e-3
- `beta`: 1.0 (KL 损失权重)
- `latent_dim`: 15
- `hidden_dim`: 300

## 运行与训练

### 前置条件

```bash
pip install torch torchvision matplotlib numpy
```

### 启动训练

```bash
& D:/Software/miniconda3/python.exe d:scripts/train.py
```

训练过程将自动：
1. 下载 MNIST 数据集到 `data/` 目录
2. 清空旧的 `outputs/results/` 目录
3. 每 2 个 epoch 保存一次重建对比图
4. 训练结束后保存损失曲线和模型权重

### 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/results/reconstruction_epoch_*.png` | 输入/目标/预测对比图 |
| `outputs/results/loss_curve.png` | 训练损失曲线 |
| `outputs/models/vae_mnist.pth` | 模型权重 |

## 开发约定

- **代码风格**: 函数式模块化设计，每个模块职责单一
- **文件命名**: 小写 + 下划线 (snake_case)
- **目录组织**: 按功能分层 (`datasets/`, `model/`, `trainer/`)
- **可视化**: 训练过程自动记录，便于调试和复现

## 扩展方向

1. **调整旋转角度**: 修改 `RotatedMNIST` 的 `angle` 参数
2. **改变隐空间维度**: 调整 `VAE` 的 `latent_dim`
3. **β-VAE 实验**: 调整 `beta` 参数控制正则化强度
4. **其他预测任务**: 扩展 `dataset.py` 实现不同的状态转移规则
