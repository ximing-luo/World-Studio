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
  - **Sekiro (只狼) 画面重建**: 输入只狼游戏的原始画面 `(3, 128, 240)`。使用 **卷积 VAE (ConvVAE)** 模型，通过局部感受野和权重共享机制，显著降低参数量，从而能够轻松处理全分辨率彩色图像。
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
│   ├── sekiro/
│   │   ├── train_vae.py      # 只狼 VAE 训练脚本
│   │   ├── train_vqvae.py    # 只狼 VQ-VAE 训练脚本
│   │   ├── train_jepa.py     # 只狼 JEPA 训练脚本
│   │   └── train_rssm.py     # 只狼 RSSM 训练脚本
│   └── tools/
│       ├── prof_block.py     # 性能分析工具
│       ├── prof_fusion.py    # 融合操作分析
│       └── prof_model.py     # 模型分析
├── src/
│   ├── datasets/
│   │   ├── mnist.py          # MNIST 数据集
│   │   └── sekiro.py         # 只狼数据集
│   ├── model/
│   │   ├── backbone/         # 骨干网络组件
│   │   │   ├── attention.py  # 注意力机制 (MHA/GQA/MLA)
│   │   │   ├── moe.py        # 混合专家系统 (MoE)
│   │   │   ├── rms.py        # RMS 归一化
│   │   │   ├── rope.py       # 旋转位置编码
│   │   │   ├── transform.py  # Transformer 块 (Standard/Advanced/DeepSeekV2/V3)
│   │   │   └── vision.py     # 视觉投影层
│   │   ├── components/       # 模型通用组件
│   │   │   ├── attention.py  # 注意力机制
│   │   │   ├── discriminator.py # 判别器
│   │   │   ├── ecr.py        # 高效演化层
│   │   │   ├── loss.py       # 损失函数 (含感知损失)
│   │   │   ├── resnet.py     # ResNet 残差块 (Basic/BottleNeck/ResBlock/EResBlock)
│   │   │   └── rms.py        # RMS 归一化
│   │   └── world/            # 世界模型实现 (基础架构)
│   │       ├── jepa.py       # JEPA 模型基类
│   │       ├── rssm.py       # RSSM 模型基类
│   │       ├── vae.py        # VAE 模型基类
│   │       └── vq_vae.py     # VQ-VAE 模型基类
│   ├── world/                # 世界模型框架 (模块化通用架构)
│   │   ├── vision/           # 视觉编码器
│   │   │   ├── base.py       # 视觉基类
│   │   │   ├── mnist.py      # MNIST 视觉模块
│   │   │   └── sekiro.py     # 只狼视觉模块 (Conv/ResNet/Brain)
│   │   ├── projection/       # 潜空间投影层
│   │   │   └── projection.py # Linear/Attention/SpatialProjection
│   │   ├── latents/          # 隐空间约束层
│   │   │   ├── vae.py        # VAE 重参数化
│   │   │   ├── vq.py         # VQ-VAE 矢量量化
│   │   │   └── vicreg.py     # VICReg 正则化
│   │   ├── dream/            # 世界模型框架
│   │   │   ├── vae.py        # StaticReconstruction 框架
│   │   │   ├── jepa.py       # JEPA 预测框架
│   │   │   └── rssm.py       # RSSM 状态空间框架
│   │   └── predictor/        # 预测器模块
│   │       ├── predictor.py  # 时序/空间预测器
│   │       └── __init__.py
│   └── train/
│       └── train_utils.py    # 训练工具 (日志目录生成)
├── configs/
│   ├── __init__.py
│   ├── model.py              # 模型配置 (VVConfig/VisualVVConfig)
│   └── world.py              # 世界模型配置
├── outputs/
│   ├── results/              # 训练结果：重建图、损失曲线
│   └── models/               # 保存的模型权重 (.pth)
├── logs/                     # TensorBoard 日志
├── tests/                    # 测试文件
├── .gitignore
└── QWEN.md                   # 项目文档
```

## 环境要求

- **Python**: 3.8+
- **PyTorch**: 1.9+
- **Torchvision**: 0.10+
- **NumPy**
- **Matplotlib**
- **TensorBoard**

## 构建与运行

### 启动训练

```powershell
# MNIST 任务
python scripts/mnist/train_vae.py
python scripts/mnist/train_vqvae.py
python scripts/mnist/train_jepa.py
python scripts/mnist/train_rssm.py

# 只狼任务
python scripts/sekiro/train_vae.py
python scripts/sekiro/train_vqvae.py
python scripts/sekiro/train_jepa.py
python scripts/sekiro/train_rssm.py
```

训练过程将自动：
1. 根据脚本中的配置加载数据集。
2. 每个 epoch 保存一次重建对比图到 `outputs/results/`。
3. 训练结束后保存损失曲线和模型权重到 `outputs/models/`。
4. 使用 TensorBoard 记录训练指标 (Sekiro 任务)。

### 配置说明

在训练脚本中可调整以下参数：

```python
# 模型架构选择
model = FCVAE(latent_dim=20).to(device)        # 全连接 VAE
model = ConvVAE(latent_dim=20).to(device)      # 卷积 VAE
model = ResNetVAE(latent_dim=20).to(device)    # 残差 VAE

# VAE 超参数
beta = 1.0          # KL 散度权重 (β-VAE)
latent_dim = 256    # 隐空间维度

# RSSM 配置
seq_len = 8         # 序列长度
action_dim = 4      # 动作维度
```

## 开发规范

### 代码风格

- **类型注解**: 推荐使用 Python 类型注解
- **命名约定**:
  - 类名：大驼峰 (PascalCase)，如 `BaseVAE`, `AttentiveRSSM`
  - 函数/变量：小写 + 下划线 (snake_case)
  - 常量：全大写 + 下划线
- **文档字符串**: 类和公共方法应包含 docstring

### 模型设计规范

1. **模块化**: 所有世界模型继承自基类 (`BaseVAE`, `BaseJEPA`, `BaseRSSM`, `BaseVQVAE`)
2. **组件复用**: 使用 `backbone/` 和 `components/` 中的通用组件
3. **配置分离**: 超参数通过 `configs/` 中的 dataclass 管理

### 测试实践

- 新模型应在 `tests/` 目录下添加单元测试
- 测试前向传播、损失计算、梯度流动

## 扩展方向

1. **调整隐空间**: 在训练脚本中修改 `latent_dim` 以观察压缩效果。
2. **多帧预测**: 扩展 `SekiroDataset` 以支持输入 $t$ 预测 $t+1$。
3. **β-VAE**: 调整 `beta` 参数控制 KL 散度的权重。
4. **注意力机制**: 集成 `backbone/attention.py` 增强特征提取能力。
5. **残差结构**: 使用 `backbone/resnet.py` 构建更深的网络。
6. **新模型架构**: 参考 `scripts/mnist/` 下的脚本模板，实现新的世界模型变体。
7. **MoE 集成**: 使用 `backbone/moe.py` 中的混合专家系统扩展模型容量。
8. **DeepSeek 架构**: 集成 `DeepSeekV2Block`/`DeepSeekV3Block` 实现 MLA + MoE 架构。

## 常见问题

### Windows 多进程数据加载

在 Windows 系统上，`DataLoader` 的 `num_workers` 应设置为 `0` 以避免内存占用飙升：

```python
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
```

### 显存优化

- 使用 `PerceptualLoss` (基于 SqueezeNet) 替代 VGG 感知损失
- 调整 `batch_size` 和 `latent_dim` 平衡显存占用
- 使用梯度累积模拟大批次训练

## 输出说明

训练完成后，可在以下目录找到结果：

- `outputs/results/{task}/{model}/epoch_{n}.png` - 重建对比图
- `outputs/models/{task}_{model}.pth` - 模型权重
- `logs/{task}/{model}/{timestamp}/` - TensorBoard 日志

查看 TensorBoard:
```bash
tensorboard --logdir=logs/sekiro/vae
```
