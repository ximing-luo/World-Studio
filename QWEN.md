# World Model Studio - QWEN.md

## 项目概述

**World-Studio** 是一个基于 **PyTorch** 的深度学习研究项目，专注于实现多种**世界模型 (World Model)** 架构。项目核心目标是学习理解和预测环境状态的时空动态变化。

### 核心技术栈

- **深度学习框架**: PyTorch 1.9+
- **编程语言**: Python 3.8+
- **辅助库**: NumPy, Matplotlib, TensorBoard, Torchvision

### 支持的模型架构

| 模型 | 描述 | 骨干网络支持 |
|------|------|-------------|
| **VAE** | 变分自编码器，连续隐空间 | FC / Conv / ResNet |
| **VQ-VAE** | 矢量量化 VAE，离散隐空间 | FC / Conv / ResNet |
| **JEPA** | 联合嵌入预测架构 | FC / Conv / ResNet |
| **RSSM** | 循环状态空间模型 | FC / Conv / ResNet |

### 任务模式

1. **MNIST 旋转预测**: 输入当前帧 (t)，预测旋转后的下一帧 (t+1)
2. **Sekiro (只狼) 画面重建**: 处理游戏画面 `(3, 128, 240)`，使用卷积 VAE 进行全分辨率彩色图像处理

---

## 项目结构

```
World-Studio/
├── data/
│   └── demos/                    # 只狼轨迹数据 (.pt 文件)
├── scripts/
│   ├── mnist/                    # MNIST 任务训练脚本
│   │   ├── train_vae.py
│   │   ├── train_vqvae.py
│   │   ├── train_jepa.py
│   │   └── train_rssm.py
│   ├── sekiro/                   # 只狼任务训练脚本
│   │   ├── train_vae.py
│   │   ├── train_vqvae.py
│   │   ├── train_jepa.py
│   │   └── train_rssm.py
│   └── tools/
│       └── prof_model.py         # 模型分析工具
├── src/
│   ├── datasets/
│   │   ├── mnist.py              # MNIST 数据集封装
│   │   └── sekiro.py             # 只狼数据集 (NPY 格式)
│   ├── model/
│   │   ├── backbone/             # 骨干网络组件
│   │   │   ├── attention.py      # MHA/GQA/MLA 注意力
│   │   │   ├── moe.py            # 混合专家系统
│   │   │   ├── rms.py            # RMS 归一化
│   │   │   ├── rope.py           # 旋转位置编码
│   │   │   ├── transform.py      # Transformer 块
│   │   │   └── vision.py         # 视觉投影层
│   │   ├── components/           # 通用模型组件
│   │   │   ├── attention.py
│   │   │   ├── focus.py          # 空间专注层
│   │   │   ├── loss.py           # 损失函数 (含感知损失)
│   │   │   ├── norm.py           # 归一化层
│   │   │   ├── resnet.py         # ResNet 残差块
│   │   │   └── cuda_norm/        # CUDA 归一化算子
│   │   ├── ecr/                  # 高效演化层 (ECR)
│   │   │   ├── ecr.py
│   │   │   └── cuda_evolution/   # CUDA 演化算子
│   │   ├── gan/                  # 判别器组件
│   │   │   └── discriminator.py
│   │   └── world/                # 世界模型基类
│   │       ├── vae.py
│   │       ├── vq_vae.py
│   │       ├── jepa.py
│   │       └── rssm.py
│   ├── world/                    # 模块化世界模型框架
│   │   ├── vision/               # 视觉编码器
│   │   │   ├── base.py
│   │   │   ├── mnist.py
│   │   │   └── sekiro.py
│   │   ├── projection/           # 潜空间投影层
│   │   │   └── projection.py
│   │   ├── latents/              # 隐空间约束层
│   │   │   ├── vae.py
│   │   │   ├── vq.py
│   │   │   └── vicreg.py
│   │   ├── dream/                # 世界模型框架
│   │   │   ├── vae.py            # StaticReconstruction
│   │   │   ├── jepa.py
│   │   │   └── rssm.py
│   │   └── predictor/            # 预测器模块
│   ├── utils/
│   │   ├── loss.py               # 损失函数工具
│   │   └── train_utils.py        # 训练工具
│   └── train/
├── configs/
│   ├── world.py                  # 世界模型配置 (MLA+MoE)
│   └── model.py                  # 模型配置
├── outputs/
│   ├── results/                  # 重建图、损失曲线
│   └── models/                   # 模型权重 (.pth)
├── logs/                         # TensorBoard 日志
└── tests/                        # 单元测试
```

---

## 构建与运行

### 环境要求

```
Python >= 3.8
PyTorch >= 1.9
Torchvision >= 0.10
NumPy
Matplotlib
TensorBoard
```

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

### 查看 TensorBoard

```bash
tensorboard --logdir=logs/sekiro/vqvae
```

---

## 核心架构说明

### 1. 静态重建框架 (StaticReconstruction)

```
Image -> [Vision Encode] -> Features -> [Projection] -> Tokens 
       -> [Latent Constraint] -> z -> [Predictor] -> z' 
       -> [Projection Decode] -> Features' -> [Vision Decode] -> Image'
```

**关键组件**:
- `vision`: 视觉编码器/解码器 (SekiroConv/SekiroResNet/MNISTConv)
- `projection`: 特征↔潜空间投影 (Linear/Spatial/Attention)
- `latent`: 隐空间约束 (VAE/VQ-VAE/VICReg)
- `predictor`: 可选的潜空间精炼器

### 2. 世界模型配置 (WorldConfig)

基于 **MLA (Multi-Head Latent Attention)** + **MoE (Mixture of Experts)** 架构:

```python
@dataclass
class WorldConfig():
    # 基础架构
    hidden_dim: int = 576
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 2
    
    # MLA 配置
    kv_lora_rank: int = 32    # KV 压缩秩
    q_lora_rank: int = 32     # Query 压缩秩
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 64
    v_head_dim: int = 64
    
    # MoE 配置
    num_experts: int = 8
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1
```

### 3. 视觉模块 (SekiroResNet)

```
输入：(B, 3, 128, 240)
  ↓
Encoder: 4 个 ResNet Stage (下采样至 4×8)
  ↓
Features: (B, 512, 4, 8)
  ↓
Projection: 空间投影至 (B, embedding_dim, 4, 8)
  ↓
Latent: VQ 量化或 VAE 重参数化
```

---

## 开发规范

### 代码风格

- **类型注解**: 推荐使用 Python 类型注解
- **命名约定**:
  - 类名：大驼峰 (PascalCase)
  - 函数/变量：小写 + 下划线 (snake_case)
  - 常量：全大写 + 下划线
- **文档字符串**: 公共 API 应包含 docstring

### 模型设计规范

1. **模块化**: 所有世界模型继承自基类
2. **组件复用**: 使用 `backbone/` 和 `components/` 中的通用组件
3. **配置分离**: 超参数通过 `configs/` 中的 dataclass 管理
4. **算子优化**: 核心计算密集型任务优先使用 CUDA 融合算子

### 测试实践

- 新模型应在 `tests/` 目录下添加单元测试
- 测试前向传播、损失计算、梯度流动

---

## 常见问题

### Windows 多进程数据加载

在 Windows 系统上，`DataLoader` 的 `num_workers` 应设置为 `0`:

```python
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
```

### 清理项目缓存

```powershell
Get-ChildItem -Path . -Filter "__pycache__" -Recurse | Remove-Item -Force -Recurse
```

### 输出目录说明

| 目录 | 内容 |
|------|------|
| `outputs/results/{task}/{model}/` | 每 epoch 的重建对比图 |
| `outputs/models/` | 保存的模型权重 (.pth) |
| `logs/{task}/{model}/` | TensorBoard 日志 |

---

## 扩展方向

1. **调整隐空间**: 修改 `latent_dim` 观察压缩效果
2. **β-VAE**: 调整 `beta` 参数控制 KL 散度权重
3. **多帧预测**: 扩展数据集支持输入 t 预测 t+1
4. **注意力机制**: 集成 `backbone/attention.py`
5. **MoE 集成**: 使用 `backbone/moe.py` 扩展模型容量
6. **新模型架构**: 参考 `scripts/mnist/` 模板实现新变体
