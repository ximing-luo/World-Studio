# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

World Model Studio is a PyTorch-based deep learning research framework focused on building **world models** that learn internal representations of environment dynamics and predict future states. The project implements multiple architectures (VAE, VQ-VAE, JEPA, RSSM) with interchangeable backbone networks (FC, Conv, ResNet) and features a modular design for vision, projection, latent constraints, and prediction components.

## Environment Setup

### Dependencies
```bash
pip install torch torchvision numpy matplotlib tensorboard
```

No additional package management files exist; dependencies are minimal as listed above.

### CUDA Operators
Custom CUDA kernels for normalization (RMSNorm2d, LayerNorm2d) and evolution layers (ECR) are automatically compiled via PyTorch's JIT compilation when first imported. No separate build step is required.

### Datasets
- **MNIST**: Automatically downloaded by `torchvision.datasets.MNIST` to `data/MNIST/raw/`.
- **Sekiro**: Expected in `data/Sekiro/recordings/` as `.npy` files; see `src/datasets/sekiro.py` for details.

## Common Commands

### Training
```bash
# MNIST tasks
python scripts/mnist/train_vae.py
python scripts/mnist/train_vqvae.py
python scripts/mnist/train_jepa.py
python scripts/mnist/train_rssm.py

# Sekiro tasks (full‑resolution game frames)
python scripts/sekiro/train_vae.py
python scripts/sekiro/train_vqvae.py
python scripts/sekiro/train_jepa.py
python scripts/sekiro/train_rssm.py
```

### Testing
```bash
# Install pytest if not already available
pip install pytest

# Run all unit tests
python -m pytest tests/

# Run a specific test file
python tests/test_refactored_vae.py
```

### Monitoring
```bash
# Start TensorBoard (logs are saved under logs/{task}/{model}/)
tensorboard --logdir=logs/sekiro/vae
```

### Model Profiling
```bash
# Analyze model parameters, memory, and FLOPs
python scripts/tools/prof_model.py
```

### CUDA Kernel Verification
```bash
# Test custom CUDA operators
python src/model/ecr/cuda_evolution/utils_compile.py
```

## High‑Level Architecture

The framework follows a **modular pipeline**:

1. **Vision** (`src/world/vision/`) – Task‑specific encoders/decoders (MNISTConv, MNISTResNet, SekiroConv, SekiroResNet)
2. **Projection** (`src/world/projection/`) – Transforms feature maps to/from token space (LinearProjection, SpatialProjection)
3. **Latent** (`src/world/latents/`) – Imposes constraints on the latent space (VAELatent, VQLatent, VICRegLatent)
4. **Dream** (`src/world/dream/`) – High‑level world‑model wrappers (StaticReconstruction for VAE/VQ‑VAE, JEPAFramework, RSSMFramework)
5. **Predictor** (`src/world/predictor/`) – Optional temporal/spatial refinement modules

### Model Families
| Architecture | Latent Space | Backbone Options | Use Case |
|--------------|--------------|------------------|----------|
| **VAE**      | Continuous (Gaussian) | FC / Conv / ResNet | Reconstruction, generation |
| **VQ‑VAE**   | Discrete (codebook)   | FC / Conv / ResNet | Compression, planning |
| **JEPA**     | Embedding             | FC / Conv / ResNet | Self‑supervised prediction |
| **RSSM**     | State‑space (RNN)     | FC / Conv / ResNet | Sequential decision‑making |

### Advanced Components
- **MLA (Multi‑Head Latent Attention)** + **MoE (Mixture of Experts)** – Inspired by DeepSeek, configurable via `WorldConfig` dataclass.
- **CUDA‑fused operators** – `ECR` (Efficient Cross‑Residual) and `Norm2D` (RMSNorm2d / LayerNorm2d) kernels for visual tensors.

## Project Structure Highlights

```
World‑Studio/
├── scripts/                     # Training scripts per task (mnist/, sekiro/)
├── src/
│   ├── model/                  # Core model components
│   │   ├── backbone/           # Attention, MoE, Transformer blocks
│   │   ├── components/         # Reusable layers (norm, resnet, loss)
│   │   ├── ecr/                # CUDA evolution operators
│   │   └── world/              # Base model classes (BaseVAE, BaseJEPA, …)
│   └── world/                  # Modular world‑model framework
│       ├── vision/             # Task‑specific encoders/decoders
│       ├── projection/         # Feature‑token projections
│       ├── latents/            # VAE/VQ/VICReg constraints
│       ├── dream/              # High‑level wrappers
│       └── predictor/          # Prediction modules
├── configs/                    # Configuration dataclasses (WorldConfig)
├── outputs/                    # Results (reconstruction images) and saved models
├── logs/                       # TensorBoard logs
└── tests/                      # Unit tests
```

## Configuration

Hyperparameters are managed via dataclasses in `configs/world.py`. The main `WorldConfig` defines dimensions, attention heads, MLA/MoE settings, and sequence length. Subclasses `VisionThinkingConfig` and `PredictorConfig` tailor the architecture for spatial understanding and temporal prediction respectively.

Example configuration snippet:
```python
from configs.world import WorldConfig
cfg = WorldConfig(
    hidden_dim=576,
    n_layer=8,
    n_head=8,
    kv_lora_rank=32,
    num_experts=8,
    num_experts_per_tok=2,
)
```

## Development Notes

### Windows Multiprocessing
On Windows, set `num_workers=0` in `DataLoader` to avoid memory spikes:
```python
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
```

### Module Import Paths
Training scripts add the project root to `sys.path` to resolve imports. When writing new scripts, follow the pattern:
```python
import os, sys
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
```

### CUDA Kernel Development
Custom CUDA operators are located in:
- `src/model/ecr/cuda_evolution/` – Evolution and cross‑fusion kernels
- `src/model/components/cuda_norm/` – Normalization kernels

Use `utils_compile.py` to verify correctness against PyTorch reference implementations.

### Adding a New Model
1. Inherit from the appropriate base class in `src/model/world/` (e.g., `BaseVAE`).
2. Implement `encode()` / `decode()` / `forward()`.
3. Assemble components using the modular framework in `src/world/dream/`.
4. Add a training script in `scripts/{task}/`.
5. Write unit tests in `tests/`.

### Output Directories
- `outputs/results/{task}/{model}/epoch_{n}.png` – Reconstruction comparison images
- `outputs/models/{task}_{model}.pth` – Saved model weights
- `logs/{task}/{model}/{timestamp}/` – TensorBoard logs

## Further Reading
- `README.md` – Project overview, installation, and extended examples.
- `QWEN.md` – Additional technical documentation (if present).