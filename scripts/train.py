import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from src.train.trainer import Trainer
from src.model.vae import VAE
from configs.vae_mnist import vae_config


def main():
    # 初始化模型和配置
    model = VAE()
    config = vae_config

    # 训练
    trainer = Trainer(model, config)
    trainer.train()


if __name__ == '__main__':
    main()
