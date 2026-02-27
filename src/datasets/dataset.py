import torch
import os
import glob
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


# 定义旋转数据集，模拟"世界模型"的下一帧预测
class RotatedMNIST(Dataset):
    def __init__(self, mnist_dataset, angle=45, angle_per_digit=False, random_angle=False):
        """
        Args:
            mnist_dataset: 原始 MNIST 数据集
            angle: 基础旋转角度（当 angle_per_digit/random_angle=False 时使用）
            angle_per_digit: 如果为 True，每个数字对应不同旋转角度
            random_angle: 如果为 True，每次采样随机生成 0-360 度旋转
        """
        self.mnist_dataset = mnist_dataset
        self.angle = angle
        self.angle_per_digit = angle_per_digit
        self.random_angle = random_angle
        # 固定旋转角度，模拟确定的物理规则
        self.rotate = transforms.RandomRotation((angle, angle))

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, label = self.mnist_dataset[idx]
        
        if self.random_angle:
            # 随机生成 0-360 度旋转
            import random
            curr_angle = random.uniform(0, 360)
            rotate_transform = transforms.RandomRotation((curr_angle, curr_angle))
            target = rotate_transform(img)
        elif self.angle_per_digit:
            # 每个数字对应不同的旋转角度
            curr_angle = label * self.angle
            rotate_transform = transforms.RandomRotation((curr_angle, curr_angle))
            target = rotate_transform(img)
        else:
            # 使用固定旋转角度
            curr_angle = self.angle
            target = self.rotate(img)
        
        # 将角度转换为弧度
        angle_rad = curr_angle / 180.0 * 3.14159
        
        # 返回：原图, 旋转后的图, 标签, 旋转角度(弧度)
        return img, target, label, torch.tensor([angle_rad], dtype=torch.float32)


# 定义时序旋转数据集，用于训练 RSSM 等具备记忆的模型
class SequentialRotatedMNIST(Dataset):
    def __init__(self, mnist_dataset, seq_len=5, angle=15):
        """
        Args:
            mnist_dataset: 原始 MNIST 数据集
            seq_len: 序列长度 (T)
            angle: 每一步旋转的角度 (固定角速度)
        """
        self.mnist_dataset = mnist_dataset
        self.seq_len = seq_len
        self.angle = angle

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, _ = self.mnist_dataset[idx]
        
        sequence = []
        actions = [] # 记录每一步做的“动作”，这里动作就是旋转角度
        
        curr_img = img
        for i in range(self.seq_len):
            sequence.append(curr_img)
            # 动作：将角度转化为 one-hot 或 连续值
            # 这里简单处理：动作就是这个固定的旋转步长
            actions.append(torch.tensor([self.angle / 180.0 * 3.1415])) # 弧度制
            
            # 生成下一帧
            angle_rad = self.angle
            rotate_transform = transforms.RandomRotation((angle_rad, angle_rad))
            curr_img = rotate_transform(curr_img)
            
        # sequence: (T, 1, 28, 28)
        # actions: (T, 1)
        return torch.stack(sequence), torch.stack(actions)


# 定义只狼数据集，从保存的 .pt 轨迹中提取画面
class SekiroDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # # 加载目录下所有的 .pt 文件
        # pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
        # for pt_file in pt_files:
        #     try:
        #         data = torch.load(pt_file)
        #         # 每个 .pt 文件是一个列表，每个元素包含一步的数据
        #         for step in data:
        #             if 'basic' in step and 'obs' in step['basic']:
        #                 obs = step['basic']['obs']
        #                 if isinstance(obs, dict) and 'policy' in obs:
        #                     # 提取图像数据 [1, 3, 136, 240] -> [3, 136, 240]
        #                     img = obs['policy'].squeeze(0)
                            
        #                     # 转换颜色空间：只狼记录的数据通常是 BGR (OpenCV 格式)，
        #                     # 而 plt.imshow 和 PIL 预期的是 RGB，因此需要翻转通道。
        #                     if img.shape[0] == 3:
        #                         img = img[[2, 1, 0], :, :]
                            
        #                     # 在加载时应用 transform 以节省内存
        #                     if self.transform:
        #                         img = self.transform(img)
                            
        #                     self.samples.append(img)
        #     except Exception as e:
        #         print(f"Error loading {pt_file}: {e}")
        
        # 加载目录下所有的 .npy 文件 (兼容 benchmark_obs.npy)
        npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
        for npy_file in npy_files:
            try:
                data = np.load(npy_file, allow_pickle=True)
                # 兼容不同维度的 npy：(Batch, C, H, W) 或 单张 (C, H, W)
                if data.ndim == 4: # (Batch, 3, 136, 240)
                    for i in range(data.shape[0]):
                        img = torch.from_numpy(data[i]).float()
                        # 统一缩放到 [0, 1]
                        if img.max() > 1.0:
                            img = img / 255.0
                        
                        # 强制执行 BGR -> RGB 转换，因为 benchmark_obs.npy 
                        # 通常也是 OpenCV 捕获的 BGR 格式
                        if img.shape[0] == 3:
                            img = img[[2, 1, 0], :, :]
                            
                        if self.transform:
                            img = self.transform(img)
                        self.samples.append(img)
                elif data.ndim == 3: # (3, 136, 240)
                    img = torch.from_numpy(data).float()
                    if img.max() > 1.0:
                        img = img / 255.0
                    
                    if img.shape[0] == 3:
                        img = img[[2, 1, 0], :, :]
                        
                    if self.transform:
                        img = self.transform(img)
                    self.samples.append(img)
            except Exception as e:
                print(f"Error loading {npy_file}: {e}")
        
        # print(f"SekiroDataset: Loaded {len(self.samples)} samples from {len(pt_files)} .pt files and {len(npy_files)} .npy files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        # 简单任务：重建原图，所以 target = img
        return img, img
