import torch
import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms


# 定义旋转数据集，模拟"世界模型"的下一帧预测
class RotatedMNIST(Dataset):
    def __init__(self, mnist_dataset, angle=45, angle_per_digit=False):
        """
        Args:
            mnist_dataset: 原始 MNIST 数据集
            angle: 基础旋转角度（当 angle_per_digit=False 时使用）
            angle_per_digit: 如果为 True，每个数字对应不同旋转角度
                           数字 i 的旋转角度 = i * angle (0→0°, 1→10°, ..., 9→90°)
        """
        self.mnist_dataset = mnist_dataset
        self.angle = angle
        self.angle_per_digit = angle_per_digit
        # 固定旋转角度，模拟确定的物理规则
        self.rotate = transforms.RandomRotation((angle, angle))

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, label = self.mnist_dataset[idx]
        
        if self.angle_per_digit:
            # 每个数字对应不同的旋转角度：数字 i → i * angle 度
            digit_angle = label * self.angle
            rotate_transform = transforms.RandomRotation((digit_angle, digit_angle))
            target = rotate_transform(img)
        else:
            # 使用固定旋转角度
            target = self.rotate(img)
        
        # 输入是当前帧，目标是旋转后的下一帧
        return img, target


# 定义只狼数据集，从保存的 .pt 轨迹中提取画面
class SekiroDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # 加载目录下所有的 .pt 文件
        pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
        for pt_file in pt_files:
            try:
                data = torch.load(pt_file)
                # 每个 .pt 文件是一个列表，每个元素包含一步的数据
                for step in data:
                    if 'basic' in step and 'obs' in step['basic']:
                        obs = step['basic']['obs']
                        if isinstance(obs, dict) and 'policy' in obs:
                            # 提取图像数据 [1, 3, 136, 240] -> [3, 136, 240]
                            img = obs['policy'].squeeze(0)
                            
                            # 转换颜色空间：只狼记录的数据通常是 BGR (OpenCV 格式)，
                            # 而 plt.imshow 和 PIL 预期的是 RGB，因此需要翻转通道。
                            if img.shape[0] == 3:
                                img = img[[2, 1, 0], :, :]
                            
                            # 在加载时应用 transform 以节省内存
                            if self.transform:
                                img = self.transform(img)
                            
                            self.samples.append(img)
            except Exception as e:
                print(f"Error loading {pt_file}: {e}")
        
        print(f"SekiroDataset: Loaded {len(self.samples)} samples from {len(pt_files)} files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        # 简单任务：重建原图，所以 target = img
        return img, img
