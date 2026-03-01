import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets


class MNIST_VAE_Dataset(Dataset):
    """
    用于 VAE 旋转预测任务的 MNIST 数据集。
    输入原图，目标是预测旋转一定角度后的图像。
    """
    def __init__(self, mnist_dataset, angle=45, angle_per_digit=True):
        self.mnist_dataset = mnist_dataset
        self.angle = angle
        self.angle_per_digit = angle_per_digit

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, label = self.mnist_dataset[idx]
        
        # 每个数字对应不同的旋转角度
        if self.angle_per_digit:
            curr_angle = label * self.angle
        else:
            curr_angle = self.angle
            
        rotate_transform = transforms.RandomRotation((curr_angle, curr_angle))
        target = rotate_transform(img)
        
        # 将角度转换为弧度
        angle_rad = curr_angle / 180.0 * 3.14159
        
        # 返回：原图, 旋转后的图, 标签, 旋转角度(弧度)
        return img, target, label, torch.tensor([angle_rad], dtype=torch.float32)


class MNIST_VQVAE_Dataset(Dataset):
    """
    用于 VQ-VAE 重建任务的标准 MNIST 数据集。
    输入原图，目标是重建原图。
    """
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, label = self.mnist_dataset[idx]
        # VQ-VAE 通常只需要原图进行重建
        return img, label


class MNIST_RSSM_Dataset(Dataset):
    """
    用于 RSSM 时序预测任务的 MNIST 数据集。
    生成一系列旋转的数字序列。
    """
    def __init__(self, mnist_dataset, seq_len=8, angle=15):
        self.mnist_dataset = mnist_dataset
        self.seq_len = seq_len
        self.angle = angle

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, _ = self.mnist_dataset[idx]
        
        sequence = []
        actions = [] 
        
        curr_img = img
        for i in range(self.seq_len):
            sequence.append(curr_img)
            # 动作：固定的旋转步长（弧度制）
            actions.append(torch.tensor([self.angle / 180.0 * 3.1415]))
            
            # 生成下一帧
            angle_rad = self.angle
            rotate_transform = transforms.RandomRotation((angle_rad, angle_rad))
            curr_img = rotate_transform(curr_img)
            
        return torch.stack(sequence), torch.stack(actions)


class MNIST_JEPA_Dataset(Dataset):
    """
    用于 JEPA 自监督学习的 MNIST 数据集。
    输入原图和随机旋转后的图，学习特征表示。
    """
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, label = self.mnist_dataset[idx]
        
        # 随机生成 0-360 度旋转
        import random
        curr_angle = random.uniform(0, 360)
        rotate_transform = transforms.RandomRotation((curr_angle, curr_angle))
        target = rotate_transform(img)
        
        # 将角度转换为弧度
        angle_rad = curr_angle / 180.0 * 3.14159
        
        # 返回：原图, 旋转后的图, 标签, 旋转角度(弧度)
        return img, target, label, torch.tensor([angle_rad], dtype=torch.float32)
