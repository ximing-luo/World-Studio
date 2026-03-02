import torch
import os
import glob
import numpy as np
from torch.utils.data import Dataset


import torch.nn.functional as F


def preprocess_sekiro_img(img, transform=None):
    """只狼图像预处理：NumPy -> Tensor (保持 uint8)"""
    img = torch.from_numpy(img) # 零拷贝转换，保持 uint8 类型
    if transform:
        img = transform(img)
    return img


class LazyNPYRecord:
    """
    延迟加载单个记录文件夹 (包含 obs.npy, action.npy)。
    使用内存映射 (mmap) 实现极致 I/O 性能。
    """
    def __init__(self, episode_dir, transform=None):
        self.episode_dir = episode_dir
        self.transform = transform
        
        # 内存映射文件
        self.obs_mmap = np.load(os.path.join(episode_dir, 'obs.npy'), mmap_mode='r')
        self.action_mmap = np.load(os.path.join(episode_dir, 'action.npy'), mmap_mode='r')
        
        # 假设 obs 形状为 (N, C, H, W) 或 (N, H, W, C)
        # 如果是 (N, H, W, C)，需要在 __getitem__ 中 transpose，但建议原生存为 (N, C, H, W)
        self._len = len(self.obs_mmap) - 1

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        # 读取当前帧和下一帧
        obs_raw = self.obs_mmap[i]
        next_obs_raw = self.obs_mmap[i+1]
        
        # 检查布局并转换 (如果原生不是 C, H, W)
        if obs_raw.shape[0] > 3: # 假设 H, W, C
            obs_raw = obs_raw.transpose(2, 0, 1)
            next_obs_raw = next_obs_raw.transpose(2, 0, 1)
            
        obs = preprocess_sekiro_img(obs_raw, self.transform)
        next_obs = preprocess_sekiro_img(next_obs_raw, self.transform)
        
        action = torch.from_numpy(self.action_mmap[i]).float()
        
        return {'obs': obs, 'next_obs': next_obs, 'action': action}


class Sekiro_BaseDataset(Dataset):
    """
    只狼数据集基类，支持文件夹嵌套的 NPY 结构。
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.records = [] 
        self._load_all_episodes()

    def _load_all_episodes(self):
        # 遍历 data_dir 下的所有子目录
        for root, dirs, files in os.walk(self.data_dir):
            if 'obs.npy' in files and 'action.npy' in files:
                try:
                    record = LazyNPYRecord(root, self.transform)
                    if len(record) > 0:
                        self.records.append(record)
                except Exception as e:
                    print(f"Error loading record in {root}: {e}")
        
        print(f"Loaded {len(self.records)} records from {self.data_dir}")


class Sekiro_GeneralDataset(Sekiro_BaseDataset):
    """
    通用重建数据集，将所有帧展平。
    """
    def __init__(self, data_dir, transform=None):
        super().__init__(data_dir, transform)
        self.frame_indices = [] # 存储 (record_idx, step_idx, key)
        for i, record in enumerate(self.records):
            for j in range(len(record)):
                self.frame_indices.append((i, j, 'obs'))
            # 加上最后一条记录的最后一帧
            self.frame_indices.append((i, len(record)-1, 'next_obs'))

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        record_idx, step_idx, key = self.frame_indices[idx]
        img = self.records[record_idx][step_idx][key]
        return img, img


class Sekiro_VAE_Dataset(Sekiro_BaseDataset):
    """
    预测未来帧数据集 (obs, target_obs, action)。
    默认预测 10 帧后（约 0.3s @ 30fps）。
    """
    def __init__(self, data_dir, frame_skip=10, transform=None):
        super().__init__(data_dir, transform)
        self.frame_skip = frame_skip
        self.pairs = [] # 存储 (record_idx, start_idx, end_idx)
        for i, record in enumerate(self.records):
            # record 里的每个 element 是 {'obs':..., 'next_obs':..., 'action':...}
            # 如果 frame_skip=1, 就是原来的逻辑（预测下一帧）
            # 如果 frame_skip=10, 我们用 record[i]['obs'] 预测 record[i+frame_skip-1]['next_obs']
            for j in range(len(record) - frame_skip + 1):
                self.pairs.append((i, j, j + frame_skip - 1))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        record_idx, start_idx, end_idx = self.pairs[idx]
        record = self.records[record_idx]
        
        obs = record[start_idx]['obs']
        target_obs = record[end_idx]['next_obs']
        action = record[start_idx]['action'] # 记录起始时刻的动作
        
        return obs, target_obs, action


class Sekiro_VQVAE_Dataset(Sekiro_VAE_Dataset):
    pass


class Sekiro_JEPA_Dataset(Sekiro_VAE_Dataset):
    pass


class Sekiro_RSSM_Dataset(Sekiro_BaseDataset):
    """
    时序预测数据集，返回长度为 seq_len 的图像和动作序列。
    支持 frame_skip 跳帧采样，例如 frame_skip=3 代表 3 帧取 1 帧（10fps @ 30fps）。
    """
    def __init__(self, data_dir, seq_len=8, frame_skip=1, transform=None):
        super().__init__(data_dir, transform)
        self.seq_len = seq_len
        self.frame_skip = frame_skip
        self.sequences = [] # 存储 (record_idx, start_idx)
        
        # 计算一个完整序列覆盖的总步数
        total_span = (seq_len - 1) * frame_skip + 1
        
        for i, record in enumerate(self.records):
            num_steps = len(record)
            if num_steps >= total_span:
                # 依然保持滑动窗口，但步长可以根据需要调整，目前保持 1 帧滑动以增加数据量
                for start_idx in range(num_steps - total_span + 1):
                    self.sequences.append((i, start_idx))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        record_idx, start_idx = self.sequences[idx]
        record = self.records[record_idx]
        
        imgs = []
        actions = []
        
        for k in range(self.seq_len):
            t = start_idx + k * self.frame_skip
            step = record[t]
            imgs.append(step['obs'])
            actions.append(step['action'])
            
        return torch.stack(imgs), torch.stack(actions)
