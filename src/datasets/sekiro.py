import torch
import os
import glob
import numpy as np
from torch.utils.data import Dataset


import torch.nn.functional as F


def preprocess_sekiro_img(img, transform=None, swap_rb=True, resize=(128, 240)):
    """只狼图像预处理：BGR->RGB, 缩放, 归一化到 [0, 1], 应用 transform"""
    if img.dim() == 4:
        img = img.squeeze(0)
    
    img = img.float()
    
    # BGR -> RGB (如果需要)
    if swap_rb and img.shape[0] == 3:
        img = img[[2, 1, 0], :, :]
    
    # 缩放处理
    if resize is not None:
        # F.interpolate 期待 (B, C, H, W)
        img = img.unsqueeze(0)
        img = F.interpolate(img, size=resize, mode='area')
        img = img.squeeze(0)
        
    # 归一化
    if img.max() > 1.0:
        img = img / 255.0
    
    img = torch.clamp(img, 0, 1)

    if transform:
        img = transform(img)
        
    return img


class LazyPTTrajectory:
    def __init__(self, pt_file, transform=None):
        self.data = torch.load(pt_file, map_location='cpu')
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        step = self.data[idx]
        obs = preprocess_sekiro_img(step['basic']['obs']['policy'], self.transform)
        next_obs = preprocess_sekiro_img(step['basic']['next_obs']['policy'], self.transform)
        action = step['basic']['action'].squeeze(0).float()
        return {'obs': obs, 'next_obs': next_obs, 'action': action}

class LazyNPYTrajectory:
    def __init__(self, npy_file, transform=None):
        self.npy_file = npy_file
        self.transform = transform
        self._data = None
        # 获取长度不需要保留句柄，获取完即关闭
        with np.load(npy_file, mmap_mode='r') as d:
            self._len = len(d) - 1 if d.ndim == 4 else 0

    def _load_data(self):
        if self._data is None:
            self._data = np.load(self.npy_file, mmap_mode='r')
        return self._data

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        data = self._load_data()
        obs = preprocess_sekiro_img(torch.from_numpy(data[i]), self.transform)
        next_obs = preprocess_sekiro_img(torch.from_numpy(data[i+1]), self.transform)
        action = torch.zeros(2)
        return {'obs': obs, 'next_obs': next_obs, 'action': action}

class LazyNPZTrajectory:
    def __init__(self, npz_file, transform=None):
        self.npz_file = npz_file
        self.transform = transform
        self._data = None
        # 获取长度
        with np.load(npz_file, mmap_mode='r') as d:
            self._len = len(d['frames']) - 1

    def _load_data(self):
        if self._data is None:
            self._data = np.load(self.npz_file, mmap_mode='r')
        return self._data

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        data = self._load_data()
        frames = data['frames']
        actions = data['actions']
        
        obs_np = frames[i].transpose(2, 0, 1)
        next_obs_np = frames[i+1].transpose(2, 0, 1)
        # .npz 已经是 RGB 格式，不需要交换通道
        obs = preprocess_sekiro_img(torch.from_numpy(obs_np), self.transform, swap_rb=False)
        next_obs = preprocess_sekiro_img(torch.from_numpy(next_obs_np), self.transform, swap_rb=False)
        action = torch.from_numpy(actions[i]).float()
        return {'obs': obs, 'next_obs': next_obs, 'action': action}

class Sekiro_BaseDataset(Dataset):
    """
    只狼数据集基类，提供统一的文件遍历和加载逻辑。
    使用延迟加载和内存映射 (mmap) 以减少内存占用。
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.trajectories = [] # 存储 LazyTrajectory 对象列表
        self._load_all_files()

    def _load_all_files(self):
        # 1. 加载 .pt 文件
        pt_files = glob.glob(os.path.join(self.data_dir, "*.pt"))
        for pt_file in pt_files:
            try:
                traj = LazyPTTrajectory(pt_file, self.transform)
                if len(traj) > 0:
                    self.trajectories.append(traj)
            except Exception as e:
                print(f"Error loading {pt_file}: {e}")

        # 2. 加载 .npy 文件
        npy_files = glob.glob(os.path.join(self.data_dir, "*.npy"))
        for npy_file in npy_files:
            try:
                traj = LazyNPYTrajectory(npy_file, self.transform)
                if len(traj) > 0:
                    self.trajectories.append(traj)
            except Exception as e:
                print(f"Error loading {npy_file}: {e}")

        # 3. 加载 .npz 文件 (由 record_world_model_data.py 保存)
        npz_files = glob.glob(os.path.join(self.data_dir, "*.npz"))
        for npz_file in npz_files:
            try:
                traj = LazyNPZTrajectory(npz_file, self.transform)
                if len(traj) > 0:
                    self.trajectories.append(traj)
            except Exception as e:
                print(f"Error loading {npz_file}: {e}")


class Sekiro_GeneralDataset(Sekiro_BaseDataset):
    """
    通用重建数据集，将所有帧展平。
    """
    def __init__(self, data_dir, transform=None):
        super().__init__(data_dir, transform)
        self.frame_indices = [] # 存储 (traj_idx, step_idx, key)
        for i, traj in enumerate(self.trajectories):
            for j in range(len(traj)):
                self.frame_indices.append((i, j, 'obs'))
            # 加上最后一条轨迹的最后一帧
            self.frame_indices.append((i, len(traj)-1, 'next_obs'))

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        traj_idx, step_idx, key = self.frame_indices[idx]
        img = self.trajectories[traj_idx][step_idx][key]
        return img, img


class Sekiro_VAE_Dataset(Sekiro_BaseDataset):
    """
    预测未来帧数据集 (obs, target_obs, action)。
    默认预测 10 帧后（约 0.3s @ 30fps）。
    """
    def __init__(self, data_dir, frame_skip=10, transform=None):
        super().__init__(data_dir, transform)
        self.frame_skip = frame_skip
        self.pairs = [] # 存储 (traj_idx, start_idx, end_idx)
        for i, traj in enumerate(self.trajectories):
            # traj 里的每个 element 是 {'obs':..., 'next_obs':..., 'action':...}
            # 如果 frame_skip=1, 就是原来的逻辑（预测下一帧）
            # 如果 frame_skip=10, 我们用 traj[i]['obs'] 预测 traj[i+frame_skip-1]['next_obs']
            for j in range(len(traj) - frame_skip + 1):
                self.pairs.append((i, j, j + frame_skip - 1))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        traj_idx, start_idx, end_idx = self.pairs[idx]
        traj = self.trajectories[traj_idx]
        
        obs = traj[start_idx]['obs']
        target_obs = traj[end_idx]['next_obs']
        action = traj[start_idx]['action'] # 记录起始时刻的动作
        
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
        self.sequences = [] # 存储 (traj_idx, start_idx)
        
        # 计算一个完整序列覆盖的总步数
        total_span = (seq_len - 1) * frame_skip + 1
        
        for i, traj in enumerate(self.trajectories):
            num_steps = len(traj)
            if num_steps >= total_span:
                # 依然保持滑动窗口，但步长可以根据需要调整，目前保持 1 帧滑动以增加数据量
                for start_idx in range(num_steps - total_span + 1):
                    self.sequences.append((i, start_idx))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        traj_idx, start_idx = self.sequences[idx]
        traj = self.trajectories[traj_idx]
        
        imgs = []
        actions = []
        
        for k in range(self.seq_len):
            t = start_idx + k * self.frame_skip
            step = traj[t]
            imgs.append(step['obs'])
            actions.append(step['action'])
            
        return torch.stack(imgs), torch.stack(actions)
