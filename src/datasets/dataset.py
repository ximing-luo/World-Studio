# 定义旋转数据集，模拟“世界模型”的下一帧预测
class RotatedMNIST(Dataset):
    def __init__(self, mnist_dataset, angle=45):
        self.mnist_dataset = mnist_dataset
        self.angle = angle
        # 固定旋转角度，模拟确定的物理规则
        self.rotate = transforms.RandomRotation((angle, angle))

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, _ = self.mnist_dataset[idx]
        # 输入是当前帧，目标是旋转后的下一帧
        target = self.rotate(img)
        return img, target
