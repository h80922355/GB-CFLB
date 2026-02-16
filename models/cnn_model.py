"""
CNN模型定义模块 - 包含标准CNN和简易CNN
支持MNIST、EMNIST、CIFAR10数据集
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvNet(nn.Module):
    """
    简易CNN模型 - 与models.py定义完全一致
    用于MNIST和EMNIST数据集
    """

    def __init__(self, input_channels=1, num_classes=10, **kwargs):
        """
        初始化简易CNN模型

        参数:
            input_channels: 输入通道数（默认1用于灰度图）
            num_classes: 输出类别数
            **kwargs: 其他未使用的参数（保持接口一致性）
        """
        super(SimpleConvNet, self).__init__()

        # 与models.py完全一致的层定义
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x):
        """
        前向传播 - 与models.py完全一致

        参数:
            x: 输入张量 [batch_size, channels, height, width]

        返回:
            输出张量 [batch_size, num_classes]
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        return x

    def get_feature_dim(self):
        """
        获取特征维度（保持接口兼容性）

        返回:
            特征维度
        """
        return 256  # 返回fc1之前的维度

    def extract_features(self, x):
        """
        提取特征（保持接口兼容性）

        参数:
            x: 输入张量

        返回:
            特征向量
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        return x


class CNN(nn.Module):
    """
    标准CNN模型（保持原有实现）
    用于CIFAR10等复杂数据集
    """

    def __init__(self, input_channels=3, num_classes=10, image_size=32):
        """
        初始化CNN模型

        参数:
            input_channels: 输入通道数
            num_classes: 分类类别数
            image_size: 输入图像尺寸
        """
        super(CNN, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.image_size = image_size

        # 第一个卷积块：卷积->批归一化->ReLU->池化
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=32, eps=1e-5, affine=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积块：卷积->批归一化->ReLU->池化
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=64, eps=1e-5, affine=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三个卷积块：卷积->批归一化->ReLU->池化
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(num_groups=16, num_channels=128, eps=1e-5, affine=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层的输入维度
        fc_input = self._calculate_fc_input()

        # 全连接层
        self.fc1 = nn.Linear(fc_input, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def _calculate_fc_input(self):
        """计算全连接层的输入维度"""
        size_after_pool1 = self.image_size // 2
        size_after_pool2 = size_after_pool1 // 2
        size_after_pool3 = size_after_pool2 // 2

        fc_input = 128 * size_after_pool3 * size_after_pool3

        # 验证维度
        if self.image_size == 28:  # MNIST, EMNIST
            assert fc_input == 128 * 3 * 3, f"MNIST/EMNIST: 期望1152，实际{fc_input}"
        elif self.image_size == 32:  # CIFAR-10
            assert fc_input == 128 * 4 * 4, f"CIFAR-10: 期望2048，实际{fc_input}"

        return fc_input

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入图像，形状为[batch_size, channels, height, width]

        返回:
            输出预测结果，形状为[batch_size, num_classes]
        """
        # 第一个卷积块
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))

        # 第二个卷积块
        x = self.pool2(F.relu(self.gn2(self.conv2(x))))

        # 第三个卷积块
        x = self.pool3(F.relu(self.gn3(self.conv3(x))))

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

    def get_feature_dim(self):
        """
        获取特征维度（用于特征提取）

        返回:
            fc1层的输出维度
        """
        return 256

    def extract_features(self, x):
        """
        提取图像特征（不包括最后的分类层）

        参数:
            x: 输入图像

        返回:
            特征向量
        """
        # 卷积层
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        x = self.pool2(F.relu(self.gn2(self.conv2(x))))
        x = self.pool3(F.relu(self.gn3(self.conv3(x))))

        # 展平
        x = x.view(x.size(0), -1)

        # 第一个全连接层（特征层）
        features = F.relu(self.fc1(x))

        return features