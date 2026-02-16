"""
数据变换和预处理模块
包含标签交换和旋转变换的实现
"""
import torch
from PIL import Image
import numpy as np


class LabelSwapTransform:
    """标签交换变换"""

    def __init__(self, label_swap_pair):
        """
        初始化标签交换变换

        参数:
            label_swap_pair: 标签交换对，格式为(label1, label2)
        """
        self.label_swap_pair = label_swap_pair

    def __call__(self, label):
        """应用标签交换"""
        if self.label_swap_pair is None:
            return label

        label1, label2 = self.label_swap_pair

        if label == label1:
            return label2
        elif label == label2:
            return label1
        else:
            return label


class RotationTransform:
    """图像旋转变换"""

    def __init__(self, angle):
        """
        初始化旋转变换

        参数:
            angle: 旋转角度（度）
        """
        self.angle = angle

    def __call__(self, img):
        """应用旋转变换"""
        if self.angle == 0:
            return img

        # 确保输入是PIL图像
        if isinstance(img, torch.Tensor):
            # 如果是tensor，转换为PIL图像
            if img.dim() == 3:
                # 3维tensor
                if img.shape[0] == 1:
                    # 灰度图像 (1, H, W)
                    img_np = img.squeeze(0).numpy()
                    mode = 'L'
                elif img.shape[0] == 3:
                    # RGB图像 (3, H, W) -> (H, W, 3)
                    img_np = img.permute(1, 2, 0).numpy()
                    mode = 'RGB'
                else:
                    raise ValueError(f"不支持的通道数: {img.shape[0]}")
            elif img.dim() == 2:
                # 灰度图像 (H, W)
                img_np = img.numpy()
                mode = 'L'
            else:
                raise ValueError(f"不支持的tensor维度: {img.dim()}")

            # 确保数值范围正确
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            img = Image.fromarray(img_np, mode=mode)

        elif isinstance(img, np.ndarray):
            # 判断图像类型
            if img.ndim == 2:
                # 灰度图像
                mode = 'L'
            elif img.ndim == 3:
                if img.shape[2] == 3:
                    # RGB图像
                    mode = 'RGB'
                elif img.shape[2] == 1:
                    # 灰度图像 (H, W, 1) -> (H, W)
                    img = img.squeeze(2)
                    mode = 'L'
                else:
                    raise ValueError(f"不支持的通道数: {img.shape[2]}")
            else:
                raise ValueError(f"不支持的数组维度: {img.ndim}")

            # 确保数值范围正确
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            img = Image.fromarray(img, mode=mode)

        elif not isinstance(img, Image.Image):
            raise TypeError(f"不支持的图像类型: {type(img)}")

        # 执行旋转
        # 对于RGB图像，使用白色填充；对于灰度图像，使用黑色填充
        fillcolor = (255, 255, 255) if img.mode == 'RGB' else 0
        rotated_img = img.rotate(self.angle, expand=False, fillcolor=fillcolor)

        return rotated_img