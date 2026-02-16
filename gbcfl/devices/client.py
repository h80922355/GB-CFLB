"""
客户端类定义模块 - 改进版
包含FederatedTrainingDevice基类和Client实现
新增：合并依据计算和管理功能
新增：恶意节点参数更新攻击功能
"""
import torch
import time
import math
from torch.utils.data import DataLoader
from torchvision import transforms
from gbcfl.utils.operations import copy, train_op, subtract_, eval_op, flatten
from gbcfl.utils.device_utils import get_device

# 获取设备
device = get_device()


class FederatedTrainingDevice(object):
    """
    联邦训练设备基类
    为Client提供基础功能
    """
    def __init__(self, model_fn, data):
        """
        初始化联邦训练设备

        参数:
            model_fn: 模型构造函数
            data: 训练数据
        """
        self.model = model_fn().to(device)
        self.data = data
        self.W = {key: value.to(device) for key, value in self.model.named_parameters()}

    def evaluate(self, loader=None):
        """
        评估模型性能

        参数:
            loader: 可选的自定义数据加载器

        返回:
            模型在验证集上的准确率
        """
        current_loader = self.eval_loader if hasattr(self, 'eval_loader') and not loader else loader
        if current_loader is None:
            return 0.0
        return eval_op(self.model, current_loader)


class Client(FederatedTrainingDevice):
    """
    联邦学习客户端类 - 改进版
    添加粒球模型同步功能、运行时数据变换更新和合并依据管理
    新增：恶意节点参数攻击功能
    """
    def __init__(self, model_fn, optimizer_fn, data, idnum, batch_size=128, train_frac=0.8,
                 rotation_angle=0, rotation_cluster_id=-1):
        """
        初始化客户端
        参数:
            rotation_angle: 初始旋转角度
            rotation_cluster_id: 所属旋转聚类ID
        """
        super().__init__(model_fn, data)

        self.model = self.model.to(device)
        self.optimizer = optimizer_fn(self.model.parameters())

        self.data = data
        self.original_data = data  # 保存原始数据引用
        self.batch_size = batch_size
        self.train_frac = train_frac

        # 记录旋转信息
        self.rotation_angle = rotation_angle
        self.rotation_cluster_id = rotation_cluster_id
        self.additional_rotation = 0  # 额外旋转角度

        # 创建数据加载器
        self._create_data_loaders()

        self.id = idnum

        # 模型参数更新
        self.dW = {key: torch.zeros_like(value).to(device) for key, value in self.model.named_parameters()}
        self.W_old = {key: torch.zeros_like(value).to(device) for key, value in self.model.named_parameters()}

        # 训练历史
        self.loss_history = []
        self.acc_updates = {key: torch.zeros_like(value).to(device) for key, value in self.model.named_parameters()}

        # 粒球归属
        self.ball_id = 0
        self.update_consistency = 0.0
        self.is_unmatched = False

        # 新增：合并依据管理
        self.merge_basis = None  # 扁平化的tensor
        self.merge_basis_initialized = False
        self.data_changed = False  # 标记数据是否发生变化

        # 新增：恶意节点标记和攻击配置
        self.is_malicious = False  # 是否为恶意客户端
        self.attack_config = None  # 攻击配置参数

    def set_malicious(self, is_malicious: bool, attack_config: dict = None):
        """
        设置客户端恶意身份和攻击配置

        参数:
            is_malicious: 是否为恶意客户端
            attack_config: 攻击配置参数
        """
        self.is_malicious = is_malicious
        self.attack_config = attack_config

    def _inject_gaussian_noise(self, current_round: int):
        """
        向模型更新参数注入高斯噪声

        参数:
            current_round: 当前训练轮次
        """
        if not self.is_malicious or self.attack_config is None:
            return

        # 检查是否启用参数攻击
        param_attack = self.attack_config.get('parameter_attack', {})
        if not param_attack.get('enabled', False):
            return

        # 检查是否达到攻击开始轮次
        if current_round < param_attack.get('start_round', 1):
            return

        # 获取高斯噪声方差
        gaussian_variance = param_attack.get('gaussian_variance', 1.0)
        gaussian_std = math.sqrt(gaussian_variance)  # 标准差 = sqrt(方差)

        # 对所有参数注入噪声
        for key in self.dW:
            if self.dW[key] is not None:
                # 生成与参数同形状的高斯噪声
                noise = torch.randn_like(self.dW[key]) * gaussian_std
                noise = noise.to(device)

                # 将噪声添加到更新参数
                self.dW[key].data += noise

    def _create_data_loaders(self):
        """创建或重建数据加载器"""
        n_train = int(len(self.data) * self.train_frac)
        n_eval = len(self.data) - n_train
        data_train, data_eval = torch.utils.data.random_split(self.data, [n_train, n_eval])

        self.train_loader = DataLoader(
            data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        self.eval_loader = DataLoader(
            data_eval,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

    def apply_additional_rotation(self, angle):
        """
        对训练数据应用额外的旋转（用于突发数据变化）
        新增：标记数据变化

        参数:
            angle: 额外的旋转角度
        """
        from data.transforms import RotationTransform

        self.additional_rotation = angle
        total_rotation = self.rotation_angle + self.additional_rotation

        # 标记数据发生变化
        self.data_changed = True

        # 创建组合变换（总旋转角度）
        if total_rotation != 0:
            # 根据数据集类型确定归一化参数
            if hasattr(self.original_data, 'dataset'):
                dataset_name = type(self.original_data.dataset).__name__
                if 'CIFAR' in dataset_name:
                    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                else:
                    normalize = transforms.Normalize((0.1307,), (0.3081,))
            else:
                normalize = transforms.Normalize((0.1307,), (0.3081,))

            new_transform = transforms.Compose([
                RotationTransform(total_rotation),
                transforms.ToTensor(),
                normalize
            ])
        else:
            # 无旋转，使用标准变换
            if hasattr(self.original_data, 'dataset'):
                dataset_name = type(self.original_data.dataset).__name__
                if 'CIFAR' in dataset_name:
                    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                else:
                    normalize = transforms.Normalize((0.1307,), (0.3081,))
            else:
                normalize = transforms.Normalize((0.1307,), (0.3081,))

            new_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

        # 更新数据集的transform
        if hasattr(self.data, 'transform'):
            self.data.transform = new_transform
        elif hasattr(self.data, 'dataset') and hasattr(self.data.dataset, 'transform'):
            self.data.dataset.transform = new_transform

        # 重建数据加载器
        self._create_data_loaders()

    def compute_merge_basis_with_anchor(self, anchor_model, epochs=None):
        """
        使用锚点模型计算合并依据

        参数:
            anchor_model: 锚点模型参数
            epochs: 训练轮数（如果为None则使用local_epochs）

        返回:
            合并依据（扁平化的tensor）
        """
        if anchor_model is None:
            print(f"警告：客户端{self.id}无法计算合并依据，锚点模型为空")
            return None

        # 保存当前模型状态
        original_W = {key: value.clone() for key, value in self.W.items()}

        # 设置锚点模型参数
        copy(target=self.W, source=anchor_model)

        # 使用锚点模型进行训练
        if epochs is None:
            # 使用完整的local_epochs
            from configs.loader import ConfigLoader
            config_loader = ConfigLoader()
            config = config_loader.get_config()
            epochs = config['training'].get('local_epochs', 3)

        # 执行训练
        self.compute_weight_update(epochs=epochs)

        # 提取更新参数作为合并依据
        merge_basis = flatten(self.dW)

        # 恢复原始模型状态
        copy(target=self.W, source=original_W)

        # 保存合并依据
        self.merge_basis = merge_basis
        self.merge_basis_initialized = True
        self.data_changed = False  # 重置数据变化标记

        return merge_basis

    def extract_merge_basis_from_update(self):
        """
        从当前的更新参数中提取合并依据
        用于阶段转换后第一轮的优化（复用训练结果）

        返回:
            合并依据（扁平化的tensor）
        """
        if self.dW is None:
            return None

        # 直接使用当前的更新参数作为合并依据
        merge_basis = flatten(self.dW)

        # 保存合并依据
        self.merge_basis = merge_basis
        self.merge_basis_initialized = True
        self.data_changed = False

        return merge_basis

    def synchronize_with_ball_model(self, ball_model_params):
        """
        从粒球模型同步参数
        参数:
            ball_model_params: 粒球模型参数字典
        """
        if ball_model_params is not None:
            copy(target=self.W, source=ball_model_params)
            for param in self.model.parameters():
                param.data = param.data.to(device)

    def update_ball_assignment(self, ball_id):
        """
        更新粒球归属
        参数:
            ball_id: 新的粒球ID，-1表示未分配
        """
        self.ball_id = ball_id
        self.is_unmatched = (ball_id == -1)

    def compute_weight_update(self, epochs=1, loader=None):
        """计算权重更新"""
        copy(target=self.W_old, source=self.W)

        self.model = self.model.to(device)
        self.optimizer.param_groups[0]["lr"] *= 1

        current_loader = self.train_loader if not loader else loader
        loss, batch_losses = train_op(self.model, current_loader, self.optimizer, epochs)

        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

        for key in self.acc_updates:
            self.acc_updates[key].data += self.dW[key].data.clone()

        self.loss_history.extend(batch_losses)

        return {
            "loss": loss,
            "batch_losses": batch_losses
        }

    def reset(self):
        """重置模型参数到上一轮的状态"""
        copy(target=self.W, source=self.W_old)
        for param in self.model.parameters():
            param.data = param.data.to(device)

    def prepare_update_message(self, current_round: int = 1):
        """
        准备参数更新消息
        新增：支持恶意节点攻击

        参数:
            current_round: 当前训练轮次

        返回:
            消息字典和参数更新
        """
        # 如果是恶意节点且满足攻击条件，注入噪声
        if self.is_malicious:
            self._inject_gaussian_noise(current_round)

        message = {
            "client_id": self.id,
            "ball_id": self.ball_id,
            "data_amount": len(self.data),
            "timestamp": time.time(),
            "is_unmatched": self.is_unmatched,
            "rotation_angle": self.rotation_angle,
            "rotation_cluster_id": self.rotation_cluster_id,
            "additional_rotation": self.additional_rotation,
            "merge_basis_initialized": self.merge_basis_initialized,  # 新增
            "data_changed": self.data_changed  # 新增
        }
        return message, self.dW