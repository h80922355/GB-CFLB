"""
统一数据集加载器 - 增强版
支持MNIST、EMNIST、CIFAR10、FEMNIST数据集
FEMNIST支持：自动检测JSON文件，验证用户数量匹配
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image
import random
import math
import json
from .transforms import LabelSwapTransform, RotationTransform


class FEMNISTDataset(Dataset):
    """
    FEMNIST数据集类
    从JSON文件加载预划分的用户数据
    """
    def __init__(self, user_data, transform=None):
        """
        初始化FEMNIST数据集

        参数:
            user_data: 包含'x'和'y'的字典，x是图像数据，y是标签
            transform: 数据变换
        """
        self.data = user_data['x']  # 图像数据列表
        self.targets = user_data['y']  # 标签列表
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """获取数据项"""
        # FEMNIST数据是28x28的灰度图像，存储为784维向量
        img_data = np.array(self.data[idx], dtype=np.float32)

        # 重塑为28x28图像
        img_data = img_data.reshape(28, 28)

        # 转换为PIL图像
        img = Image.fromarray((img_data * 255).astype(np.uint8), mode='L')

        # 应用变换
        if self.transform:
            img = self.transform(img)

        label = self.targets[idx]

        return img, label


class DatasetLoader:
    """
    统一数据集加载器
    支持MNIST、EMNIST、CIFAR10、FEMNIST数据集的加载和预处理
    FEMNIST增强：自动检测文件，验证用户数量
    """

    def __init__(self):
        """初始化数据集加载器"""
        self.supported_datasets = ['MNIST', 'EMNIST_DIGITS', 'EMNIST_BYCLASS',
                                   'EMNIST_LETTERS', 'CIFAR10', 'FEMNIST']

    def detect_femnist_files(self, femnist_dir):
        """
        自动检测FEMNIST数据文件

        参数:
            femnist_dir: FEMNIST数据目录

        返回:
            (train_files, test_files): 训练和测试文件列表
        """
        train_dir = os.path.join(femnist_dir, 'train')
        test_dir = os.path.join(femnist_dir, 'test')

        train_files = []
        test_files = []

        # 检测训练文件 (train_*.json格式)
        if os.path.exists(train_dir):
            train_pattern = os.path.join(train_dir, 'train_*.json')
            train_paths = sorted(glob.glob(train_pattern))
            train_files = [os.path.basename(f) for f in train_paths]

        # 检测测试文件 (test_*.json格式)
        if os.path.exists(test_dir):
            test_pattern = os.path.join(test_dir, 'test_*.json')
            test_paths = sorted(glob.glob(test_pattern))
            test_files = [os.path.basename(f) for f in test_paths]

        return train_files, test_files

    def count_femnist_users(self, femnist_dir, train_files):
        """
        统计FEMNIST数据集中的用户数量

        参数:
            femnist_dir: FEMNIST数据目录
            train_files: 训练文件列表

        返回:
            (user_count, user_info): 用户数量和详细信息
        """
        train_dir = os.path.join(femnist_dir, 'train')
        all_users = {}  # {user_id: {'num_samples': int, 'file': str}}

        for train_file in train_files:
            file_path = os.path.join(train_dir, train_file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    users = data.get('users', [])
                    user_data = data.get('user_data', {})

                    for user in users:
                        if user in user_data:
                            all_users[user] = {
                                'num_samples': len(user_data[user]['x']),
                                'file': train_file
                            }
            except Exception as e:
                print(f"警告: 读取文件 {file_path} 时出错: {e}")

        return len(all_users), all_users

    def load_dataset(self, dataset_config, n_clients):
        """
        加载指定数据集并进行预处理

        参数:
            dataset_config: 数据集配置字典
            n_clients: 客户端数量

        返回:
            client_data: 客户端数据列表
            test_data: 测试数据集
            mapp: 类别映射数组
        """
        dataset_name = dataset_config.get('name', 'MNIST').upper()
        root_dir = dataset_config.get('root', './data/datasets')

        if dataset_name not in self.supported_datasets:
            raise ValueError(f"不支持的数据集: {dataset_name}，支持的数据集: {self.supported_datasets}")

        print(f"正在加载{dataset_name}数据集...")

        # 确保数据目录存在
        os.makedirs(root_dir, exist_ok=True)

        # FEMNIST特殊处理
        if dataset_name == 'FEMNIST':
            return self._load_femnist_dataset(dataset_config, n_clients)

        # 加载原始数据集（其他数据集保持原有逻辑）
        train_dataset, test_dataset, mapp = self._load_raw_dataset(dataset_name, root_dir)

        # 应用数据集大小限制
        train_dataset, test_dataset = self._apply_size_limits(
            train_dataset, test_dataset, dataset_config
        )

        # 进行Non-IID数据划分
        client_indices = self._partition_data(
            train_dataset,
            n_clients,
            dataset_config.get('partitioning', {})
        )

        # 应用预处理操作
        client_data = self._apply_preprocessing(
            train_dataset,
            client_indices,
            dataset_config,
            dataset_name
        )

        # 创建测试数据集
        test_data = self._create_test_dataset(test_dataset, dataset_name)

        print(f"{dataset_name}数据集加载完成: {len(client_data)}个客户端")
        print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

        return client_data, test_data, mapp

    def _load_femnist_dataset(self, dataset_config, n_clients):
        """
        加载FEMNIST数据集
        自动检测文件，验证用户数量，一对一分配

        参数:
            dataset_config: 数据集配置
            n_clients: 客户端数量

        返回:
            client_data: 客户端数据列表
            test_data: 测试数据集
            mapp: 类别映射数组
        """
        root_dir = dataset_config.get('root', './data/datasets')
        femnist_dir = os.path.join(root_dir, 'FEMNIST')

        # 检查目录是否存在
        if not os.path.exists(femnist_dir):
            raise FileNotFoundError(
                f"FEMNIST数据集目录不存在: {femnist_dir}\n"
                f"请确保数据文件放置在正确的位置:\n"
                f"  - {os.path.join(femnist_dir, 'train', 'train_*.json')}\n"
                f"  - {os.path.join(femnist_dir, 'test', 'test_*.json')}"
            )

        # 步骤1: 自动检测数据文件
        print("检测FEMNIST数据文件...")
        train_files, test_files = self.detect_femnist_files(femnist_dir)

        if not train_files:
            raise FileNotFoundError(
                f"未找到任何训练文件！\n"
                f"请确保训练文件命名格式为 train_*.json 并放置在:\n"
                f"  {os.path.join(femnist_dir, 'train')}"
            )

        if not test_files:
            raise FileNotFoundError(
                f"未找到任何测试文件！\n"
                f"请确保测试文件命名格式为 test_*.json 并放置在:\n"
                f"  {os.path.join(femnist_dir, 'test')}"
            )

        print(f"  找到 {len(train_files)} 个训练文件: {train_files}")
        print(f"  找到 {len(test_files)} 个测试文件: {test_files}")

        # 步骤2: 统计用户数量
        print("统计用户数量...")
        user_count, user_info = self.count_femnist_users(femnist_dir, train_files)

        if user_count == 0:
            raise ValueError(
                f"FEMNIST数据集中没有找到任何有效用户数据！\n"
                f"请检查JSON文件格式是否正确。"
            )

        print(f"  FEMNIST数据集包含 {user_count} 个用户")

        # 计算用户样本统计
        sample_counts = [info['num_samples'] for info in user_info.values()]
        print(f"  用户样本数: 最小={min(sample_counts)}, "
              f"最大={max(sample_counts)}, "
              f"平均={np.mean(sample_counts):.1f}")

        # 步骤3: 验证用户数量是否匹配
        if user_count < n_clients:
            raise ValueError(
                f"\n" + "="*60 + "\n"
                f"错误: 用户数量不匹配！\n"
                f"  - FEMNIST数据集包含: {user_count} 个用户\n"
                f"  - 配置文件要求: {n_clients} 个客户端\n"
                f"\n解决方案:\n"
                f"  1. 修改配置文件，设置 num_clients <= {user_count}\n"
                f"  2. 添加更多的FEMNIST数据文件 (train_{len(train_files)}.json, train_{len(train_files)+1}.json ...)\n"
                f"="*60
            )

        if user_count > n_clients:
            print(f"  注意: 数据集有 {user_count} 个用户，但只使用前 {n_clients} 个")

        # 步骤4: 加载用户数据
        print("加载用户数据...")
        train_dir = os.path.join(femnist_dir, 'train')
        all_train_users = []

        for train_file in train_files:
            file_path = os.path.join(train_dir, train_file)
            print(f"  读取文件: {train_file}")

            with open(file_path, 'r') as f:
                data = json.load(f)
                users = data.get('users', [])
                user_data = data.get('user_data', {})

                for user in users:
                    if user in user_data:
                        all_train_users.append({
                            'user_id': user,
                            'data': user_data[user],
                            'num_samples': len(user_data[user]['x']),
                            'source_file': train_file
                        })

            # 如果已经有足够的用户，可以提前停止
            if len(all_train_users) >= n_clients:
                print(f"  已加载足够的用户数据 ({n_clients}个)")
                break

        # 步骤5: 为每个客户端分配一个用户（一对一）
        print("分配客户端数据...")
        client_data = []

        # 创建数据变换
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
        ]
        combined_transform = transforms.Compose(transform_list)

        # 为每个客户端创建数据集
        for i in range(n_clients):
            user_info = all_train_users[i]
            user_data = user_info['data']

            # 创建FEMNIST数据集
            client_dataset = FEMNISTDataset(user_data, transform=combined_transform)

            client_info_dict = {
                'dataset': client_dataset,
                'label_swap': None,
                'rotation_angle': 0,
                'user_id': user_info['user_id'],
                'num_samples': user_info['num_samples'],
                'source_file': user_info['source_file']
            }

            client_data.append(client_info_dict)

            # 每10个客户端打印一次进度
            if (i + 1) % 10 == 0 or (i + 1) == n_clients:
                print(f"    已分配 {i + 1}/{n_clients} 个客户端")

        # 步骤6: 加载测试数据
        print("加载测试数据...")
        test_dir = os.path.join(femnist_dir, 'test')
        all_test_data = {'x': [], 'y': []}

        for test_file in test_files:
            file_path = os.path.join(test_dir, test_file)
            print(f"  读取文件: {test_file}")

            with open(file_path, 'r') as f:
                data = json.load(f)
                user_data = data.get('user_data', {})

                # 合并所有用户的测试数据
                for user, udata in user_data.items():
                    if 'x' in udata and 'y' in udata:
                        all_test_data['x'].extend(udata['x'])
                        all_test_data['y'].extend(udata['y'])

        # 创建测试数据集
        test_data = FEMNISTDataset(all_test_data, transform=combined_transform)

        # FEMNIST有62个类别（10个数字 + 26个大写字母 + 26个小写字母）
        mapp = np.array([str(i) for i in range(10)] +
                       [chr(i) for i in range(ord('A'), ord('Z') + 1)] +
                       [chr(i) for i in range(ord('a'), ord('z') + 1)], dtype='<U1')

        # 打印最终统计
        print("\n" + "="*60)
        print("FEMNIST数据集加载完成:")
        print(f"  ✓ 客户端数量: {len(client_data)}")
        print(f"  ✓ 每个客户端一个独立用户")
        client_samples = [c['num_samples'] for c in client_data]
        print(f"  ✓ 客户端样本数: 最小={min(client_samples)}, "
              f"最大={max(client_samples)}, "
              f"平均={np.mean(client_samples):.1f}")
        print(f"  ✓ 测试集大小: {len(test_data)} 个样本")
        print("="*60 + "\n")

        return client_data, test_data, mapp

    def get_femnist_info(self, dataset_config):
        """
        获取FEMNIST数据集信息（公开方法）

        参数:
            dataset_config: 数据集配置

        返回:
            包含数据集信息的字典
        """
        root_dir = dataset_config.get('root', './data/datasets')
        femnist_dir = os.path.join(root_dir, 'FEMNIST')

        info = {
            'available': False,
            'train_files': [],
            'test_files': [],
            'user_count': 0,
            'user_details': {},
            'error': None
        }

        try:
            # 检查目录
            if not os.path.exists(femnist_dir):
                info['error'] = f"FEMNIST目录不存在: {femnist_dir}"
                return info

            # 检测文件
            train_files, test_files = self.detect_femnist_files(femnist_dir)
            info['train_files'] = train_files
            info['test_files'] = test_files

            if not train_files:
                info['error'] = "未找到训练文件"
                return info

            # 统计用户
            user_count, user_details = self.count_femnist_users(femnist_dir, train_files)
            info['user_count'] = user_count
            info['user_details'] = user_details
            info['available'] = True

        except Exception as e:
            info['error'] = str(e)

        return info

    def _apply_size_limits(self, train_dataset, test_dataset, dataset_config):
        """
        应用数据集大小限制

        参数:
            train_dataset: 训练数据集
            test_dataset: 测试数据集
            dataset_config: 数据集配置

        返回:
            限制大小后的训练和测试数据集
        """
        train_size = dataset_config.get('train_size', None)
        test_size = dataset_config.get('test_size', None)

        # 限制训练集大小
        if train_size is not None and train_size > 0:
            if train_size < len(train_dataset):
                # 随机选择指定数量的样本
                indices = list(range(len(train_dataset)))
                random.shuffle(indices)
                selected_indices = indices[:train_size]
                train_dataset = Subset(train_dataset, selected_indices)
                print(f"训练集大小限制为: {train_size}")

        # 限制测试集大小
        if test_size is not None and test_size > 0:
            if test_size < len(test_dataset):
                # 随机选择指定数量的样本
                indices = list(range(len(test_dataset)))
                random.shuffle(indices)
                selected_indices = indices[:test_size]
                test_dataset = Subset(test_dataset, selected_indices)
                print(f"测试集大小限制为: {test_size}")

        return train_dataset, test_dataset

    def _load_raw_dataset(self, dataset_name, root_dir):
        """加载原始数据集"""
        if dataset_name == 'MNIST':
            train_dataset = datasets.MNIST(root=root_dir, train=True, download=True, transform=None)
            test_dataset = datasets.MNIST(root=root_dir, train=False, download=True, transform=None)
            mapp = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype='<U1')

        elif dataset_name == 'EMNIST_DIGITS':
            train_dataset = datasets.EMNIST(root=root_dir, split='digits', train=True, download=True, transform=None)
            test_dataset = datasets.EMNIST(root=root_dir, split='digits', train=False, download=True, transform=None)
            mapp = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype='<U1')

        elif dataset_name == 'EMNIST_BYCLASS':
            train_dataset = datasets.EMNIST(root=root_dir, split='byclass', train=True, download=True, transform=None)
            test_dataset = datasets.EMNIST(root=root_dir, split='byclass', train=False, download=True, transform=None)
            # byclass包含数字0-9、大写字母A-Z、小写字母a-z
            mapp = np.array([str(i) for i in range(10)] +
                            [chr(i) for i in range(ord('A'), ord('Z') + 1)] +
                            [chr(i) for i in range(ord('a'), ord('z') + 1)], dtype='<U1')

        elif dataset_name == 'EMNIST_LETTERS':
            train_dataset = datasets.EMNIST(root=root_dir, split='letters', train=True, download=True, transform=None)
            test_dataset = datasets.EMNIST(root=root_dir, split='letters', train=False, download=True, transform=None)
            # letters只包含大写字母A-Z (26个类别)
            mapp = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)], dtype='<U1')

        elif dataset_name == 'CIFAR10':
            train_dataset = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=None)
            test_dataset = datasets.CIFAR10(root=root_dir, train=False, download=True, transform=None)
            mapp = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer',
                             'dog', 'frog', 'horse', 'ship', 'truck'], dtype='<U10')

        return train_dataset, test_dataset, mapp

    def _partition_data(self, dataset, n_clients, partition_config):
        """使用Dirichlet分布进行Non-IID数据划分"""
        if not partition_config.get('enabled', True):
            # IID划分
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            client_indices = np.array_split(indices, n_clients)
            return [indices.tolist() for indices in client_indices]

        # 获取标签
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        else:
            # 处理Subset包装的数据集
            if isinstance(dataset, Subset):
                original_dataset = dataset.dataset
                if hasattr(original_dataset, 'targets'):
                    all_labels = np.array(original_dataset.targets)
                    labels = all_labels[dataset.indices]
                elif hasattr(original_dataset, 'labels'):
                    all_labels = np.array(original_dataset.labels)
                    labels = all_labels[dataset.indices]
                else:
                    labels = np.array([original_dataset[dataset.indices[i]][1] for i in range(len(dataset))])
            else:
                labels = np.array([dataset[i][1] for i in range(len(dataset))])

        # Dirichlet分布划分
        alpha = partition_config.get('alpha', 1)
        n_classes = len(np.unique(labels))

        # 为每个类别计算Dirichlet分布
        label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

        # 按类别分组数据索引
        class_indices = [np.where(labels == y)[0] for y in range(n_classes)]

        # 分配给客户端
        client_indices = [[] for _ in range(n_clients)]

        for c, indices in enumerate(class_indices):
            # 根据Dirichlet分布分割这个类别的数据
            proportions = label_distribution[c]
            split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]

            class_splits = np.split(indices, split_points)

            for i, split in enumerate(class_splits):
                if i < n_clients:
                    client_indices[i].extend(split.tolist())

        # 随机打乱每个客户端的数据
        for client_data in client_indices:
            random.shuffle(client_data)

        return client_indices

    def _apply_preprocessing(self, dataset, client_indices, dataset_config, dataset_name):
        """应用预处理操作（标签交换和旋转）"""
        # 第一阶段：为每个客户端分配变换参数
        # 分配标签交换参数
        client_configs = self._apply_label_swaps(
            client_indices,
            dataset_config.get('label_swap', {})
        )

        # 分配旋转参数
        client_configs = self._apply_rotations(
            client_configs,
            dataset_config.get('rotation', {})
        )

        # 第二阶段：创建带有实际变换的客户端数据集
        client_data = []
        for i, client_config in enumerate(client_configs):
            indices = client_config['indices']
            label_swap = client_config.get('label_swap', None)
            rotation_angle = client_config.get('rotation_angle', 0)

            # 创建数据变换链
            transform_list = []

            # 添加旋转变换（如果需要）
            if rotation_angle != 0:
                transform_list.append(RotationTransform(rotation_angle))

            # 添加ToTensor变换
            transform_list.append(transforms.ToTensor())

            # 添加标准化
            if dataset_name in ['MNIST', 'EMNIST_DIGITS', 'EMNIST_BYCLASS', 'EMNIST_LETTERS']:
                transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
            elif dataset_name == 'CIFAR10':
                transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

            # 创建组合变换
            combined_transform = transforms.Compose(transform_list)

            # 创建标签变换（如果需要）
            label_transform = None
            if label_swap:
                label_transform = LabelSwapTransform(label_swap)

            # 创建客户端数据集
            client_dataset = CustomSubset(
                dataset,
                indices,
                combined_transform,
                label_transform
            )

            # 存储客户端信息
            client_info_dict = {
                'dataset': client_dataset,
                'label_swap': label_swap,
                'rotation_angle': rotation_angle
            }

            client_data.append(client_info_dict)

        return client_data

    def _apply_label_swaps(self, client_indices, label_swap_config):
        """应用标签交换"""
        if not label_swap_config.get('enabled', False):
            # 如果不启用标签交换，直接返回原始索引
            return [{'indices': indices} for indices in client_indices]

        # 获取聚类数量
        num_clusters = label_swap_config.get('num_clusters', 4)
        n_clients = len(client_indices)

        # 生成标签交换对
        label_pairs = label_swap_config.get('pairs', None)
        if label_pairs is None:
            # 自动生成标签交换对
            label_pairs = self._generate_label_pairs(num_clusters)

        # 将客户端分配到聚类
        clients_per_cluster = max(1, n_clients // num_clusters)
        client_indices_with_swaps = []

        for i, indices in enumerate(client_indices):
            cluster_id = i // clients_per_cluster
            cluster_id = min(cluster_id, len(label_pairs) - 1)  # 确保不超出范围

            # 获取该聚类的标签交换对
            label_swap = label_pairs[cluster_id] if cluster_id < len(label_pairs) else None

            client_info = {'indices': indices, 'label_swap': label_swap}
            client_indices_with_swaps.append(client_info)

        return client_indices_with_swaps

    def _apply_rotations(self, client_indices_with_swaps, rotation_config):
        """应用旋转"""
        if not rotation_config.get('enabled', False):
            return client_indices_with_swaps

        num_clusters = rotation_config.get('num_clusters', 4)
        rotation_angles = rotation_config.get('rotation_angles', None)

        if rotation_angles is None:
            # 使用统一旋转角度
            base_angle = rotation_config.get('rotation_angle', 90)
            rotation_angles = [i * base_angle for i in range(num_clusters)]

        n_clients = len(client_indices_with_swaps)
        clients_per_cluster = max(1, n_clients // num_clusters)

        for i, client_info in enumerate(client_indices_with_swaps):
            cluster_id = i // clients_per_cluster
            cluster_id = min(cluster_id, len(rotation_angles) - 1)

            rotation_angle = rotation_angles[cluster_id] if cluster_id < len(rotation_angles) else 0

            client_info['rotation_angle'] = rotation_angle
            client_info['rotation_cluster_id'] = cluster_id

        return client_indices_with_swaps

    def _generate_label_pairs(self, num_clusters, max_labels=10):
        """生成随机标签交换对"""
        label_pairs = []
        available_labels = list(range(max_labels))

        for _ in range(num_clusters):
            if len(available_labels) >= 2:
                pair = random.sample(available_labels, 2)
                label_pairs.append((pair[0], pair[1]))
                available_labels.remove(pair[0])
                available_labels.remove(pair[1])
            else:
                break

        return label_pairs

    def _create_test_dataset(self, test_dataset, dataset_name):
        """创建测试数据集"""
        # 创建基本变换
        transform_list = [transforms.ToTensor()]

        # 添加标准化
        if dataset_name in ['MNIST', 'EMNIST_DIGITS', 'EMNIST_BYCLASS', 'EMNIST_LETTERS']:
            transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        elif dataset_name == 'CIFAR10':
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        test_transform = transforms.Compose(transform_list)

        # 创建测试数据集
        test_data = CustomSubset(
            test_dataset,
            list(range(len(test_dataset))),
            test_transform,
            None
        )

        return test_data


class CustomSubset(Dataset):
    """
    自定义数据子集类
    支持自定义数据转换和标签转换
    """

    def __init__(self, dataset, indices, transform=None, label_transform=None):
        """
        初始化自定义子集

        参数:
            dataset: 原始数据集
            indices: 子集索引
            transform: 数据变换
            label_transform: 标签变换
        """
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.label_transform = label_transform

    def __getitem__(self, idx):
        """获取数据项"""
        x, y = self.dataset[self.indices[idx]]

        # 应用标签变换
        if self.label_transform:
            y = self.label_transform(y)

        # 应用数据变换
        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        """返回数据集大小"""
        return len(self.indices)