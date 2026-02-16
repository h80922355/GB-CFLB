"""
模型工厂模块 - 改进版
根据数据集自动选择合适的模型架构
MNIST使用简易CNN，CIFAR10/EMNIST/FEMNIST使用标准CNN
"""
import torch
import torch.nn as nn

from .cnn_model import CNN, SimpleConvNet
from .lstm_model import LSTM


class ModelFactory:
    """
    模型工厂类，根据数据集自动选择最优模型架构
    """

    # 数据集配置映射
    DATASET_CONFIGS = {
        'mnist': {
            'input_channels': 1,
            'num_classes': 10,
            'image_size': 28,
            'use_simple_cnn': True  # MNIST使用简易CNN
        },
        'emnist_digits': {
            'input_channels': 1,
            'num_classes': 10,
            'image_size': 28,
            'use_simple_cnn': False  # EMNIST使用标准CNN
        },
        'emnist_byclass': {
            'input_channels': 1,
            'num_classes': 62,
            'image_size': 28,
            'use_simple_cnn': False  # EMNIST使用标准CNN
        },
        'emnist_letters': {
            'input_channels': 1,
            'num_classes': 26,
            'image_size': 28,
            'use_simple_cnn': False  # EMNIST使用标准CNN
        },
        'femnist': {
            'input_channels': 1,
            'num_classes': 62,  # 10个数字 + 26个大写字母 + 26个小写字母
            'image_size': 28,
            'use_simple_cnn': False  # FEMNIST使用标准CNN
        },
        'cifar10': {
            'input_channels': 3,
            'num_classes': 10,
            'image_size': 32,
            'use_simple_cnn': False  # CIFAR10使用标准CNN
        }
    }

    @staticmethod
    def create_model(model_config):
        """
        创建指定类型的模型，自动根据数据集选择最优架构

        参数:
            model_config: 模型配置字典，包含模型类型和参数

        返回:
            模型构造函数
        """
        dataset_name = model_config.get('dataset', 'mnist').lower()
        model_name = model_config.get('name', 'cnn').lower()

        # 获取数据集配置
        if dataset_name in ModelFactory.DATASET_CONFIGS:
            dataset_config = ModelFactory.DATASET_CONFIGS[dataset_name]
        else:
            print(f"警告: 未知数据集 '{dataset_name}'，使用MNIST默认配置")
            dataset_config = ModelFactory.DATASET_CONFIGS['mnist']

        # 从模型配置中覆盖默认值（如果提供）
        input_channels = model_config.get('input_channels', dataset_config['input_channels'])
        num_classes = model_config.get('num_classes', dataset_config['num_classes'])
        image_size = model_config.get('image_size', dataset_config['image_size'])

        # 根据数据集决定是否使用简易CNN
        use_simple_cnn = dataset_config.get('use_simple_cnn', False)

        # CNN模型处理
        if model_name in ['cnn', 'convnet', '']:
            if use_simple_cnn:
                # MNIST使用简易CNN（与models.py完全一致）
                print(f"为{dataset_name.upper()}数据集使用简易CNN模型")
                return lambda: SimpleConvNet(
                    input_channels=input_channels,
                    num_classes=num_classes
                )
            else:
                # CIFAR10/EMNIST/FEMNIST使用标准CNN
                print(f"为{dataset_name.upper()}数据集使用标准CNN模型")
                return lambda: CNN(
                    input_channels=input_channels,
                    num_classes=num_classes,
                    image_size=image_size
                )

        elif model_name == 'lstm':
            # LSTM模型（主要用于序列任务，但也可用于图像分类）
            input_size = image_size * image_size * input_channels  # 展平的图像尺寸
            print(f"为{dataset_name.upper()}数据集使用LSTM模型")
            return lambda: LSTM(
                input_size=input_size,
                hidden_size=256,
                num_layers=2,
                num_classes=num_classes
            )

        else:
            raise ValueError(f"不支持的模型类型: {model_name}")

    @staticmethod
    def get_default_config(dataset_name):
        """
        获取数据集的默认模型配置

        参数:
            dataset_name: 数据集名称

        返回:
            默认模型配置字典
        """
        dataset_name = dataset_name.lower()

        if dataset_name in ModelFactory.DATASET_CONFIGS:
            config = ModelFactory.DATASET_CONFIGS[dataset_name].copy()
            config['name'] = 'cnn'
            config['dataset'] = dataset_name
            return config
        else:
            # 未知数据集使用MNIST配置
            return {
                'name': 'cnn',
                'dataset': dataset_name,
                'input_channels': 1,
                'num_classes': 10,
                'image_size': 28,
                'use_simple_cnn': True
            }

    @staticmethod
    def get_supported_datasets():
        """
        获取支持的数据集列表

        返回:
            支持的数据集名称列表
        """
        return list(ModelFactory.DATASET_CONFIGS.keys())

    @staticmethod
    def get_supported_models():
        """
        获取支持的模型类型列表

        返回:
            支持的模型类型列表
        """
        return ['cnn', 'lstm']

    @staticmethod
    def validate_config(model_config):
        """
        验证模型配置的有效性

        参数:
            model_config: 模型配置字典

        返回:
            (is_valid, error_message)
        """
        dataset_name = model_config.get('dataset', '').lower()
        model_name = model_config.get('name', '').lower()

        # 检查数据集
        if dataset_name and dataset_name not in ModelFactory.DATASET_CONFIGS:
            return False, f"不支持的数据集: {dataset_name}"

        # 检查模型类型
        if model_name and model_name not in ['cnn', 'convnet', 'lstm', '']:
            return False, f"不支持的模型类型: {model_name}"

        return True, None

    @staticmethod
    def get_model_info(dataset_name):
        """
        获取指定数据集的模型信息

        参数:
            dataset_name: 数据集名称

        返回:
            模型信息字典
        """
        dataset_name = dataset_name.lower()

        if dataset_name in ModelFactory.DATASET_CONFIGS:
            config = ModelFactory.DATASET_CONFIGS[dataset_name]
            model_type = "简易CNN" if config.get('use_simple_cnn', False) else "标准CNN"

            return {
                'dataset': dataset_name.upper(),
                'model_type': model_type,
                'input_channels': config['input_channels'],
                'num_classes': config['num_classes'],
                'image_size': config['image_size'],
                'description': f"{dataset_name.upper()}数据集使用{model_type}模型"
            }

        return None