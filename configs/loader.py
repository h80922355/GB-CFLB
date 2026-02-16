"""
配置加载器模块 - 增强版
负责读取和管理配置，支持MNIST、EMNIST、CIFAR10数据集
新增：恶意节点攻击配置支持
"""
import os
import yaml
from typing import Dict, Any


class ConfigLoader:
    """配置加载器 - 增强版"""

    # 默认配置值
    DEFAULT_CONFIG = {
        # 新增随机种子配置
        'random_seed': {
            'enabled': True,  # 是否启用随机种子设置
            'seed': 42,       # 默认随机种子值
            'use_deterministic': True,  # 是否使用确定性算法以保证可重现性
        },
        'dataset': {
            'name': 'MNIST',  # 数据集名称
            'root': './data/datasets',  # 数据集存储路径
            'train_size': None,  # 训练集大小限制
            'test_size': None,   # 测试集大小限制

            # Non-IID数据划分配置
            'partitioning': {
                'enabled': True,
                'method': 'dirichlet',
                'alpha': 0.1
            },

            # 标签交换配置
            'label_swap': {
                'enabled': False,
                'num_clusters': 4,
                'pairs': None
            },

            # 数据旋转配置
            'rotation': {
                'enabled': False,
                'num_clusters': 4,
                'rotation_angle': 90,
                'rotation_angles': None
            },

            # 新增：突发数据变化配置
            'sudden_change': {
                'enabled': False,
                'trigger_round': 40,
                'affected_ratio': 0.5,
                'additional_rotation': -1
            }
        },
        'clients': {
            'num_clients': 20,
            'batch_size': None,  # None表示自动调整
            'train_frac': 0.9,
        },
        'model': {
            'name': 'cnn',
        },
        'optimizer': {
            'name': 'SGD',
            'lr': 0.005,
            'momentum': 0.9
        },
        'training': {
            'communication_rounds': 200,
            'local_epochs': 3,
            'granular_ball': {
                'init_rounds': 50,
                'min_size': 5,
                'convergence_threshold': 0.95,
                'convergence_window_size': 3,
                'purity_threshold': 0.3,
                'purity_window_size': 3,
                'cooling_period': 3,
                'direction_threshold': 0.0,
                'data_similarity_threshold': 0.6,  # 新增：数据相似度阈值
                'similarity_threshold': 0.0
            }
        },
        # 新增：恶意节点攻击配置
        'attack': {
            'enabled': False,  # 攻击功能总开关
            'parameter_attack': {
                'enabled': False,  # 参数更新攻击开关
                'malicious_ratio': 0.2,  # 恶意客户端比例
                'gaussian_variance': 1.0,  # 高斯噪声方差
                'start_round': 1  # 开始攻击的轮次
            },
            'voting_attack': {
                'enabled': False,  # 投票反转攻击开关
                'malicious_ratio': 0.2,  # 恶意委员比例（与parameter_attack共用）
                'start_round': 1  # 开始攻击的轮次
            }
        },
        'visualization': {
            'plot_interval': 10,
            'save_models': True
        },
        'output': {
            'dir': 'outputs'
        },
        'blockchain': {
            'committee': {
                'size': 5,
                'consensus_threshold': 0.8,
                'rotation_interval': 10,
                'rotation_percentage': 0.4,
                'candidate_pool_percentage': 0.7
            },
            'validation': {
                'enabled': True,
                'standard_phase': {
                    'norm_threshold': 50.0,
                    'similarity_threshold': -1.0
                },
                'ball_phase': {
                    'norm_threshold': 10.0,
                    'similarity_threshold': -0.5
                }
            },
            'reward': {
                'base_reward': 2.0,
                'update_reward_factor': 0.5,
                'validation_reward_factor': 0.15,
                'governance_reward_factor': 0.2,
                'max_reward_coefficient': 4.0,
                'validation_penalty_ratio': 0.1,
                'block_failure_penalty_ratio': 0.25
            },
            'storage': {
                'path': './parameter_storage'
            }
        }
    }

    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化配置加载器"""
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        self._load_config()

    def _load_config(self) -> None:
        """加载并合并配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        self._update_dict(self.config, file_config)
        except Exception as e:
            print(f"警告: 加载配置文件失败 {self.config_path}: {e}")
            print("将使用默认配置值")

    def _update_dict(self, base_dict: dict, update_dict: dict) -> None:
        """递归更新字典"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._update_dict(base_dict[key], value)
            else:
                base_dict[key] = value

    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config

    def update_args(self, args) -> None:
        """使用配置更新命令行参数"""
        # 随机种子配置
        if not hasattr(args, 'seed') or args.seed is None:
            args.seed = self.config['random_seed'].get('seed', 42)

        # 是否启用确定性算法配置
        if not hasattr(args, 'use_deterministic'):
            args.use_deterministic = self.config['random_seed'].get('use_deterministic', True)

        # 客户端数量
        if args.n_clients is None:
            args.n_clients = self.config['clients']['num_clients']

        # 客户端配置
        if not hasattr(args, 'batch_size'):
            args.batch_size = self.config['clients'].get('batch_size', None)

        if not hasattr(args, 'train_frac'):
            args.train_frac = self.config['clients'].get('train_frac', 0.9)

        # 数据集划分参数
        if not hasattr(args, 'dirichlet_alpha') or args.dirichlet_alpha is None:
            args.dirichlet_alpha = self.config['dataset']['partitioning']['alpha']

        # 优化器参数
        if args.lr is None:
            args.lr = self.config['optimizer']['lr']

        if args.momentum is None:
            args.momentum = self.config['optimizer']['momentum']

        # 训练参数
        if args.communication_rounds is None:
            args.communication_rounds = self.config['training']['communication_rounds']

        if args.local_epochs is None:
            args.local_epochs = self.config['training']['local_epochs']

        if args.init_rounds is None:
            args.init_rounds = self.config['training']['granular_ball']['init_rounds']

        # 粒球参数
        if args.convergence_threshold is None:
            args.convergence_threshold = self.config['training']['granular_ball'].get('convergence_threshold', 0.95)

        if args.convergence_window_size is None:
            args.convergence_window_size = self.config['training']['granular_ball'].get('convergence_window_size', 3)

        if args.purity_threshold is None:
            args.purity_threshold = self.config['training']['granular_ball']['purity_threshold']

        if not hasattr(args, 'purity_window_size') or args.purity_window_size is None:
            args.purity_window_size = self.config['training']['granular_ball'].get('purity_window_size', 3)

        if args.min_size is None:
            args.min_size = self.config['training']['granular_ball']['min_size']

        if not hasattr(args, 'cooling_period') or args.cooling_period is None:
            args.cooling_period = self.config['training']['granular_ball'].get('cooling_period', 3)

        # 合并参数
        if args.direction_threshold is None:
            args.direction_threshold = self.config['training']['granular_ball'].get('direction_threshold', 0.0)

        # 新增：数据相似度阈值
        if not hasattr(args, 'data_similarity_threshold') or args.data_similarity_threshold is None:
            args.data_similarity_threshold = self.config['training']['granular_ball'].get('data_similarity_threshold', 0.6)

        # 重分配参数
        if args.similarity_threshold is None:
            args.similarity_threshold = self.config['training']['granular_ball'].get('similarity_threshold', 0.0)

        # 数据集配置
        if not hasattr(args, 'dataset_config') or args.dataset_config is None:
            args.dataset_config = self.config['dataset']

        # 模型配置
        if not hasattr(args, 'model_config') or args.model_config is None:
            args.model_config = self.config['model']

        # 可视化参数
        if args.plot_interval is None:
            args.plot_interval = self.config['visualization'].get('plot_interval', 10)

        # 输出目录
        if args.output_dir is None:
            args.output_dir = self.config['output'].get('dir', 'outputs')

        # 区块链相关参数
        if not hasattr(args, 'blockchain_config') or args.blockchain_config is None:
            args.blockchain_config = self.config['blockchain']

        # 新增：突发数据变化配置
        if not hasattr(args, 'sudden_change_config'):
            args.sudden_change_config = self.config['dataset'].get('sudden_change', {
                'enabled': False,
                'trigger_round': 40,
                'affected_ratio': 0.5,
                'additional_rotation': -1
            })

        # 新增：恶意节点攻击配置
        if not hasattr(args, 'attack_config'):
            args.attack_config = self.config.get('attack', {
                'enabled': False,
                'parameter_attack': {
                    'enabled': False,
                    'malicious_ratio': 0.2,
                    'gaussian_variance': 1.0,
                    'start_round': 1
                },
                'voting_attack': {
                    'enabled': False,
                    'malicious_ratio': 0.2,
                    'start_round': 1
                }
            })

    def get_seed_config(self) -> Dict[str, Any]:
        """获取随机种子配置"""
        return self.config.get('random_seed', {
            'enabled': True,
            'seed': 42,
            'use_deterministic': True
        })

    def get_dataset_size_limits(self) -> Dict[str, Any]:
        """获取数据集大小限制配置"""
        return {
            'train_size': self.config['dataset'].get('train_size', None),
            'test_size': self.config['dataset'].get('test_size', None)
        }

    def get_client_config(self) -> Dict[str, Any]:
        """获取客户端配置"""
        return {
            'batch_size': self.config['clients'].get('batch_size', None),
            'train_frac': self.config['clients'].get('train_frac', 0.9),
            'num_clients': self.config['clients'].get('num_clients', 20)
        }

    def get_sudden_change_config(self) -> Dict[str, Any]:
        """获取突发数据变化配置"""
        return self.config['dataset'].get('sudden_change', {
            'enabled': False,
            'trigger_round': 40,
            'affected_ratio': 0.5,
            'additional_rotation': -1
        })

    def get_attack_config(self) -> Dict[str, Any]:
        """获取恶意节点攻击配置"""
        return self.config.get('attack', {
            'enabled': False,
            'parameter_attack': {
                'enabled': False,
                'malicious_ratio': 0.2,
                'gaussian_variance': 1.0,
                'start_round': 1
            },
            'voting_attack': {
                'enabled': False,
                'malicious_ratio': 0.2,
                'start_round': 1
            }
        })