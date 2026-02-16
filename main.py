"""
主入口文件 - 改进版
使用临时粒球机制和完整共识流程
新增：支持突发数据变化和改进的合并机制
新增：恶意节点攻击模拟支持
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

import argparse
import torch
import numpy as np
import random
import time
from colorama import Fore, Style, init

# 导入项目模块
from configs.loader import ConfigLoader
from data.dataset_loader import DatasetLoader
from models.model_factory import ModelFactory
from gbcfl.utils.device_utils import get_device_manager, print_device_info
from gbcfl.utils.logger import ExperimentLogger, ensure_output_dir
from gbcfl.visualization.console_output import print_title, print_info, print_success, print_warning
from gbcfl.devices.client import Client
from gbcfl.training.bc_trainer_improved import train_bc_gbcfl_improved

def setup_reproducibility(seed, use_deterministic=True):
    """设置随机种子"""
    print_info(f"设置随机种子: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if use_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def select_malicious_clients(n_clients, malicious_ratio, seed=None):
    """
    选择恶意客户端

    参数:
        n_clients: 总客户端数量
        malicious_ratio: 恶意客户端比例
        seed: 随机种子（用于确保可重现性）

    返回:
        恶意客户端ID列表
    """
    if malicious_ratio <= 0:
        return []

    # 设置临时随机种子确保可重现性
    if seed is not None:
        random_state = random.getstate()
        random.seed(seed)

    # 计算恶意客户端数量
    n_malicious = max(1, int(n_clients * malicious_ratio))

    # 随机选择恶意客户端
    all_client_ids = list(range(n_clients))
    malicious_clients = random.sample(all_client_ids, n_malicious)

    # 恢复随机状态
    if seed is not None:
        random.setstate(random_state)

    return sorted(malicious_clients)

def main(args):
    """主函数"""
    init(autoreset=True)

    print_title("GB-CFLB系统启动")
    print_info("基于临时粒球机制的区块链联邦学习系统")

    # 设备信息检测
    print_device_info()

    # 获取配置
    config_loader = ConfigLoader(args.config)
    config_loader.update_args(args)

    # 设置随机种子
    seed_config = config_loader.get_seed_config()
    if seed_config.get('enabled', True):
        setup_reproducibility(args.seed, seed_config.get('use_deterministic', True))
        print_success("✓ 随机种子设置完成")

    # 创建输出目录
    output_dir = args.output_dir
    ensure_output_dir(output_dir)

    # 数据集准备
    print_title("数据集准备")
    dataset_loader = DatasetLoader()

    try:
        client_data, test_data, mapp = dataset_loader.load_dataset(
            args.dataset_config, args.n_clients
        )
    except Exception as e:
        print_warning(f"数据集加载失败: {e}")
        return

    # 模型准备
    print_title("初始化模型和客户端")
    model_config = {
        'name': args.model_config.get('name', 'cnn'),
        'dataset': args.dataset_config.get('name', 'MNIST')
    }

    model_func = ModelFactory.create_model(model_config)

    # 获取设备
    device_manager = get_device_manager()
    device = device_manager.get_device()

    # 确定batch_size
    config_batch_size = getattr(args, 'batch_size', None)
    if config_batch_size is None or config_batch_size <= 0:
        if device.type == 'cuda':
            batch_size = 128
        else:
            batch_size = 64
    else:
        batch_size = config_batch_size

    # 获取攻击配置
    attack_config = args.attack_config
    malicious_client_ids = []

    # 如果启用攻击，选择恶意客户端
    if attack_config.get('enabled', False):
        print_title("恶意节点配置")

        # 使用parameter_attack的恶意比例（两种攻击共用相同的恶意客户端）
        malicious_ratio = attack_config['parameter_attack'].get('malicious_ratio', 0.2)

        # 选择恶意客户端（使用固定种子确保可重现）
        malicious_client_ids = select_malicious_clients(
            args.n_clients,
            malicious_ratio,
            seed=args.seed + 1000  # 使用不同的种子避免与其他随机过程冲突
        )

        print_info(f"恶意客户端比例: {malicious_ratio * 100:.0f}%")
        print_info(f"恶意客户端数量: {len(malicious_client_ids)}")

        if len(malicious_client_ids) <= 20:
            print_info(f"恶意客户端ID: {malicious_client_ids}")
        else:
            print_info(f"恶意客户端ID: {malicious_client_ids[:10]}...{malicious_client_ids[-10:]}")

        # 输出攻击配置
        if attack_config['parameter_attack'].get('enabled', False):
            print_info("参数更新攻击已启用:")
            print_info(f"  高斯噪声方差: {attack_config['parameter_attack']['gaussian_variance']}")
            print_info(f"  开始轮次: {attack_config['parameter_attack']['start_round']}")

        if attack_config['voting_attack'].get('enabled', False):
            print_info("投票反转攻击已启用:")
            print_info(f"  开始轮次: {attack_config['voting_attack']['start_round']}")

    # 创建客户端（传递旋转信息和恶意标记）
    clients = []
    for i, client_info in enumerate(client_data):
        client_dataset = client_info['dataset']
        rotation_angle = client_info.get('rotation_angle', 0)
        rotation_cluster_id = client_info.get('rotation_cluster_id', -1)

        optimizer_fn = lambda params: torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)

        client = Client(
            model_fn=model_func,
            optimizer_fn=optimizer_fn,
            data=client_dataset,
            idnum=i,
            batch_size=batch_size,
            train_frac=getattr(args, 'train_frac', 0.9),
            rotation_angle=rotation_angle,
            rotation_cluster_id=rotation_cluster_id
        )

        # 设置恶意身份
        if i in malicious_client_ids:
            client.set_malicious(True, attack_config)

        clients.append(client)

    print_info(f"客户端初始化完成: {len(clients)}个客户端")

    # 输出突发变化配置信息（如果启用）
    if args.sudden_change_config.get('enabled', False):
        print_info("突发数据变化已启用:")
        print_info(f"  触发轮次: {args.sudden_change_config['trigger_round']}")
        print_info(f"  受影响比例: {args.sudden_change_config['affected_ratio'] * 100:.0f}%")
        additional_rotation = args.sudden_change_config['additional_rotation']
        if additional_rotation == -1:
            print_info(f"  额外旋转: 自动计算")
        else:
            print_info(f"  额外旋转: {additional_rotation}°")

    # 输出合并策略配置信息
    print_info("合并策略配置:")
    print_info(f"  模型相似度阈值: {args.direction_threshold}")
    print_info(f"  数据相似度阈值: {args.data_similarity_threshold}")

    # 实验记录器
    cfl_stats = ExperimentLogger()

    # 开始训练
    print_title("开始训练")
    start_time = time.time()

    try:
        # 使用改进的训练器（传入test_data、model_func和恶意客户端列表）
        cfl_stats = train_bc_gbcfl_improved(
            clients, test_data, model_func, args, cfl_stats,
            malicious_client_ids=malicious_client_ids
        )

        end_time = time.time()
        total_time = end_time - start_time

        print_title("训练完成")
        print_info(f"总训练时间: {total_time / 60:.2f} 分钟")
        print_info(f"结果已保存到: {output_dir}")

    except Exception as e:
        print_warning(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GB-CFLB改进系统")

    # 基础参数
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--n-clients", type=int, default=None,
                        help="客户端数量")
    parser.add_argument("--dirichlet-alpha", type=float, default=None,
                        help="Dirichlet分布参数")

    # 优化器参数
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率")
    parser.add_argument("--momentum", type=float, default=None,
                        help="SGD动量参数")

    # 训练参数
    parser.add_argument("--communication-rounds", type=int, default=None,
                        help="通信轮数")
    parser.add_argument("--local-epochs", type=int, default=None,
                        help="本地训练轮数")
    parser.add_argument("--init-rounds", type=int, default=None,
                        help="初始标准FL轮数")

    # 粒球参数
    parser.add_argument("--convergence-threshold", type=float, default=None,
                        help="收敛阈值")
    parser.add_argument("--convergence-window-size", type=int, default=None,
                        help="收敛窗口大小")
    parser.add_argument("--purity-threshold", type=float, default=None,
                        help="纯度阈值")
    parser.add_argument("--purity-window-size", type=int, default=None,
                        help="纯度窗口大小")
    parser.add_argument("--min-size", type=int, default=None,
                        help="最小粒球大小")
    parser.add_argument("--cooling-period", type=int, default=None,
                        help="冷却期")
    parser.add_argument("--direction-threshold", type=float, default=None,
                        help="模型方向相似度阈值")
    parser.add_argument("--data-similarity-threshold", type=float, default=None,
                        help="数据相似度阈值")  # 新增参数
    parser.add_argument("--similarity-threshold", type=float, default=None,
                        help="相似度阈值")

    # 其他参数
    parser.add_argument("--plot-interval", type=int, default=None,
                        help="绘图间隔")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="输出目录")

    args = parser.parse_args()
    main(args)