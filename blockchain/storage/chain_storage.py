"""
链下存储管理器
负责区块链数据和模型参数的持久化存储
新增：锚点模型和客户端合并依据存储功能
"""
import os
import json
import time
import torch
import pickle
from typing import Dict, Any, Optional
from gbcfl.utils.logger import ensure_output_dir
from gbcfl.utils.operations import flatten


class ChainStorage:
    """
    链下存储管理器类
    增强：支持锚点模型和合并依据的存储
    """

    def __init__(self, base_dir='outputs'):
        """
        初始化链存储管理器
        参数:
            base_dir: 基础存储目录
        """
        self.base_dir = base_dir
        self.blocks_dir = os.path.join(base_dir, 'blocks')
        self.models_dir = os.path.join(base_dir, 'models')
        self.checkpoints_dir = os.path.join(base_dir, 'checkpoints')
        self.reputation_dir = os.path.join(base_dir, 'reputation')

        # 新增：锚点模型和合并依据目录
        self.anchors_dir = os.path.join(base_dir, 'anchors')
        self.merge_basis_dir = os.path.join(base_dir, 'merge_basis')

        # 创建必要的目录
        os.makedirs(self.blocks_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.reputation_dir, exist_ok=True)
        os.makedirs(self.anchors_dir, exist_ok=True)
        os.makedirs(self.merge_basis_dir, exist_ok=True)

        # 内存缓存
        self.anchor_model = None  # 缓存的锚点模型
        self.client_merge_basis = {}  # 缓存的客户端合并依据

    def save_anchor_model(self, model_params: Dict, round_num: int):
        """
        保存锚点模型（阶段转换时的最终全局模型）

        参数:
            model_params: 模型参数字典
            round_num: 保存时的轮次
        """
        # 保存到文件
        anchor_file = os.path.join(self.anchors_dir, 'anchor_model.pt')

        # 确保参数在CPU上并detach
        cpu_params = {}
        for key, value in model_params.items():
            if isinstance(value, torch.Tensor):
                cpu_params[key] = value.cpu().detach().clone()
            else:
                cpu_params[key] = value

        # 保存模型参数
        torch.save({
            'model_params': cpu_params,
            'round': round_num,
            'timestamp': time.time()
        }, anchor_file)

        # 更新内存缓存
        self.anchor_model = cpu_params

        # 保存元数据
        metadata_file = os.path.join(self.anchors_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'anchor_round': round_num,
                'timestamp': time.time(),
                'model_size': sum(p.numel() for p in cpu_params.values() if isinstance(p, torch.Tensor))
            }, f, indent=2)

        print(f"锚点模型已保存 (轮次 {round_num})")

    def load_anchor_model(self) -> Optional[Dict]:
        """
        加载锚点模型

        返回:
            模型参数字典，如果不存在则返回None
        """
        # 优先从缓存返回
        if self.anchor_model is not None:
            return self.anchor_model

        anchor_file = os.path.join(self.anchors_dir, 'anchor_model.pt')
        if os.path.exists(anchor_file):
            checkpoint = torch.load(anchor_file, map_location='cpu')
            self.anchor_model = checkpoint['model_params']
            return self.anchor_model

        return None

    def save_client_merge_basis(self, client_id: int, merge_basis: torch.Tensor, round_num: int):
        """
        保存客户端合并依据

        参数:
            client_id: 客户端ID
            merge_basis: 合并依据（扁平化的tensor）
            round_num: 当前轮次
        """
        # 保存到文件
        basis_file = os.path.join(self.merge_basis_dir, f'client_{client_id}.pt')

        # 确保tensor在CPU上
        if merge_basis.is_cuda:
            merge_basis_cpu = merge_basis.cpu().detach().clone()
        else:
            merge_basis_cpu = merge_basis.detach().clone()

        # 保存
        torch.save({
            'merge_basis': merge_basis_cpu,
            'client_id': client_id,
            'round': round_num,
            'timestamp': time.time()
        }, basis_file)

        # 更新内存缓存
        self.client_merge_basis[client_id] = merge_basis_cpu

    def load_client_merge_basis(self, client_id: int) -> Optional[torch.Tensor]:
        """
        加载客户端合并依据

        参数:
            client_id: 客户端ID

        返回:
            合并依据tensor，如果不存在则返回None
        """
        # 优先从缓存返回
        if client_id in self.client_merge_basis:
            return self.client_merge_basis[client_id]

        basis_file = os.path.join(self.merge_basis_dir, f'client_{client_id}.pt')
        if os.path.exists(basis_file):
            checkpoint = torch.load(basis_file, map_location='cpu')
            merge_basis = checkpoint['merge_basis']
            # 更新缓存
            self.client_merge_basis[client_id] = merge_basis
            return merge_basis

        return None

    def batch_save_merge_basis(self, basis_dict: Dict[int, torch.Tensor], round_num: int):
        """
        批量保存客户端合并依据

        参数:
            basis_dict: {client_id: merge_basis}
            round_num: 当前轮次
        """
        for client_id, merge_basis in basis_dict.items():
            self.save_client_merge_basis(client_id, merge_basis, round_num)

        # 更新元数据
        metadata_file = os.path.join(self.merge_basis_dir, 'metadata.json')
        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        metadata.update({
            'last_update_round': round_num,
            'total_clients': len(basis_dict),
            'timestamp': time.time()
        })

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_ball_merge_basis(self, ball_client_ids: list, data_amounts: Dict[int, int] = None) -> Optional[torch.Tensor]:
        """
        计算并返回粒球的合并依据（客户端合并依据的加权平均）

        参数:
            ball_client_ids: 粒球包含的客户端ID列表
            data_amounts: 客户端数据量字典（用于加权）

        返回:
            粒球合并依据tensor，如果任何客户端缺失则返回None
        """
        if not ball_client_ids:
            return None

        # 收集所有客户端的合并依据
        client_basis = []
        weights = []

        for client_id in ball_client_ids:
            basis = self.load_client_merge_basis(client_id)
            if basis is None:
                print(f"警告：客户端{client_id}的合并依据不存在")
                return None

            client_basis.append(basis)

            # 确定权重
            if data_amounts and client_id in data_amounts:
                weights.append(data_amounts[client_id])
            else:
                weights.append(1.0)

        # 计算加权平均
        total_weight = sum(weights)
        weighted_sum = torch.zeros_like(client_basis[0])

        for basis, weight in zip(client_basis, weights):
            weighted_sum += basis * (weight / total_weight)

        return weighted_sum

    def save_block(self, block: Dict, round_num: int):
        """
        保存区块
        参数:
            block: 区块数据
            round_num: 轮次
        """
        filename = os.path.join(self.blocks_dir, f'block_{round_num:04d}.json')
        with open(filename, 'w') as f:
            # 处理不可序列化的对象
            serializable_block = self._make_serializable(block)
            json.dump(serializable_block, f, indent=2)

    def save_models(self, ball_models: Dict, round_num: int):
        """
        保存粒球模型
        参数:
            ball_models: 粒球模型字典
            round_num: 轮次
        """
        for ball_id, model_params in ball_models.items():
            filename = os.path.join(self.models_dir, f'model_r{round_num:04d}_b{ball_id}.pt')

            # 创建用于保存的干净状态字典
            state_dict_for_save = {}

            for key, value in model_params.items():
                if isinstance(value, torch.Tensor):
                    if value.is_cuda:
                        state_dict_for_save[key] = value.data.cpu().detach().clone()
                    else:
                        state_dict_for_save[key] = value.data.detach().clone()
                else:
                    state_dict_for_save[key] = value

            try:
                torch.save(state_dict_for_save, filename)
            except RuntimeError as e:
                print(f"警告：标准保存失败，使用备选方案保存模型 ball_{ball_id}")
                # 备选方案：转换为numpy数组
                numpy_dict = {}
                for key, value in state_dict_for_save.items():
                    if isinstance(value, torch.Tensor):
                        numpy_dict[key] = value.numpy()
                    else:
                        numpy_dict[key] = value

                backup_filename = filename.replace('.pt', '_backup.pkl')
                with open(backup_filename, 'wb') as f:
                    pickle.dump(numpy_dict, f)

    def save_checkpoint(self, system_state, round_num: int):
        """
        保存检查点
        参数:
            system_state: 系统状态
            round_num: 轮次
        """
        checkpoint = {
            'round': round_num,
            'balls': system_state.to_dict()['balls'],
            'unmatched_clients': system_state.unmatched_clients,
            'max_ball_id': system_state.max_ball_id
        }

        filename = os.path.join(self.checkpoints_dir, f'checkpoint_{round_num:04d}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

    def save_final_results(self, system_state, blockchain, cfl_stats):
        """
        保存最终结果
        参数:
            system_state: 系统状态
            blockchain: 区块链
            cfl_stats: 训练统计
        """
        # 保存系统状态
        state_file = os.path.join(self.base_dir, 'final_state.json')
        with open(state_file, 'w') as f:
            json.dump(system_state.to_dict(), f, indent=2)

        # 保存区块链
        chain_file = os.path.join(self.base_dir, 'blockchain.json')
        with open(chain_file, 'w') as f:
            serializable_chain = [self._make_serializable(block) for block in blockchain]
            json.dump(serializable_chain, f, indent=2)

        # 保存训练统计
        if cfl_stats:
            stats_file = os.path.join(self.base_dir, 'training_stats.json')

            export_data = {
                'experiment_info': {
                    'start_time': cfl_stats.start_time,
                    'export_time': time.time(),
                    'total_rounds': len(cfl_stats.rounds)
                },
                'rounds': cfl_stats.rounds,
                'accuracy_data': cfl_stats.acc_clients,
                'granular_balls': cfl_stats.granular_balls,
                'unmatched_clients': cfl_stats.unmatched_clients,
                'ball_events': cfl_stats.ball_events if hasattr(cfl_stats, 'ball_events') else [],
                'block_heights': cfl_stats.block_heights,
                'summary': cfl_stats.get_summary()
            }

            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

    def _make_serializable(self, obj):
        """
        将对象转换为可序列化格式
        参数:
            obj: 原始对象
        返回:
            可序列化对象
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                return obj.data.cpu().detach().numpy().tolist()
            else:
                return obj.data.detach().numpy().tolist()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj