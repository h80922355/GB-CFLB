"""
临时粒球状态管理
用于委员会独立操作的临时状态
"""
import torch
import copy
from typing import List, Dict, Any
from gbcfl.utils.device_utils import get_device
import hashlib

device = get_device()


class TempState:
    """
    临时粒球状态管理器
    用于委员会成员独立执行操作
    """

    def __init__(self):
        """初始化临时状态"""
        self.balls = []  # 临时粒球列表
        self.ball_models = {}  # 临时粒球模型参数
        self.unmatched_clients = []  # 临时未分配客户端列表
        self.max_ball_id = 0  # 临时最大粒球ID
        self.transactions = []  # 操作产生的交易列表

    def clone_from_system(self, system_state):
        """
        从系统状态深拷贝到临时状态
        参数:
            system_state: 系统状态对象
        """
        # 深拷贝粒球
        self.balls = [ball.clone() for ball in system_state.balls]

        # 深拷贝模型参数
        self.ball_models = {}
        for ball_id, model_params in system_state.ball_models.items():
            self.ball_models[ball_id] = {
                key: value.clone().to(device)
                for key, value in model_params.items()
            }

        # 拷贝其他状态
        self.unmatched_clients = system_state.unmatched_clients.copy()
        self.max_ball_id = system_state.max_ball_id
        self.transactions = []  # 重置交易列表

    def add_transaction(self, transaction):
        """
        添加交易
        参数:
            transaction: 交易对象
        """
        self.transactions.append(transaction)

    def get_ball_by_id(self, ball_id):
        """根据ID获取临时粒球"""
        for ball in self.balls:
            if ball.ball_id == ball_id:
                return ball
        return None

    def get_ball_model(self, ball_id):
        """
        获取粒球模型参数
        参数:
            ball_id: 粒球ID
        返回:
            模型参数字典或None
        """
        return self.ball_models.get(ball_id, None)

    def remove_ball(self, ball_id):
        """移除粒球"""
        self.balls = [ball for ball in self.balls if ball.ball_id != ball_id]
        if ball_id in self.ball_models:
            del self.ball_models[ball_id]

    def add_ball(self, ball, model_params=None):
        """
        添加新粒球
        参数:
            ball: 粒球对象
            model_params: 模型参数(可选)
        """
        self.balls.append(ball)
        if model_params is not None:
            self.ball_models[ball.ball_id] = {
                key: value.clone().to(device)
                for key, value in model_params.items()
            }

    def allocate_new_ball_id(self):
        """分配新的粒球ID"""
        self.max_ball_id += 1
        return self.max_ball_id

    def update_ball_model(self, ball_id, model_update):
        """
        更新粒球模型
        参数:
            ball_id: 粒球ID
            model_update: 模型更新参数
        """
        if ball_id in self.ball_models and model_update is not None:
            for key in self.ball_models[ball_id]:
                if key in model_update:
                    self.ball_models[ball_id][key].data += model_update[key].data

    def get_cluster_states(self):
        """
        获取聚类状态列表(用于Merkle计算)
        返回:
            聚类状态列表，按粒球ID排序
        """
        cluster_states = []

        # 按ID排序的粒球状态
        sorted_balls = sorted(self.balls, key=lambda x: x.ball_id)
        for ball in sorted_balls:
            # 计算模型哈希
            model_params = self.ball_models.get(ball.ball_id, {})
            if model_params:
                model_bytes = b''
                for key in sorted(model_params.keys()):
                    # 使用detach()分离梯度计算图
                    model_bytes += model_params[key].detach().cpu().numpy().tobytes()
                model_hash = hashlib.sha256(model_bytes).hexdigest()
            else:
                model_hash = "no_model"

            state = {
                'ball_id': ball.ball_id,
                'client_list': sorted(ball.client_ids),
                'center_hash': ball.calculate_center_hash(),
                'param_hash': model_hash
            }
            cluster_states.append(state)

        return cluster_states

    def get_sorted_transactions(self):
        """
        获取排序后的交易列表
        返回:
            按类型和ID排序的交易列表
        """
        # 按交易类型分组
        agg_trans = []
        split_trans = []
        merge_trans = []
        reassign_trans = []

        for tx in self.transactions:
            if tx['type'] == 'aggregation':
                agg_trans.append(tx)
            elif tx['type'] == 'split':
                split_trans.append(tx)
            elif tx['type'] == 'merge':
                merge_trans.append(tx)
            elif tx['type'] == 'reassign':
                reassign_trans.append(tx)

        # 各类型内部按粒球ID排序
        agg_trans.sort(key=lambda x: x.get('ball_id', 0))
        split_trans.sort(key=lambda x: x.get('source_ball_id', x.get('ball_id', 0)))
        merge_trans.sort(key=lambda x: x.get('merged_ball_id', 0))
        reassign_trans.sort(key=lambda x: min([a.get('ball_id', 0) for a in x.get('assignments', [{'ball_id': 0}])]))

        # 按顺序组合
        sorted_transactions = agg_trans + split_trans + merge_trans + reassign_trans
        return sorted_transactions