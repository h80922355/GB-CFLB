"""
粒球(Granular Ball)类定义
实现粒球的基本属性和操作方法
改进：支持有效客户端标记和动态更新
"""
import torch
import copy
import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from gbcfl.utils.device_utils import get_device
from gbcfl.utils.operations import flatten

device = get_device()


class BallIDManager:
    """粒球ID管理器，确保ID的唯一性和连续性"""
    def __init__(self):
        self.next_id = 0

    def get_next_id(self):
        """获取下一个可用的ID"""
        current_id = self.next_id
        self.next_id += 1
        return current_id

    def set_next_id(self, next_id):
        """设置下一个ID（用于恢复）"""
        self.next_id = next_id


class GranularBall:
    """
    粒球类：表示联邦学习中的一个客户端簇
    改进：支持区分有效和无效客户端
    """
    # 类级别的ID管理器
    id_manager = BallIDManager()

    def __init__(self, client_ids_or_empty_list=None, ball_id=None):
        """
        初始化粒球
        参数:
            client_ids_or_empty_list: 客户端ID列表或空列表
            ball_id: 指定的粒球ID（如果为None则自动分配）
        """
        # 基本属性
        if ball_id is not None:
            self.ball_id = ball_id
            # 更新ID管理器
            if ball_id >= GranularBall.id_manager.next_id:
                GranularBall.id_manager.set_next_id(ball_id + 1)
        else:
            self.ball_id = GranularBall.id_manager.get_next_id()

        # 客户端管理 - 直接使用客户端ID列表
        self.clients = []  # 保持为空，仅用于兼容

        # 处理输入参数
        if client_ids_or_empty_list is None:
            self.client_ids = []
        elif isinstance(client_ids_or_empty_list, list):
            # 检查是否是ID列表
            if len(client_ids_or_empty_list) > 0 and isinstance(client_ids_or_empty_list[0], int):
                self.client_ids = client_ids_or_empty_list.copy()
            else:
                # 空列表或客户端对象列表（后者不应出现）
                self.client_ids = []
        else:
            self.client_ids = []

        # 新增：有效客户端集合（用于跟踪验证通过的客户端）
        self.valid_clients: Set[int] = set()

        # 粒球中心和质量指标
        self.center = None  # 粒球中心（参数空间）
        self.prev_center = None  # 上一轮的中心
        self.purity = 0.0  # 粒球纯度
        self.convergence = float('inf')  # 收敛指标
        self.distance_history = []  # 中心移动距离历史

        # 操作控制
        self.cooling_period = 0  # 冷却期计数器
        self.aggregate_update = None  # 聚合的更新参数

        # 历史记录
        self.purity_history = []  # 纯度历史
        self.client_similarities = {}  # 客户端相似度记录

        # 记录上次操作轮次
        self.last_operation_round = 0

    def clone(self):
        """
        深拷贝粒球对象（用于创建临时粒球）
        """
        cloned = GranularBall([], ball_id=self.ball_id)
        cloned.client_ids = self.client_ids.copy()
        cloned.valid_clients = self.valid_clients.copy()  # 新增：拷贝有效客户端集合
        cloned.center = self.center.clone() if self.center is not None else None
        cloned.prev_center = self.prev_center.clone() if self.prev_center is not None else None
        cloned.purity = self.purity
        cloned.convergence = self.convergence
        cloned.distance_history = self.distance_history.copy()
        cloned.cooling_period = self.cooling_period
        cloned.aggregate_update = copy.deepcopy(self.aggregate_update)
        cloned.purity_history = self.purity_history.copy()
        cloned.client_similarities = self.client_similarities.copy()
        cloned.last_operation_round = self.last_operation_round
        cloned.clients = []  # 保持为空
        return cloned

    def set_valid_clients(self, valid_client_ids: List[int]):
        """
        设置有效客户端ID集合

        参数:
            valid_client_ids: 验证通过的客户端ID列表
        """
        self.valid_clients = set(valid_client_ids)

    def get_valid_client_ids(self) -> List[int]:
        """
        获取有效客户端ID列表

        返回:
            有效客户端ID列表（在client_ids中且验证通过的）
        """
        if not self.valid_clients:
            # 如果没有设置有效客户端，返回所有客户端（兼容旧逻辑）
            return self.client_ids.copy()

        # 返回既在client_ids中又在valid_clients中的客户端
        return [cid for cid in self.client_ids if cid in self.valid_clients]

    def get_invalid_client_ids(self) -> List[int]:
        """
        获取无效客户端ID列表

        返回:
            无效客户端ID列表（在client_ids中但验证未通过的）
        """
        if not self.valid_clients:
            # 如果没有设置有效客户端，认为所有客户端都有效
            return []

        # 返回在client_ids中但不在valid_clients中的客户端
        return [cid for cid in self.client_ids if cid not in self.valid_clients]

    def get_valid_client_count(self) -> int:
        """
        获取有效客户端数量

        返回:
            有效客户端数量
        """
        return len(self.get_valid_client_ids())

    def calculate_center_hash(self):
        """计算粒球中心的哈希值"""
        if self.center is None:
            return "no_center_available"
        center_bytes = self.center.detach().cpu().numpy().tobytes()
        return hashlib.sha256(center_bytes).hexdigest()

    def update_center(self, client_updates_dict, valid_only=True):
        """
        更新粒球中心
        改进：支持只使用有效客户端更新

        参数:
            client_updates_dict: 客户端更新字典 {client_id: update_tensor}
            valid_only: 是否只使用有效客户端（默认True）
        """
        old_center = self.center
        self.prev_center = self.center

        # 决定使用哪些客户端
        if valid_only and self.valid_clients:
            clients_to_use = self.get_valid_client_ids()
        else:
            clients_to_use = self.client_ids

        if not clients_to_use:
            self.center = None
            return

        # 计算加权平均更新
        updates = []
        weights = []
        for client_id in clients_to_use:
            if client_id in client_updates_dict:
                updates.append(client_updates_dict[client_id])
                weights.append(1.0)  # 可以根据数据量加权

        if updates:
            total_weight = sum(weights)
            weighted_sum = torch.zeros_like(updates[0])
            for update, weight in zip(updates, weights):
                weighted_sum += update * (weight / total_weight)
            self.center = weighted_sum.to(device)

            # 计算中心移动距离
            if old_center is not None:
                movement = torch.norm(self.center - old_center).item()
                self.distance_history.append(movement)
                # 保持历史窗口大小
                if len(self.distance_history) > 10:
                    self.distance_history.pop(0)

    def update_purity(self, client_similarities=None, valid_only=True):
        """
        更新粒球纯度
        改进：支持只使用有效客户端计算纯度

        参数:
            client_similarities: 客户端相似度字典 {client_id: similarity}
            valid_only: 是否只使用有效客户端（默认True）
        """
        if client_similarities:
            self.client_similarities = client_similarities

            # 决定使用哪些客户端
            if valid_only and self.valid_clients:
                clients_to_use = self.get_valid_client_ids()
                filtered_similarities = {cid: sim for cid, sim in client_similarities.items()
                                       if cid in clients_to_use}
            else:
                filtered_similarities = client_similarities

            if len(filtered_similarities) > 0:
                self.purity = sum(filtered_similarities.values()) / len(filtered_similarities)
            else:
                self.purity = 0.0
        else:
            # 如果没有提供相似度，使用历史数据
            if self.client_similarities:
                if valid_only and self.valid_clients:
                    clients_to_use = self.get_valid_client_ids()
                    filtered_similarities = {cid: sim for cid, sim in self.client_similarities.items()
                                           if cid in clients_to_use}
                else:
                    filtered_similarities = self.client_similarities

                if len(filtered_similarities) > 0:
                    self.purity = sum(filtered_similarities.values()) / len(filtered_similarities)
                else:
                    self.purity = 0.0
            else:
                self.purity = 0.0

        # 记录纯度历史
        self.purity_history.append(self.purity)
        # 保持历史窗口大小
        if len(self.purity_history) > 10:
            self.purity_history.pop(0)

    def get_center_movement(self):
        """获取中心移动距离"""
        if self.prev_center is None or self.center is None:
            return float('inf')
        return torch.norm(self.center - self.prev_center).item()

    def calculate_convergence_indicator(self, window_size=3, min_history=3):
        """
        计算收敛指标
        参数:
            window_size: 历史窗口大小
            min_history: 最小历史记录数
        返回:
            (convergence_indicator, is_calculable)
        """
        history_length = len(self.distance_history)

        if history_length < min_history:
            return float('inf'), False

        # 获取当前移动距离
        current_movement = self.get_center_movement()
        if current_movement == float('inf'):
            return float('inf'), False

        # 计算历史平均移动距离
        actual_window = min(window_size, history_length - 1)
        if actual_window < 1:
            return 1.0 if current_movement < 1e-8 else float('inf'), True

        # 使用最近的历史记录（不包括当前）
        previous_movements = self.distance_history[-(actual_window + 1):-1]
        avg_previous_movement = sum(previous_movements) / len(previous_movements)

        if avg_previous_movement < 1e-8:
            return 1.0 if current_movement < 1e-8 else float('inf'), True

        convergence_indicator = current_movement / avg_previous_movement
        self.convergence = convergence_indicator
        return convergence_indicator, True

    def get_average_purity(self, window_size=1, min_history=1):
        """计算窗口期平均纯度"""
        history_length = len(self.purity_history)

        if history_length < min_history:
            return self.purity, False

        actual_window = min(window_size, history_length)
        recent_purities = self.purity_history[-actual_window:]
        average_purity = sum(recent_purities) / len(recent_purities)

        return average_purity, True

    def to_dict(self):
        """转换为字典表示"""
        return {
            'ball_id': self.ball_id,
            'client_ids': self.client_ids,
            'valid_client_ids': list(self.valid_clients),  # 新增：包含有效客户端信息
            'valid_client_count': self.get_valid_client_count(),
            'purity': self.purity,
            'convergence': self.convergence,
            'center_hash': self.calculate_center_hash(),
            'cooling_period': self.cooling_period,
            'last_operation_round': self.last_operation_round
        }

    def calculate_client_similarity(self, client):
        """
        计算客户端与粒球中心的相似度（基于更新参数）
        注意：这个方法保留是为了兼容，但不应该在委员会操作中使用

        参数:
            client: 客户端对象

        返回:
            相似度值 [-1, 1]
        """
        if self.center is None:
            return 0.0

        # 获取客户端更新参数
        if hasattr(client, 'dW') and client.dW:
            client_update = flatten(client.dW)
        else:
            return 0.0

        # 计算余弦相似度
        center_norm = torch.norm(self.center)
        update_norm = torch.norm(client_update)

        if center_norm < 1e-8 or update_norm < 1e-8:
            return 0.0

        cos_sim = torch.dot(self.center, client_update) / (center_norm * update_norm)
        return cos_sim.item()

    def check_split_feasibility(self, min_size):
        """
        检查粒球是否可以分裂
        改进：基于有效客户端数量判断

        参数:
            min_size: 最小粒球大小

        返回:
            是否可以分裂
        """
        valid_count = self.get_valid_client_count()
        return valid_count >= 2 * min_size