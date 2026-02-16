"""
交易定义
"""
from typing import Dict, List, Any
import hashlib
import json
import time


class Transaction:
    """
    交易类
    """

    def __init__(self, tx_type: str, data: Dict):
        """
        初始化交易
        参数:
            tx_type: 交易类型
            data: 交易数据
        """
        self.type = tx_type
        self.data = data
        self.timestamp = time.time()
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        """
        计算交易哈希
        返回:
            哈希值
        """
        tx_str = json.dumps({
            'type': self.type,
            'data': self.data,
            'timestamp': self.timestamp
        }, sort_keys=True)
        return hashlib.sha256(tx_str.encode()).hexdigest()

    def to_dict(self) -> Dict:
        """
        转换为字典
        返回:
            交易字典
        """
        return {
            'type': self.type,
            'data': self.data,
            'timestamp': self.timestamp,
            'hash': self.hash
        }

    @staticmethod
    def create_aggregation_tx(ball_id: int, valid_clients: List[int]) -> Dict:
        """
        创建聚合交易
        参数:
            ball_id: 粒球ID
            valid_clients: 有效客户端列表
        返回:
            交易字典
        """
        return {
            'type': 'aggregation',
            'ball_id': ball_id,
            'valid_clients': valid_clients
        }

    @staticmethod
    def create_split_tx(ball_id: int, new_balls: List[Dict]) -> Dict:
        """
        创建分裂交易
        参数:
            ball_id: 原粒球ID
            new_balls: 新粒球列表
        返回:
            交易字典
        """
        return {
            'type': 'split',
            'ball_id': ball_id,
            'new_balls': new_balls
        }