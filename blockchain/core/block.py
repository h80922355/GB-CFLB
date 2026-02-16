"""
区块定义
"""
from typing import Dict, List, Any
import hashlib
import json
import time


class Block:
    """
    区块类
    """

    def __init__(self, header: Dict, body: Dict):
        """
        初始化区块
        参数:
            header: 区块头
            body: 区块体
        """
        self.header = header
        self.body = body
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        """
        计算区块哈希
        返回:
            哈希值
        """
        header_str = json.dumps(self.header, sort_keys=True)
        return hashlib.sha256(header_str.encode()).hexdigest()

    def to_dict(self) -> Dict:
        """
        转换为字典
        返回:
            区块字典
        """
        return {
            'header': self.header,
            'body': self.body,
            'hash': self.hash
        }