"""
区块链实现
"""
from typing import List, Dict, Any
from blockchain.core.block import Block


class Blockchain:
    """
    区块链类
    """

    def __init__(self):
        """初始化区块链"""
        self.chain = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        """创建创世区块"""
        genesis_header = {
            'proposer_id': 0,
            'view': 0,
            'parent_hash': 'genesis',
            'parent_qc': None,
            'timestamp': 0,
            'merkle_cluster_hash': 'genesis',
            'merkle_trans_hash': 'genesis'
        }

        genesis_body = {
            'cluster_states': [],
            'transactions': [],
            'qc': None
        }

        genesis = Block(genesis_header, genesis_body)
        self.chain.append(genesis)

    def add_block(self, block: Block) -> bool:
        """
        添加区块
        参数:
            block: 区块对象
        返回:
            是否成功
        """
        if self._validate_block(block):
            self.chain.append(block)
            return True
        return False

    def _validate_block(self, block: Block) -> bool:
        """
        验证区块
        参数:
            block: 区块对象
        返回:
            验证结果
        """
        if len(self.chain) == 0:
            return False

        # 验证父区块哈希
        last_block = self.chain[-1]
        if block.header.get('parent_hash') != last_block.hash:
            return False

        return True

    def get_latest_block(self) -> Block:
        """
        获取最新区块
        返回:
            最新区块
        """
        return self.chain[-1] if self.chain else None

    def get_chain_length(self) -> int:
        """
        获取链长度
        返回:
            链长度
        """
        return len(self.chain)