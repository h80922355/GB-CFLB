"""
Merkle树实现
用于交易和状态的哈希验证
"""
import hashlib
import json
from typing import List, Any, Dict, Optional


class MerkleTree:
    """
    Merkle树实现类
    """

    def __init__(self):
        """初始化Merkle树"""
        self.leaves = []
        self.tree = []  # 存储完整的树结构
        self.root = None

    def compute_root(self, data_list: List[Any]) -> str:
        """
        计算Merkle树根
        参数:
            data_list: 数据列表
        返回:
            Merkle根哈希值
        """
        if not data_list:
            return hashlib.sha256(b'empty').hexdigest()

        # 将数据转换为哈希值
        self.leaves = []
        for data in data_list:
            if isinstance(data, dict):
                data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = str(data).encode('utf-8')

            leaf_hash = hashlib.sha256(data_bytes).hexdigest()
            self.leaves.append(leaf_hash)

        # 构建完整的Merkle树
        self.tree = [self.leaves.copy()]
        current_level = self.leaves.copy()

        while len(current_level) > 1:
            next_level = []

            # 两两配对计算哈希
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    combined = current_level[i] + current_level[i + 1]
                else:
                    # 奇数个节点，最后一个节点与自己配对
                    combined = current_level[i] + current_level[i]

                parent_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
                next_level.append(parent_hash)

            self.tree.append(next_level)
            current_level = next_level

        self.root = current_level[0] if current_level else hashlib.sha256(b'empty').hexdigest()
        return self.root

    def get_proof(self, index: int) -> List[Dict[str, str]]:
        """
        获取指定索引数据的Merkle证明
        参数:
            index: 数据索引
        返回:
            证明路径
        """
        if index >= len(self.leaves) or not self.tree:
            return []

        proof = []
        current_index = index

        # 遍历树的每一层
        for level in range(len(self.tree) - 1):
            current_level = self.tree[level]

            # 找到兄弟节点
            if current_index % 2 == 0:
                # 当前节点是左节点
                if current_index + 1 < len(current_level):
                    sibling = current_level[current_index + 1]
                    position = 'right'
                else:
                    # 没有右兄弟，与自己配对
                    sibling = current_level[current_index]
                    position = 'right'
            else:
                # 当前节点是右节点
                sibling = current_level[current_index - 1]
                position = 'left'

            proof.append({
                'hash': sibling,
                'position': position
            })

            # 更新索引到父节点
            current_index = current_index // 2

        return proof

    def verify_inclusion(self, data: Any, proof: List[Dict[str, str]]) -> bool:
        """
        验证数据是否包含在Merkle树中
        参数:
            data: 要验证的数据
            proof: Merkle证明路径
        返回:
            验证结果
        """
        if not self.root:
            return False

        # 计算数据的哈希
        if isinstance(data, dict):
            data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = str(data).encode('utf-8')

        current_hash = hashlib.sha256(data_bytes).hexdigest()

        # 沿着证明路径计算根哈希
        for proof_element in proof:
            sibling_hash = proof_element['hash']
            position = proof_element['position']

            if position == 'left':
                combined = sibling_hash + current_hash
            else:  # position == 'right'
                combined = current_hash + sibling_hash

            current_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()

        # 验证计算出的根哈希是否与实际根哈希匹配
        return current_hash == self.root

    @staticmethod
    def hash_data(data: Any) -> str:
        """
        计算数据的哈希值
        参数:
            data: 要哈希的数据
        返回:
            哈希值
        """
        if isinstance(data, dict):
            data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = str(data).encode('utf-8')

        return hashlib.sha256(data_bytes).hexdigest()