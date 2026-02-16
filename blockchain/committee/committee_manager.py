"""
委员会管理器
负责委员会的选举、轮换和管理
基于信誉分的加权选举机制
"""
import random
import numpy as np
import hashlib
import time
from typing import List, Dict, Any


class CommitteeManager:
    """
    委员会管理器类
    """

    def __init__(self, committee_size: int = 5, rotation_interval: int = 10,
                 candidate_pool_percentage: float = 0.7):
        """
        初始化委员会管理器
        参数:
            committee_size: 委员会大小
            rotation_interval: 轮换间隔
            candidate_pool_percentage: 候选池占总客户端的比例
        """
        self.committee_size = committee_size
        self.rotation_interval = rotation_interval
        self.candidate_pool_percentage = candidate_pool_percentage
        self.current_committee = []
        self.chair_id = None
        self.round_counter = 0

        # 所有客户端的持久密钥对（用于签名和验证）
        self.client_keys = {}  # {client_id: {'private': hex, 'public': hex}}

        # 委员会加密密钥对（用于参数加密/解密）
        self.committee_encryption_keys = {'private': None, 'public': None}

        # 信誉管理器引用（将在初始化时设置）
        self.reputation_manager = None

    def set_reputation_manager(self, reputation_manager):
        """
        设置信誉管理器引用
        参数:
            reputation_manager: 信誉管理器实例
        """
        self.reputation_manager = reputation_manager

    def initialize_all_client_keys(self, client_ids: List[int]):
        """
        为所有客户端初始化持久密钥对（只在系统启动时调用一次）
        参数:
            client_ids: 所有客户端ID列表
        """
        from blockchain.consensus.crypto_signer import CryptoSigner
        signer = CryptoSigner()

        for client_id in client_ids:
            if client_id not in self.client_keys:
                private_key_hex, public_key_hex = signer.generate_keypair()
                self.client_keys[client_id] = {
                    'private': private_key_hex,
                    'public': public_key_hex
                }

    def _generate_committee_encryption_keys(self):
        """
        生成委员会加密密钥对（用于参数加密/解密）
        在委员会选举、轮换或重组时调用
        """
        # 使用简单的哈希方法生成密钥对（模拟）
        seed = f"{self.current_committee}:{time.time()}"
        private_key = hashlib.sha256(seed.encode()).hexdigest()
        public_key = hashlib.sha256(private_key.encode()).hexdigest()

        self.committee_encryption_keys = {
            'private': private_key,
            'public': public_key
        }

    def initialize_committee(self, client_ids: List[int]) -> Dict:
        """
        初始化委员会（基于信誉分的加权选举）
        参数:
            client_ids: 所有客户端ID列表
        返回:
            委员会信息
        """
        # 确保所有客户端都有持久密钥对
        self.initialize_all_client_keys(client_ids)

        # 执行基于信誉分的选举
        self.current_committee = self._elect_committee(client_ids)

        # 选择信誉分最高的成员作为委员长
        self.chair_id = self._elect_chair()

        # 生成委员会加密密钥对
        self._generate_committee_encryption_keys()

        return self.get_committee_info()

    def should_rotate(self, current_round: int) -> bool:
        """
        判断是否需要轮换
        参数:
            current_round: 当前轮次
        返回:
            是否需要轮换
        """
        return current_round > 0 and current_round % self.rotation_interval == 0

    def rotate_committee(self, client_ids: List[int]) -> Dict:
        """
        轮换委员会（基于信誉分的加权选举）
        参数:
            client_ids: 所有客户端ID列表
        返回:
            新委员会信息
        """
        # 基于信誉分重新选举委员会
        self.current_committee = self._elect_committee(client_ids)

        # 重新选举委员长（信誉分最高者）
        self.chair_id = self._elect_chair()

        # 更新委员会加密密钥对
        self._generate_committee_encryption_keys()

        return self.get_committee_info()

    def handle_impeachment(self, client_ids: List[int]) -> Dict:
        """
        处理弹劾，重组委员会
        参数:
            client_ids: 所有客户端ID列表
        返回:
            新委员会信息
        """
        # 移除当前委员长
        if self.chair_id in self.current_committee:
            self.current_committee.remove(self.chair_id)

        # 基于信誉分补充新成员
        available_clients = [cid for cid in client_ids
                             if cid not in self.current_committee]

        if available_clients and self.reputation_manager:
            # 创建候选池
            candidate_pool_size = max(1, int(len(available_clients) * self.candidate_pool_percentage))
            candidates = self.reputation_manager.get_top_clients(
                candidate_pool_size,
                exclude_ids=self.current_committee
            )

            if candidates:
                # 基于信誉分加权选择
                new_member = self._weighted_random_selection(candidates, 1)[0]
                self.current_committee.append(new_member)

        # 重新选举委员长（信誉分最高者）
        self.chair_id = self._elect_chair()

        # 更新委员会加密密钥对
        self._generate_committee_encryption_keys()

        return self.get_committee_info()

    def _elect_committee(self, client_ids: List[int]) -> List[int]:
        """
        基于信誉分的加权选举委员会
        参数:
            client_ids: 所有客户端ID列表
        返回:
            选出的委员会成员ID列表
        """
        if not self.reputation_manager:
            # 如果没有信誉管理器，使用随机选择（初始化时）
            return random.sample(client_ids, min(self.committee_size, len(client_ids)))

        # 步骤1：根据信誉分排名创建候选池
        candidate_pool_size = max(self.committee_size,
                                  int(len(client_ids) * self.candidate_pool_percentage))
        candidates = self.reputation_manager.get_top_clients(candidate_pool_size)

        # 确保候选池中只包含有效的客户端ID
        candidates = [cid for cid in candidates if cid in client_ids]

        if len(candidates) < self.committee_size:
            # 如果候选池不足，补充其他客户端
            remaining = [cid for cid in client_ids if cid not in candidates]
            candidates.extend(remaining[:self.committee_size - len(candidates)])

        # 步骤2：基于信誉分的加权随机选择
        selected = self._weighted_random_selection(candidates, self.committee_size)

        return selected

    def _weighted_random_selection(self, candidates: List[int], n: int) -> List[int]:
        """
        基于信誉分的加权随机选择
        参数:
            candidates: 候选者ID列表
            n: 要选择的数量
        返回:
            选中的ID列表
        """
        if not self.reputation_manager or len(candidates) <= n:
            return candidates[:n]

        # 获取候选者的信誉分
        scores = [self.reputation_manager.get_score(cid) for cid in candidates]

        # 计算选择概率（与信誉分成正比）
        total_score = sum(scores)
        if total_score == 0:
            # 如果所有分数都是0，使用均匀分布
            probabilities = [1.0 / len(candidates)] * len(candidates)
        else:
            probabilities = [score / total_score for score in scores]

        # 加权随机选择（不重复）
        selected = np.random.choice(
            candidates,
            size=min(n, len(candidates)),
            replace=False,
            p=probabilities
        ).tolist()

        return selected

    def _elect_chair(self) -> int:
        """
        选举委员长（信誉分最高的委员会成员）
        返回:
            委员长ID
        """
        if not self.current_committee:
            return None

        if not self.reputation_manager:
            # 如果没有信誉管理器，选择第一个成员
            return self.current_committee[0]

        # 选择信誉分最高的成员作为委员长
        max_score = -1
        chair = self.current_committee[0]

        for member_id in self.current_committee:
            score = self.reputation_manager.get_score(member_id)
            if score > max_score:
                max_score = score
                chair = member_id

        return chair

    def get_committee_info(self) -> Dict:
        """
        获取委员会信息
        返回:
            委员会信息字典
        """
        # 获取当前委员会成员的签名公钥信息
        member_public_keys = {}
        for member_id in self.current_committee:
            if member_id in self.client_keys:
                member_public_keys[member_id] = self.client_keys[member_id]['public']

        return {
            'committee': self.current_committee,
            'chair_id': self.chair_id,
            'size': len(self.current_committee),
            'member_public_keys': member_public_keys,
            'encryption_public_key': self.committee_encryption_keys['public']
        }

    def get_member_private_key(self, member_id: int) -> str:
        """
        获取成员签名私钥
        参数:
            member_id: 成员ID
        返回:
            私钥十六进制字符串
        """
        if member_id in self.client_keys:
            return self.client_keys[member_id]['private']
        return None

    def get_client_public_key(self, client_id: int) -> str:
        """
        获取任意客户端的签名公钥（用于验证历史QC）
        参数:
            client_id: 客户端ID
        返回:
            公钥十六进制字符串
        """
        if client_id in self.client_keys:
            return self.client_keys[client_id]['public']
        return None

    def get_committee_encryption_private_key(self) -> str:
        """
        获取委员会加密私钥（用于解密参数）
        返回:
            加密私钥
        """
        return self.committee_encryption_keys['private']

    def get_committee_encryption_public_key(self) -> str:
        """
        获取委员会加密公钥（用于加密参数）
        返回:
            加密公钥
        """
        return self.committee_encryption_keys['public']

    def is_chair(self, member_id: int) -> bool:
        """
        判断是否为委员长
        参数:
            member_id: 成员ID
        返回:
            是否为委员长
        """
        return member_id == self.chair_id

    def is_member(self, member_id: int) -> bool:
        """
        判断是否为委员会成员
        参数:
            member_id: 成员ID
        返回:
            是否为成员
        """
        return member_id in self.current_committee