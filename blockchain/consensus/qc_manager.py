"""
QC (Quorum Certificate) 管理器
负责QC的生成和验证
"""
import hashlib
import json
from typing import List, Dict, Any
from blockchain.consensus.crypto_signer import CryptoSigner


class QCManager:
    """
    QC管理器类
    """

    def __init__(self, threshold: float = 0.67):
        """
        初始化QC管理器
        参数:
            threshold: 共识阈值(默认2/3)
        """
        self.threshold = threshold
        self.signer = CryptoSigner()
        self.committee_manager = None  # 将在初始化时设置

    def set_committee_manager(self, committee_manager):
        """
        设置委员会管理器引用
        参数:
            committee_manager: 委员会管理器实例
        """
        self.committee_manager = committee_manager

    def generate_qc(self, block_hash: str, view: int, votes: List[Dict],
                   committee_members: List[int] = None) -> Dict:
        """
        生成QC
        参数:
            block_hash: 区块哈希
            view: 轮次编号
            votes: 投票列表
            committee_members: 委员会成员列表
        返回:
            QC字典
        """
        qc = {
            'block_hash': block_hash,
            'view': view,
            'votes': votes,
            'committee_members': committee_members if committee_members else [],
            'timestamp': self._get_timestamp()
        }
        return qc

    def verify_qc(self, qc: Dict, committee_size: int = None,
                 member_public_keys: Dict = None) -> bool:
        """
        验证QC有效性
        参数:
            qc: QC对象
            committee_size: 委员会大小（可选，从QC中获取）
            member_public_keys: 公钥字典（已弃用，从committee_manager获取）
        返回:
            验证结果
        """
        if not qc or 'votes' not in qc:
            return False

        # 获取QC中记录的委员会成员
        qc_committee_members = qc.get('committee_members', [])
        if not qc_committee_members and committee_size:
            # 兼容旧版本QC
            qc_committee_size = committee_size
        else:
            qc_committee_size = len(qc_committee_members)

        if qc_committee_size == 0:
            return False

        # 检查是否有委员会管理器
        if not self.committee_manager:
            return False

        # 检查投票数量是否达到阈值
        valid_votes = 0
        block_hash = qc.get('block_hash', '')

        for vote in qc['votes']:
            if self._verify_vote_with_persistent_key(vote, block_hash):
                valid_votes += 1

        required_votes = int(qc_committee_size * self.threshold)
        return valid_votes >= required_votes

    def _verify_vote_with_persistent_key(self, vote: Dict, expected_hash: str) -> bool:
        """
        使用持久化的客户端公钥验证投票
        参数:
            vote: 投票对象
            expected_hash: 期望的区块哈希
        返回:
            验证结果
        """
        member_id = vote.get('member_id')
        block_hash = vote.get('block_hash')
        signature = vote.get('signature')

        # 验证区块哈希一致性
        if block_hash != expected_hash:
            return False

        # 从委员会管理器获取该成员的持久公钥
        if not self.committee_manager:
            return False

        public_key_hex = self.committee_manager.get_client_public_key(member_id)
        if not public_key_hex:
            return False

        # 使用持久公钥验证签名
        return self.signer.verify(block_hash, signature, public_key_hex)

    def create_vote(self, member_id: int, block_hash: str, private_key_hex: str) -> Dict:
        """
        创建投票
        参数:
            member_id: 成员ID
            block_hash: 区块哈希
            private_key_hex: 私钥十六进制字符串
        返回:
            投票对象
        """
        # 使用真实签名
        signature = self.signer.sign(block_hash, private_key_hex)

        vote = {
            'member_id': member_id,
            'block_hash': block_hash,
            'signature': signature,
            'timestamp': self._get_timestamp()
        }
        return vote

    def _get_timestamp(self) -> float:
        """获取当前时间戳"""
        import time
        return time.time()