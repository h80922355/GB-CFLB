"""
共识验证器
负责区块和交易的验证
"""
import hashlib
from typing import Dict, List, Any, Optional
from blockchain.consensus.crypto_signer import CryptoSigner
from blockchain.consensus.merkle_tree import MerkleTree


class ConsensusValidator:
    """
    共识验证器类
    """

    def __init__(self, qc_manager=None, committee_manager=None):
        """
        初始化验证器
        参数:
            qc_manager: QC管理器实例
            committee_manager: 委员会管理器实例
        """
        self.signer = CryptoSigner()
        self.qc_manager = qc_manager
        self.committee_manager = committee_manager

    def set_qc_manager(self, qc_manager):
        """设置QC管理器"""
        self.qc_manager = qc_manager

    def set_committee_manager(self, committee_manager):
        """设置委员会管理器"""
        self.committee_manager = committee_manager

    def validate_block_proposal(self, proposal: Dict, parent_block: Dict = None,
                                chair_id: int = None, committee_info: Dict = None) -> bool:
        """
        验证区块提议
        参数:
            proposal: 区块提议（包含merkle根的哈希）
            parent_block: 父区块
            chair_id: 委员长ID
            committee_info: 委员会信息
        返回:
            验证结果
        """
        # 验证必要字段
        required_fields = ['proposer_id', 'view', 'parent_hash', 'timestamp',
                           'merkle_cluster_hash', 'merkle_trans_hash']

        for field in required_fields:
            if field not in proposal:
                return False

        # 验证提议者是否为委员长
        if chair_id is not None and proposal['proposer_id'] != chair_id:
            return False

        # 验证父区块哈希
        if parent_block:
            parent_hash = self.signer.compute_block_hash(parent_block.get('header'))
            if proposal['parent_hash'] != parent_hash:
                return False

        # 验证parent_qc
        if proposal.get('parent_qc'):
            if not self._verify_parent_qc(proposal['parent_qc']):
                return False

        # 验证时间戳
        if proposal['timestamp'] <= 0:
            return False

        return True

    def _verify_parent_qc(self, parent_qc: Dict) -> bool:
        """
        验证父QC
        参数:
            parent_qc: 父QC
        返回:
            验证结果
        """
        if not parent_qc:
            return True  # 创世区块没有parent_qc

        if not self.qc_manager:
            return False

        # 使用已设置的qc_manager验证QC
        return self.qc_manager.verify_qc(parent_qc)

    def validate_vote(self, vote: Dict, expected_block_hash: str) -> bool:
        """
        验证投票
        参数:
            vote: 投票对象
            expected_block_hash: 期望的区块哈希
        返回:
            验证结果
        """
        # 验证必要字段
        if 'member_id' not in vote or 'block_hash' not in vote or 'signature' not in vote:
            return False

        # 验证区块哈希一致性
        if vote['block_hash'] != expected_block_hash:
            return False

        # 获取成员公钥并验证签名
        if not self.committee_manager:
            return False

        member_id = vote['member_id']
        public_key_hex = self.committee_manager.get_client_public_key(member_id)

        if not public_key_hex:
            return False

        # 验证签名
        return self.signer.verify(vote['block_hash'], vote['signature'], public_key_hex)

    def validate_block_proposal_with_merkle(self, proposal: Dict, local_merkle_cluster: str,
                                           local_merkle_trans: str, committee_info: Dict) -> Dict:
        """
        验证区块提议（包含merkle验证）
        参数:
            proposal: 区块提议
            local_merkle_cluster: 本地计算的聚类merkle根
            local_merkle_trans: 本地计算的交易merkle根
            committee_info: 委员会信息
        返回:
            验证结果字典
        """
        # (1) 验证parent_qc是否为真实合法qc
        parent_qc_valid = True
        if proposal.get('parent_qc'):
            parent_qc_valid = self._verify_parent_qc(proposal['parent_qc'])

        # (2) 验证proposer_id是否为合法的委员长
        proposer_valid = proposal['proposer_id'] == committee_info['chair_id']

        # (3) 验证merkle哈希一致性
        merkle_valid = (
            MerkleTree.hash_data(local_merkle_cluster) == proposal['merkle_cluster_hash'] and
            MerkleTree.hash_data(local_merkle_trans) == proposal['merkle_trans_hash']
        )

        # 综合验证结果
        valid = parent_qc_valid and proposer_valid and merkle_valid

        return {
            'valid': valid,
            'parent_qc_valid': parent_qc_valid,
            'proposer_valid': proposer_valid,
            'merkle_valid': merkle_valid
        }

    def validate_transaction(self, transaction: Dict) -> bool:
        """
        验证交易
        参数:
            transaction: 交易对象
        返回:
            验证结果
        """
        # 验证交易类型
        valid_types = ['aggregation', 'split', 'merge', 'reassign']
        if transaction.get('type') not in valid_types:
            return False

        # 根据类型验证必要字段
        tx_type = transaction['type']

        if tx_type == 'aggregation':
            return 'ball_id' in transaction and 'valid_clients' in transaction
        elif tx_type == 'split':
            return ('source_ball_id' in transaction or 'ball_id' in transaction) and 'child_balls' in transaction
        elif tx_type == 'merge':
            return 'ball_ids' in transaction and 'merged_ball_id' in transaction
        elif tx_type == 'reassign':
            return 'assignments' in transaction

        return False

    def validate_block(self, block: Dict, committee_info: Dict, blockchain: List = None,
                      round_num: int = None) -> Dict:
        """
        验证完整区块
        改进：从header读取QC进行验证

        参数:
            block: 区块
            committee_info: 委员会信息
            blockchain: 区块链（用于验证父区块）
            round_num: 当前轮次
        返回:
            验证结果字典
        """
        result = {
            'valid': False,
            'proposer_valid': False,
            'parent_valid': False,
            'qc_threshold_valid': False,
            'qc_hash_valid': False,
            'qc_view_valid': False,
            'qc_signature_valid': False
        }

        # 验证区块结构
        if 'header' not in block or 'body' not in block:
            return result

        # 验证区块头必要字段
        header = block['header']
        required_header_fields = ['proposer_id', 'view', 'parent_hash', 'timestamp',
                                 'merkle_cluster', 'merkle_trans', 'qc']  # 改进：QC在header中

        for field in required_header_fields:
            if field not in header:
                return result

        # 验证区块体
        body = block['body']
        if 'cluster_states' not in body or 'transactions' not in body:
            return result

        # (1) 验证提议人是否为委员长
        result['proposer_valid'] = header['proposer_id'] == committee_info['chair_id']

        # (2) 父节点哈希和父节点QC是否对应
        result['parent_valid'] = True
        if blockchain and len(blockchain) > 1:  # 不是创世区块
            prev_block = blockchain[-2]  # 获取前一个区块
            # 区块哈希计算不包含QC
            expected_parent_hash = self.signer.compute_block_hash(prev_block['header'])
            if header['parent_hash'] != expected_parent_hash:
                result['parent_valid'] = False
            # 验证parent_qc是否是父区块的QC
            if header.get('parent_qc') != prev_block['header'].get('qc'):
                result['parent_valid'] = False

        # (3) 区块QC数量是否满足阈值
        qc = header['qc']  # 改进：从header获取QC
        qc_committee_size = len(qc.get('committee_members', committee_info['committee']))
        required_votes = int(qc_committee_size * 0.67)  # 使用默认阈值
        result['qc_threshold_valid'] = len(qc.get('votes', [])) >= required_votes

        # (4) 区块内本轮QC是否与本轮区块哈希、当前轮次相对应
        # 区块哈希永远不包含QC
        expected_block_hash = self.signer.compute_block_hash(header)
        result['qc_hash_valid'] = qc.get('block_hash') == expected_block_hash
        result['qc_view_valid'] = True
        if round_num is not None:
            result['qc_view_valid'] = qc.get('view') == round_num

        # (5) 验证QC的合法性（使用持久公钥验证）
        result['qc_signature_valid'] = False
        if self.qc_manager:
            result['qc_signature_valid'] = self.qc_manager.verify_qc(qc)

        # 综合判断
        result['valid'] = (
            result['proposer_valid'] and
            result['parent_valid'] and
            result['qc_threshold_valid'] and
            result['qc_hash_valid'] and
            result['qc_view_valid'] and
            result['qc_signature_valid']
        )

        return result

    def validate_all_transactions(self, transactions: List[Dict]) -> bool:
        """
        验证所有交易
        参数:
            transactions: 交易列表
        返回:
            是否所有交易都有效
        """
        for tx in transactions:
            if not self.validate_transaction(tx):
                return False
        return True