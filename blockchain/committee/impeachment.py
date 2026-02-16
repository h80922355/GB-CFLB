"""
弹劾机制管理器
处理委员长弹劾相关逻辑
"""
from typing import List, Dict


class ImpeachmentManager:
    """
    弹劾管理器类
    """

    def __init__(self, threshold: float = 0.67):
        """
        初始化弹劾管理器
        参数:
            threshold: 弹劾阈值
        """
        self.threshold = threshold

    def check_impeachment(self, impeachment_votes: List[Dict], committee_size: int) -> bool:
        """
        检查是否触发弹劾
        参数:
            impeachment_votes: 弹劾投票列表
            committee_size: 委员会大小
        返回:
            是否触发弹劾
        """
        if not impeachment_votes:
            return False

        # 按类型统计弹劾投票（三种类型独立计数）
        vote_types = {
            'timeout': 0,           # 区块生成超时
            'proposal_invalid': 0,  # 提议验证不通过
            'block_invalid': 0      # 区块验证不通过
        }

        for vote in impeachment_votes:
            vote_type = vote.get('type', 'unknown')
            if vote_type in vote_types:
                vote_types[vote_type] += 1

        # 检查每种类型是否达到阈值
        required_votes = int(committee_size * self.threshold)

        for vote_type, count in vote_types.items():
            if count >= required_votes:
                print(f"弹劾触发: {vote_type}类型投票数({count})达到阈值({required_votes})")
                return True

        return False