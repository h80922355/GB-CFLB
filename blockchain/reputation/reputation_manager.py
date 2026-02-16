"""
信誉管理器
管理节点信誉分的计算和更新
"""
import os
import json
from typing import List, Dict, Any
import numpy as np


class ReputationManager:
    """
    信誉管理器类
    """

    def __init__(self, output_dir: str = 'outputs'):
        """
        初始化信誉管理器
        参数:
            output_dir: 输出目录
        """
        self.reputation_scores = {}  # {client_id: score}
        self.history = []  # 历史记录

        # 设置信誉分存储路径
        self.reputation_dir = os.path.join(output_dir, 'reputation')
        os.makedirs(self.reputation_dir, exist_ok=True)

    def initialize_scores(self, client_ids: List[int], initial_score: float = 10.0):
        """
        初始化信誉分
        参数:
            client_ids: 客户端ID列表
            initial_score: 初始分数
        """
        for client_id in client_ids:
            self.reputation_scores[client_id] = initial_score

        # 保存初始信誉分
        self.save_reputation_scores(0)

    def get_score(self, client_id: int) -> float:
        """
        获取客户端信誉分
        参数:
            client_id: 客户端ID
        返回:
            信誉分值
        """
        return self.reputation_scores.get(client_id, 10.0)

    def update_scores(self, update_messages, committee_info, valid_votes, impeachment_votes,
                      consensus_reached, reward_config, valid_updates=None):
        """
        更新信誉分
        参数:
            update_messages: 更新消息列表，包含客户端ID和数据量
            committee_info: 委员会信息
            valid_votes: 有效投票列表（已验证的投票）
            impeachment_votes: 弹劾投票
            consensus_reached: 是否达成共识
            reward_config: 奖励配置
            valid_updates: 有效更新的客户端ID列表
        """
        # 从配置读取参数
        base_reward = reward_config.get('base_reward', 2.0)
        update_factor = reward_config.get('update_reward_factor', 1)
        validation_factor = reward_config.get('validation_reward_factor', 0.15)
        governance_factor = reward_config.get('governance_reward_factor', 0.2)
        max_reward_coefficient = reward_config.get('max_reward_coefficient', 4.0)
        validation_penalty_ratio = reward_config.get('validation_penalty_ratio', 0.1)
        block_penalty_ratio = reward_config.get('block_failure_penalty_ratio', 0.25)

        # 从更新消息中提取客户端ID和数据量映射
        client_data_amounts = {}
        client_ball_ids = {}
        for msg in update_messages:
            client_id = msg.get('client_id')
            data_amount = msg.get('data_amount', 1)
            ball_id = msg.get('ball_id', 0)
            client_data_amounts[client_id] = data_amount
            client_ball_ids[client_id] = ball_id

        # 计算每个粒球的平均数据量
        ball_data_amounts = self._calculate_ball_average_data_amounts(
            client_data_amounts, client_ball_ids
        )

        # 获取委员会信息
        committee = committee_info.get('committee', [])
        chair_id = committee_info.get('chair_id')

        # 处理训练奖励
        if valid_updates:
            for client_id in valid_updates:
                if client_id not in self.reputation_scores:
                    self.reputation_scores[client_id] = 10.0

                # 计算数据量因子
                data_amount = client_data_amounts.get(client_id, 1)
                ball_id = client_ball_ids.get(client_id, 0)
                ball_avg = ball_data_amounts.get(ball_id, data_amount)
                data_factor = data_amount / ball_avg if ball_avg > 0 else 1.0

                # 计算训练奖励
                training_reward = base_reward * data_factor * update_factor
                training_reward = min(training_reward, base_reward * max_reward_coefficient)
                self.reputation_scores[client_id] += training_reward

        # 处理委员会奖惩
        if consensus_reached:
            # 从有效投票中提取成员ID
            voted_members = set()
            for vote in valid_votes:
                member_id = vote.get('member_id')
                if member_id is not None:
                    voted_members.add(member_id)

            # 处理委员会成员
            for member_id in committee:
                if member_id not in self.reputation_scores:
                    self.reputation_scores[member_id] = 10.0

                # 获取成员的数据量因子
                member_data_amount = client_data_amounts.get(member_id, 1)
                member_ball_id = client_ball_ids.get(member_id, 0)
                member_ball_avg = ball_data_amounts.get(member_ball_id, member_data_amount)
                data_factor = member_data_amount / member_ball_avg if member_ball_avg > 0 else 1.0

                if member_id in voted_members:
                    # 投票被纳入QC，获得验证奖励
                    validation_reward = base_reward * (data_factor + validation_factor)
                    validation_reward = min(validation_reward, base_reward * max_reward_coefficient)
                    self.reputation_scores[member_id] += validation_reward
                else:
                    # 未投票或投票无效，扣分
                    penalty = self.reputation_scores[member_id] * validation_penalty_ratio
                    self.reputation_scores[member_id] -= penalty

            # 委员长额外治理奖励
            if chair_id in self.reputation_scores:
                chair_data_amount = client_data_amounts.get(chair_id, 1)
                chair_ball_id = client_ball_ids.get(chair_id, 0)
                chair_ball_avg = ball_data_amounts.get(chair_ball_id, chair_data_amount)
                chair_data_factor = chair_data_amount / chair_ball_avg if chair_ball_avg > 0 else 1.0

                governance_reward = base_reward * (chair_data_factor + governance_factor)
                governance_reward = min(governance_reward, base_reward * max_reward_coefficient)
                self.reputation_scores[chair_id] += governance_reward
        else:
            # 未达成共识或被弹劾
            # 检查是否是弹劾成功
            impeached = False
            if impeachment_votes:
                # 统计弹劾类型
                vote_types = {}
                for vote in impeachment_votes:
                    vote_type = vote.get('type', 'unknown')
                    if vote_type not in vote_types:
                        vote_types[vote_type] = 0
                    vote_types[vote_type] += 1

                # 检查是否有类型达到阈值
                required_votes = int(len(committee) * 0.67)  # 使用2/3阈值
                for vote_type, count in vote_types.items():
                    if count >= required_votes:
                        impeached = True
                        break

            if impeached:
                # 弹劾成功，委员长受罚
                if chair_id in self.reputation_scores:
                    penalty = self.reputation_scores[chair_id] * block_penalty_ratio
                    self.reputation_scores[chair_id] -= penalty

                # 提出成功弹劾的委员获得验证奖励
                for vote in impeachment_votes:
                    member_id = vote['member_id']
                    if member_id in self.reputation_scores:
                        # 获取成员的数据量因子
                        member_data_amount = client_data_amounts.get(member_id, 1)
                        member_ball_id = client_ball_ids.get(member_id, 0)
                        member_ball_avg = ball_data_amounts.get(member_ball_id, member_data_amount)
                        data_factor = member_data_amount / member_ball_avg if member_ball_avg > 0 else 1.0

                        validation_reward = base_reward * (data_factor + validation_factor)
                        validation_reward = min(validation_reward, base_reward * max_reward_coefficient)
                        self.reputation_scores[member_id] += validation_reward

    def save_reputation_scores(self, round_num: int):
        """
        保存信誉分到文件
        参数:
            round_num: 当前轮次
        """
        # 准备保存数据
        reputation_data = {
            'round': round_num,
            'scores': self.reputation_scores.copy(),
            'statistics': {
                'mean': float(np.mean(list(self.reputation_scores.values()))) if self.reputation_scores else 0,
                'std': float(np.std(list(self.reputation_scores.values()))) if self.reputation_scores else 0,
                'max': float(max(self.reputation_scores.values())) if self.reputation_scores else 0,
                'min': float(min(self.reputation_scores.values())) if self.reputation_scores else 0
            }
        }

        # 保存到JSON文件
        filename = os.path.join(self.reputation_dir, f'reputation_round_{round_num:04d}.json')
        with open(filename, 'w') as f:
            json.dump(reputation_data, f, indent=2)

        # 同时保存最新状态
        latest_file = os.path.join(self.reputation_dir, 'reputation_latest.json')
        with open(latest_file, 'w') as f:
            json.dump(reputation_data, f, indent=2)

    def _calculate_ball_average_data_amounts(self, client_data_amounts: Dict[int, int],
                                            client_ball_ids: Dict[int, int]) -> Dict[int, float]:
        """
        计算每个粒球的平均数据量
        参数:
            client_data_amounts: 客户端数据量
            client_ball_ids: 客户端所属粒球
        返回:
            粒球平均数据量
        """
        ball_totals = {}
        ball_counts = {}

        for client_id, ball_id in client_ball_ids.items():
            if ball_id not in ball_totals:
                ball_totals[ball_id] = 0
                ball_counts[ball_id] = 0

            ball_totals[ball_id] += client_data_amounts.get(client_id, 1)
            ball_counts[ball_id] += 1

        ball_averages = {}
        for ball_id in ball_totals:
            if ball_counts[ball_id] > 0:
                ball_averages[ball_id] = ball_totals[ball_id] / ball_counts[ball_id]
            else:
                ball_averages[ball_id] = 1

        return ball_averages

    def get_top_clients(self, n: int, exclude_ids: List[int] = None) -> List[int]:
        """
        获取信誉分最高的n个客户端
        参数:
            n: 数量
            exclude_ids: 排除的ID列表
        返回:
            客户端ID列表
        """
        exclude_ids = exclude_ids or []
        eligible_clients = [(cid, score) for cid, score in self.reputation_scores.items()
                           if cid not in exclude_ids]
        eligible_clients.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in eligible_clients[:n]]