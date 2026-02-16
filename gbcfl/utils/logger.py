"""
日志记录和工具函数模块 - 改进版
支持区分诚实节点和恶意节点的准确率记录
"""
import os
import numpy as np
import time
import json
from typing import Dict, Any, List


def ensure_output_dir(output_dir: str) -> None:
    """
    确保输出目录存在，必要时创建

    参数:
        output_dir: 要创建的目录路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


class ExperimentLogger:
    """
    训练统计和结果的实验记录器
    改进：支持诚实节点准确率的单独记录
    """

    def __init__(self, output_dir: str = "outputs"):
        """
        初始化实验记录器

        参数:
            output_dir: 日志和数据的输出目录
        """
        self.output_dir = output_dir
        self.rounds = []
        self.acc_clients = []  # 所有客户端准确率（包含NaN）
        self.honest_acc_clients = []  # 诚实客户端准确率
        self.honest_client_ids_history = []  # 诚实客户端ID历史
        self.granular_balls = []
        self.unmatched_clients = []
        self.ball_events = []
        self.block_heights = []
        self.start_time = time.time()

        # 确保输出目录存在
        ensure_output_dir(self.output_dir)

        # 初始化日志文件
        self.log_file = os.path.join(self.output_dir, "experiment_log.txt")
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("GB-CFLB 实验日志\n")
            f.write("=" * 30 + "\n")
            f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def log(self, data: Dict[str, Any]) -> None:
        """
        记录实验数据
        改进：支持诚实节点准确率的单独记录

        参数:
            data: 包含实验数据的字典
        """
        # 存储轮次和准确率数据
        if 'rounds' in data:
            if isinstance(data['rounds'], list):
                self.rounds.extend(data['rounds'])
            else:
                self.rounds.append(data['rounds'])

        # 存储所有客户端准确率（包含NaN）
        if 'acc_clients' in data:
            if isinstance(data['acc_clients'], list) and len(data['acc_clients']) > 0:
                if isinstance(data['acc_clients'][0], list):
                    self.acc_clients.extend(data['acc_clients'])
                else:
                    self.acc_clients.append(data['acc_clients'])

        # 新增：存储诚实客户端准确率
        if 'honest_acc_clients' in data:
            if isinstance(data['honest_acc_clients'], list) and len(data['honest_acc_clients']) > 0:
                if isinstance(data['honest_acc_clients'][0], list):
                    self.honest_acc_clients.extend(data['honest_acc_clients'])
                else:
                    self.honest_acc_clients.append(data['honest_acc_clients'])

        # 新增：存储诚实客户端ID列表
        if 'honest_client_ids' in data:
            self.honest_client_ids_history.append(data['honest_client_ids'])

        # 存储粒球数据
        if 'granular_balls' in data:
            self.granular_balls.append(data['granular_balls'])

        if 'unmatched_clients' in data:
            self.unmatched_clients.append(data['unmatched_clients'])

        # 存储区块链数据
        if 'block_heights' in data:
            self.block_heights.append(data['block_heights'])

        # 写入日志文件
        self._write_log_entry(data)

    def _write_log_entry(self, data: Dict[str, Any]) -> None:
        """
        将日志条目写入文件
        改进：记录诚实节点的平均准确率而不是全体平均准确率

        参数:
            data: 要记录的数据
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            timestamp = time.strftime('%H:%M:%S')
            f.write(f"[{timestamp}] ")

            if 'rounds' in data:
                round_num = data['rounds'] if not isinstance(data['rounds'], list) else data['rounds'][-1]
                f.write(f"第 {round_num} 轮: ")

            # 改进：优先使用诚实节点准确率
            if 'honest_acc_clients' in data:
                honest_acc_data = data['honest_acc_clients']
                if isinstance(honest_acc_data, list) and len(honest_acc_data) > 0:
                    if isinstance(honest_acc_data[0], list):
                        mean_acc = np.mean(honest_acc_data[-1])
                    else:
                        mean_acc = np.mean(honest_acc_data)

                    # 获取诚实节点数量
                    if 'honest_client_ids' in data:
                        honest_count = len(data['honest_client_ids'])
                    else:
                        honest_count = len(honest_acc_data)

                    f.write(f"诚实节点平均准确率: {mean_acc:.4f} ({honest_count}个节点) ")
            elif 'acc_clients' in data:
                # 后备：使用所有客户端准确率（过滤NaN）
                acc_data = data['acc_clients']
                if isinstance(acc_data, list) and len(acc_data) > 0:
                    if isinstance(acc_data[0], list):
                        valid_acc = [a for a in acc_data[-1] if not np.isnan(a)]
                    else:
                        valid_acc = [a for a in acc_data if not np.isnan(a)]

                    if valid_acc:
                        mean_acc = np.mean(valid_acc)
                        f.write(f"平均准确率: {mean_acc:.4f} ")

            if 'granular_balls' in data:
                ball_count = len(data['granular_balls'])
                f.write(f"粒球数: {ball_count} ")

            f.write("\n")

    def get_summary(self) -> Dict[str, Any]:
        """
        获取实验汇总统计
        改进：包含诚实节点准确率统计

        返回:
            包含汇总统计的字典
        """
        summary = {
            'total_rounds': len(self.rounds),
            'total_runtime': time.time() - self.start_time,
            'ball_events_count': len(self.ball_events) if hasattr(self, 'ball_events') else 0,
            'final_accuracy': None,
            'final_honest_accuracy': None,  # 新增
            'max_accuracy': None,
            'max_honest_accuracy': None,  # 新增
            'min_accuracy': None,
            'min_honest_accuracy': None,  # 新增
            'honest_client_count': None  # 新增
        }

        # 计算诚实节点准确率统计
        if self.honest_acc_clients:
            final_round_acc = self.honest_acc_clients[-1] if self.honest_acc_clients else []
            if final_round_acc:
                summary['final_honest_accuracy'] = np.mean(final_round_acc)

            all_accuracies = [np.mean(round_acc) for round_acc in self.honest_acc_clients if round_acc]
            if all_accuracies:
                summary['max_honest_accuracy'] = np.max(all_accuracies)
                summary['min_honest_accuracy'] = np.min(all_accuracies)

        # 获取诚实客户端数量
        if self.honest_client_ids_history:
            summary['honest_client_count'] = len(self.honest_client_ids_history[-1])

        # 保留原有的全体准确率统计（用于比较）
        if self.acc_clients:
            final_round_acc = self.acc_clients[-1] if self.acc_clients else []
            if final_round_acc:
                # 过滤NaN值
                valid_acc = [a for a in final_round_acc if not np.isnan(a)]
                if valid_acc:
                    summary['final_accuracy'] = np.mean(valid_acc)

            all_accuracies = []
            for round_acc in self.acc_clients:
                if round_acc:
                    valid_acc = [a for a in round_acc if not np.isnan(a)]
                    if valid_acc:
                        all_accuracies.append(np.mean(valid_acc))

            if all_accuracies:
                summary['max_accuracy'] = np.max(all_accuracies)
                summary['min_accuracy'] = np.min(all_accuracies)

        return summary

    def get_honest_accuracy_history(self) -> List[float]:
        """
        获取诚实节点准确率历史

        返回:
            每轮诚实节点平均准确率列表
        """
        if not self.honest_acc_clients:
            # 如果没有诚实节点数据，尝试从原始数据计算
            return self._calculate_honest_accuracy_from_all()

        return [np.mean(round_acc) if round_acc else 0.0
                for round_acc in self.honest_acc_clients]

    def _calculate_honest_accuracy_from_all(self) -> List[float]:
        """
        从所有客户端准确率数据中计算诚实节点准确率
        （过滤掉NaN值）

        返回:
            每轮有效准确率的平均值列表
        """
        honest_means = []
        for round_acc in self.acc_clients:
            if round_acc:
                valid_acc = [a for a in round_acc if not np.isnan(a)]
                if valid_acc:
                    honest_means.append(np.mean(valid_acc))
                else:
                    honest_means.append(0.0)
            else:
                honest_means.append(0.0)

        return honest_means