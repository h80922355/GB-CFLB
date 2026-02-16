"""
委员会独立操作器
实现委员会成员的独立验证、聚合和聚类操作流程控制
改进：确保只有通过验证的客户端参与粒球操作
"""
import torch
from typing import List, Dict, Any, Tuple
from gbcfl.utils.device_utils import get_device
from gbcfl.utils.operations import flatten
from blockchain.consensus.merkle_tree import MerkleTree

# 导入粒球操作模块
from gbcfl.granular_ball.split import should_split, split_ball_with_params
from gbcfl.granular_ball.merge import merge_balls_with_params, get_merged_model_params
from gbcfl.granular_ball.reassign import reassign_clients_with_params

device = get_device()


class IndependentOperator:
    """
    委员会成员独立操作器
    专注于训练逻辑控制，具体操作委托给granular_ball模块
    改进：确保验证结果贯穿整个操作流程
    """

    def __init__(self, member_id: int, args, chain_storage=None):
        """
        初始化操作器
        参数:
            member_id: 成员ID
            args: 训练参数
            chain_storage: 链下存储管理器
        """
        self.member_id = member_id
        self.args = args
        self.chain_storage = chain_storage
        self.update_messages = []  # 保存更新消息
        self.update_params_dict = {}  # 客户端ID -> 更新参数的映射
        self.client_data_amounts = {}  # 客户端ID -> 数据量的映射
        self.valid_clients = set()  # 新增：记录通过验证的客户端ID集合

    def execute_operations(self, temp_state, update_messages, phase='standard', current_round=1):
        """
        执行独立操作流程（验证、聚合、聚类）

        参数:
            temp_state: 临时状态
            update_messages: 更新消息列表
            phase: 训练阶段 'standard' 或 'granular'
            current_round: 当前轮次

        返回:
            (交易列表, 聚类状态, merkle_trans, merkle_cluster)
        """
        # 保存更新消息
        self.update_messages = update_messages

        # 统一构建更新参数字典和数据量字典
        self.update_params_dict = {}
        self.client_data_amounts = {}

        for msg in update_messages:
            client_id = msg['client_id']
            self.update_params_dict[client_id] = msg.get('update_params', {})
            self.client_data_amounts[client_id] = msg.get('data_amount', 1)

        # 步骤1: 参数验证
        valid_updates = self._validate_updates(temp_state, update_messages, phase)

        # 记录通过验证的客户端ID
        self.valid_clients = set(valid_updates.keys())

        # 步骤2: 粒球聚合（只使用验证通过的更新）
        self._aggregate_balls(temp_state, valid_updates)

        # 步骤3: 聚类操作（根据阶段执行不同操作，只使用验证通过的客户端）
        if phase == 'standard':
            self._handle_standard_phase(temp_state, current_round)
        else:
            self._handle_granular_phase(temp_state, current_round)

        # 步骤4: 生成merkle根
        transactions = temp_state.get_sorted_transactions()
        cluster_states = temp_state.get_cluster_states()

        merkle_trans = MerkleTree().compute_root(transactions)
        merkle_cluster = MerkleTree().compute_root(cluster_states)

        return transactions, cluster_states, merkle_trans, merkle_cluster

    def _validate_updates(self, temp_state, update_messages, phase):
        """
        验证参数更新

        参数:
            temp_state: 临时状态
            update_messages: 更新消息
            phase: 训练阶段

        返回:
            有效更新字典 {client_id: update_params}
        """
        valid_updates = {}

        # 根据阶段选择验证参数
        if phase == 'standard':
            norm_threshold = self.args.blockchain_config['validation']['standard_phase']['norm_threshold']
            similarity_threshold = self.args.blockchain_config['validation']['standard_phase']['similarity_threshold']
        else:
            norm_threshold = self.args.blockchain_config['validation']['ball_phase']['norm_threshold']
            similarity_threshold = self.args.blockchain_config['validation']['ball_phase']['similarity_threshold']

        for msg in update_messages:
            client_id = msg['client_id']
            ball_id = msg['ball_id']
            update_params = msg['update_params']

            # 验证范数
            update_tensor = flatten(update_params)
            update_norm = torch.norm(update_tensor).item()

            if update_norm > norm_threshold:
                print(f"客户端{client_id}更新参数范数{update_norm:.4f}超过阈值{norm_threshold}")
                continue

            # 验证相似度
            ball = temp_state.get_ball_by_id(ball_id)
            if ball and ball.center is not None:
                center_norm = torch.norm(ball.center)
                if center_norm > 1e-8 and update_norm > 1e-8:
                    similarity = torch.dot(ball.center, update_tensor) / (center_norm * update_norm)
                    similarity = similarity.item()

                    if similarity < similarity_threshold:
                        print(f"客户端{client_id}相似度{similarity:.4f}低于阈值{similarity_threshold}")
                        continue

            # 通过验证
            valid_updates[client_id] = update_params

        return valid_updates

    def _aggregate_balls(self, temp_state, valid_updates):
        """
        执行粒球聚合
        改进：只使用验证通过的更新

        参数:
            temp_state: 临时状态
            valid_updates: 有效更新
        """
        # 对每个粒球执行聚合
        for ball in temp_state.balls:
            # 筛选属于该粒球的有效更新
            ball_updates = {
                cid: valid_updates[cid]
                for cid in ball.client_ids
                if cid in valid_updates
            }

            if ball_updates:
                # 计算聚合更新（数据量加权）
                aggregate_update = self._compute_weighted_aggregate(ball_updates)

                # 更新粒球模型
                temp_state.update_ball_model(ball.ball_id, aggregate_update)

                # 更新粒球状态（中心、纯度等）
                self._update_ball_metrics(ball, ball_updates)

                # 生成聚合交易
                tx = {
                    'type': 'aggregation',
                    'ball_id': ball.ball_id,
                    'valid_clients': list(ball_updates.keys()),
                    'total_data_amount': sum(self.client_data_amounts.get(cid, 1)
                                            for cid in ball_updates.keys())
                }
                temp_state.add_transaction(tx)

    def _compute_weighted_aggregate(self, ball_updates):
        """
        计算数据量加权的聚合更新

        参数:
            ball_updates: 粒球内的更新参数

        返回:
            聚合后的更新参数
        """
        aggregate_update = {}

        # 使用统一的client_data_amounts
        total_data = sum(self.client_data_amounts.get(cid, 1)
                        for cid in ball_updates.keys())

        if total_data == 0:
            print(f"警告：粒球总数据量为0，使用均等权重")
            total_data = len(ball_updates)
            for cid in ball_updates.keys():
                self.client_data_amounts[cid] = 1

        for cid, update_params in ball_updates.items():
            weight = self.client_data_amounts.get(cid, 1) / total_data

            for key, value in update_params.items():
                if key not in aggregate_update:
                    aggregate_update[key] = torch.zeros_like(value).to(device)
                aggregate_update[key] = aggregate_update[key] + value * weight

        return aggregate_update

    def _update_ball_metrics(self, ball, ball_updates):
        """
        更新粒球指标（中心、纯度等）

        参数:
            ball: 粒球对象
            ball_updates: 粒球内的更新参数
        """
        # 更新粒球中心
        update_tensors = {
            cid: flatten(update)
            for cid, update in ball_updates.items()
        }
        ball.update_center(update_tensors)

        # 计算相似度并更新纯度
        similarities = {}
        if ball.center is not None:
            for cid, update_tensor in update_tensors.items():
                center_norm = torch.norm(ball.center)
                update_norm = torch.norm(update_tensor)
                if center_norm > 1e-8 and update_norm > 1e-8:
                    sim = torch.dot(ball.center, update_tensor) / (center_norm * update_norm)
                    similarities[cid] = sim.item()
                else:
                    similarities[cid] = 0.0

        ball.update_purity(similarities)
        ball.calculate_convergence_indicator()

    def _handle_standard_phase(self, temp_state, current_round):
        """
        处理标准联邦学习阶段
        只检查粒球0的分裂条件，用于阶段转换

        参数:
            temp_state: 临时状态
            current_round: 当前轮次
        """
        initial_ball = temp_state.get_ball_by_id(0)
        print("")
        print("="*60)
        print(f"粒球0:纯度：{initial_ball.purity:.2f}，收敛度：{initial_ball.convergence:.2f}")

        # 检查是否达到初始稳定轮次
        if current_round < self.args.init_rounds:
            return

        # 只检查ball_id=0的分裂条件
        if not initial_ball:
            return

        # 判断是否应该分裂
        can_split, split_details = should_split(initial_ball, self.args, current_round)

        print("")
        print(f"    委员{self.member_id} - 粒球0分裂判定:")
        print(f"      收敛指标: {split_details['convergence']:.4f} (阈值: {self.args.convergence_threshold})")
        print(f"      平均纯度: {split_details.get('average_purity', 0):.4f} (阈值: {self.args.purity_threshold})")
        print(f"      客户端数: {len(initial_ball.client_ids)} (最小: {2 * self.args.min_size})")

        if can_split:
            # 检查是否有足够的有效客户端进行分裂
            valid_clients_in_ball = [cid for cid in initial_ball.client_ids if cid in self.valid_clients]
            if len(valid_clients_in_ball) < 2 * self.args.min_size:
                print(f"      判定结果: 有效客户端不足({len(valid_clients_in_ball)}个)，无法分裂")
                return

            print(f"      判定结果: 满足分裂条件，执行分裂")
            self._execute_split(temp_state, initial_ball, current_round)
        else:
            print(f"      判定结果: 不满足分裂条件")

    def _handle_granular_phase(self, temp_state, current_round):
        """
        处理粒球聚类阶段
        执行分裂、合并、重分配操作

        参数:
            temp_state: 临时状态
            current_round: 当前轮次
        """
        print("")
        print("="*60)
        print(f"    委员{self.member_id} - 粒球聚类操作:")

        # 更新冷却期
        for ball in temp_state.balls:
            if ball.cooling_period > 0:
                ball.cooling_period -= 1
                print(f"      粒球{ball.ball_id}冷却期剩余: {ball.cooling_period}轮")

        # 1. 分裂操作
        self._perform_split_operations(temp_state, current_round)

        # 2. 合并操作
        self._perform_merge_operations(temp_state, current_round)

        # 3. 重分配操作
        self._perform_reassign_operations(temp_state)

    def _perform_split_operations(self, temp_state, current_round):
        """
        执行分裂操作
        """
        print(f"      分裂判定:")
        balls_to_split = []

        for ball in temp_state.balls:
            if ball.cooling_period > 0:
                print(f"        粒球{ball.ball_id}: 处于冷却期，跳过")
                continue

            # 检查粒球中是否有足够的有效客户端
            valid_clients_in_ball = [cid for cid in ball.client_ids if cid in self.valid_clients]
            if len(valid_clients_in_ball) < 2 * self.args.min_size:
                print(f"        粒球{ball.ball_id}: 有效客户端不足({len(valid_clients_in_ball)}个)，跳过")
                continue

            can_split, split_details = should_split(ball, self.args, current_round)

            print(f"        粒球{ball.ball_id}:")
            print(f"          收敛指标: {split_details['convergence']:.4f}")
            print(f"          平均纯度: {split_details.get('average_purity', 0):.4f}")
            print(f"          客户端数: {len(ball.client_ids)}")
            print(f"          有效客户端数: {len(valid_clients_in_ball)}")

            if can_split:
                balls_to_split.append(ball)
                print(f"          判定: 满足分裂条件")
            else:
                print(f"          判定: 不满足分裂条件")

        for ball in balls_to_split:
            self._execute_split(temp_state, ball, current_round)

    def _execute_split(self, temp_state, ball, current_round):
        """
        执行单个粒球的分裂
        改进：只使用通过验证的客户端进行分裂
        """
        # 只准备通过验证的客户端的更新参数
        valid_client_updates = {}
        invalid_client_ids = []

        for cid in ball.client_ids:
            if cid in self.valid_clients and cid in self.update_params_dict:
                # 构建包含参数和数据量的字典
                valid_client_updates[cid] = {
                    'params': self.update_params_dict[cid],
                    'data_amount': self.client_data_amounts.get(cid, 1)
                }
            else:
                # 记录未通过验证的客户端
                invalid_client_ids.append(cid)

        # 如果有效客户端不足，不执行分裂
        if len(valid_client_updates) < 2 * self.args.min_size:
            print(f"      粒球{ball.ball_id}有效客户端不足，取消分裂")
            return

        # 创建一个只包含有效客户端的临时粒球
        temp_ball = ball.clone()
        temp_ball.client_ids = list(valid_client_updates.keys())

        # 执行分裂（只对有效客户端）
        split_result, unmatched_ids = split_ball_with_params(
            temp_ball,
            valid_client_updates,
            self.args.min_size,
            self.args.similarity_threshold,
            current_round
        )

        if len(split_result) > 1:
            print(f"      粒球{ball.ball_id}分裂成功: 生成{len(split_result)}个新粒球")

            # 获取父粒球模型
            parent_model = temp_state.get_ball_model(ball.ball_id)

            # 移除原粒球
            temp_state.remove_ball(ball.ball_id)

            # 添加新粒球
            new_ball_ids = []
            for new_ball in split_result:
                new_ball.ball_id = temp_state.allocate_new_ball_id()
                new_ball_ids.append(new_ball.ball_id)
                # 使用配置文件的冷却期
                new_ball.cooling_period = self.args.cooling_period
                temp_state.add_ball(new_ball, parent_model)

                print(f"        子粒球{new_ball.ball_id}: {new_ball.client_ids} ")

            # 添加未分配客户端（包括未通过验证的客户端）
            all_unmatched = unmatched_ids + invalid_client_ids
            if all_unmatched:
                print(f"        分裂期间未分配客户端:{all_unmatched}")
                for cid in all_unmatched:
                    if cid not in temp_state.unmatched_clients:
                        temp_state.unmatched_clients.append(cid)

            # 生成分裂交易
            tx = {
                'type': 'split',
                'source_ball_id': ball.ball_id,
                'child_balls': [
                    {
                        'ball_id': new_ball.ball_id,
                        'clients': new_ball.client_ids,
                        'data_amount': sum(self.client_data_amounts.get(cid, 1)
                                         for cid in new_ball.client_ids)
                    }
                    for new_ball in split_result
                ],
                'unassigned_clients': all_unmatched
            }
            temp_state.add_transaction(tx)

    def _perform_merge_operations(self, temp_state, current_round):
        """
        执行合并操作
        改进：只使用通过验证的客户端参与合并
        """
        print(f"      合并判定:")

        # 准备所有粒球的更新参数（只包含有效客户端）
        balls_updates = {}

        for ball in temp_state.balls:
            ball_updates = {}
            for cid in ball.client_ids:
                if cid in self.valid_clients and cid in self.update_params_dict:
                    ball_updates[cid] = self.update_params_dict[cid]
            balls_updates[ball.ball_id] = ball_updates

        # 执行三阶段合并
        merged, new_balls = merge_balls_with_params(
            temp_state.balls,
            balls_updates,
            self.args,
            current_round,
            self.chain_storage
        )

        if merged:
            # 计算每个粒球的总数据量（只计算有效客户端）
            ball_data_amounts = {}
            for ball in temp_state.balls:
                ball_total_data = sum(self.client_data_amounts.get(cid, 1)
                                      for cid in ball.client_ids
                                      if cid in self.valid_clients)
                ball_data_amounts[ball.ball_id] = ball_total_data

            # 找出被合并的粒球
            old_ball_ids = set(ball.ball_id for ball in temp_state.balls)
            remaining_ball_ids = set(ball.ball_id for ball in new_balls if ball.ball_id in old_ball_ids)
            merged_ball_ids = old_ball_ids - remaining_ball_ids

            print(f"        被合并的粒球: {list(merged_ball_ids)}")

            # 收集所有被合并粒球的模型和数据量
            merge_groups = []
            processed_ids = set()

            # 从new_balls中找出新创建的粒球
            new_merged_balls = [ball for ball in new_balls if ball.ball_id not in old_ball_ids]

            # 为每个新合并的粒球找出其源粒球
            for new_ball in new_merged_balls:
                # 根据client_ids找出源粒球
                source_ball_ids = []
                source_models = []
                source_data_amounts = []

                for old_ball_id in merged_ball_ids:
                    if old_ball_id not in processed_ids:
                        old_ball = temp_state.get_ball_by_id(old_ball_id)
                        if old_ball:
                            # 检查客户端是否在新粒球中
                            if any(cid in new_ball.client_ids for cid in old_ball.client_ids):
                                source_ball_ids.append(old_ball_id)
                                model = temp_state.get_ball_model(old_ball_id)
                                if model:
                                    source_models.append(model)
                                    source_data_amounts.append(ball_data_amounts.get(old_ball_id, 0))
                                    processed_ids.add(old_ball_id)

                if len(source_models) >= 2:
                    merge_groups.append({
                        'new_ball': new_ball,
                        'source_ids': source_ball_ids,
                        'source_models': source_models,
                        'source_data_amounts': source_data_amounts
                    })

            # 清除所有被合并的旧粒球
            for ball_id in merged_ball_ids:
                temp_state.remove_ball(ball_id)
                print(f"        移除旧粒球: {ball_id}")

            # 处理每个合并组
            for merge_group in merge_groups:
                new_ball = merge_group['new_ball']
                source_ids = merge_group['source_ids']
                source_models = merge_group['source_models']
                source_data_amounts = merge_group['source_data_amounts']

                # 分配新ID
                new_ball.ball_id = temp_state.allocate_new_ball_id()

                # 计算合并后的模型（基于数据量加权）
                if len(source_models) >= 2:
                    # 递归合并多个模型
                    merged_model = source_models[0]
                    accumulated_data = source_data_amounts[0]

                    for i in range(1, len(source_models)):
                        print(f"        合并模型: 数据量{accumulated_data} + {source_data_amounts[i]}")
                        merged_model = get_merged_model_params(
                            merged_model,
                            source_models[i],
                            accumulated_data,
                            source_data_amounts[i]
                        )
                        accumulated_data += source_data_amounts[i]
                else:
                    merged_model = source_models[0] if source_models else None

                # 添加到临时状态
                temp_state.add_ball(new_ball, merged_model)

                # 计算新粒球的总数据量
                new_ball_data = sum(self.client_data_amounts.get(cid, 1)
                                    for cid in new_ball.client_ids
                                    if cid in self.valid_clients)

                print(f"        新粒球{new_ball.ball_id}创建: {len(source_ids)}个源粒球合并")
                print(f"          源粒球: {source_ids}")
                print(f"          源数据量: {source_data_amounts}")
                print(f"          客户端数: {len(new_ball.client_ids)}")
                print(f"          总数据量: {new_ball_data}")

                # 生成合并交易
                tx = {
                    'type': 'merge',
                    'ball_ids': source_ids,
                    'merged_ball_id': new_ball.ball_id,
                    'source_data_amounts': source_data_amounts,
                    'merged_data_amount': new_ball_data
                }
                temp_state.add_transaction(tx)

    def _perform_reassign_operations(self, temp_state):
        """
        执行重分配操作
        改进：只重分配通过验证的未匹配客户端
        """
        print(f"      重分配判定:")

        if not temp_state.unmatched_clients:
            print(f"        无未分配客户端，跳过重分配")
            return

        # 只重分配通过验证的未匹配客户端
        valid_unmatched = [cid for cid in temp_state.unmatched_clients if cid in self.valid_clients]

        if not valid_unmatched:
            print(f"        无有效的未分配客户端，跳过重分配")
            return

        print(f"        未分配客户端数量: {len(temp_state.unmatched_clients)}")
        print(f"        有效未分配客户端数量: {len(valid_unmatched)}")

        # 准备未匹配客户端的更新参数
        unmatched_updates = {}
        for cid in valid_unmatched:
            if cid in self.update_params_dict:
                unmatched_updates[cid] = self.update_params_dict[cid]

        # 执行重分配
        assignments = reassign_clients_with_params(
            valid_unmatched,
            unmatched_updates,
            temp_state.balls,
            self.args.similarity_threshold
        )

        if assignments:
            # 更新粒球和未匹配列表
            for ball_id, client_ids in assignments.items():
                ball = temp_state.get_ball_by_id(ball_id)
                if ball:
                    for cid in client_ids:
                        if cid not in ball.client_ids:
                            ball.client_ids.append(cid)
                        if cid in temp_state.unmatched_clients:
                            temp_state.unmatched_clients.remove(cid)

                        data_amount = self.client_data_amounts.get(cid, 1)
                        print(f"        客户端{cid}(数据量{data_amount})分配至粒球{ball.ball_id}")

            # 生成重分配交易
            tx = {
                'type': 'reassign',
                'assignments': [
                    {
                        'ball_id': ball_id,
                        'clients': client_ids,
                        'total_data_amount': sum(self.client_data_amounts.get(cid, 1)
                                               for cid in client_ids)
                    }
                    for ball_id, client_ids in assignments.items()
                ]
            }
            temp_state.add_transaction(tx)

            print(f"        重分配完成: {len(assignments)}个粒球接收了新客户端")