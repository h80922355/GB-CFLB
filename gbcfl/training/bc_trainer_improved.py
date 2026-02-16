"""
改进的区块链粒球联邦学习训练器
实现完整的共识流程和临时粒球机制
新增：锚点模型和合并依据管理
新增：恶意节点投票攻击支持
改进：只评估诚实节点的准确率
修复：保持原有突发数据变化API调用
"""
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from colorama import Fore, Style

from gbcfl.utils.logger import ExperimentLogger, ensure_output_dir
from gbcfl.utils.device_utils import get_device
from gbcfl.utils.operations import flatten
from gbcfl.granular_ball.ball import GranularBall
from gbcfl.visualization.console_output import (
    print_title, print_stage, print_complete, print_info,
    print_success, print_warning, print_error
)

# 导入区块链模块
from blockchain.state.system_state import SystemState
from blockchain.state.temp_state import TempState
from blockchain.committee.committee_manager import CommitteeManager
from blockchain.committee.independent_operator import IndependentOperator
from blockchain.committee.impeachment import ImpeachmentManager
from blockchain.consensus.merkle_tree import MerkleTree
from blockchain.consensus.qc_manager import QCManager
from blockchain.consensus.validator import ConsensusValidator
from blockchain.reputation.reputation_manager import ReputationManager
from blockchain.storage.chain_storage import ChainStorage

device = get_device()


def train_bc_gbcfl_improved(clients, test_data, model_func, args, cfl_stats=None,
                           malicious_client_ids=None):
    """
    改进的GB-CFLB训练主函数
    实现完整共识流程和临时粒球机制
    新增：锚点模型保存和合并依据初始化
    新增：恶意节点攻击支持
    改进：只评估诚实节点的准确率

    参数:
        clients: 客户端列表
        test_data: 测试数据集
        model_func: 模型构造函数
        args: 训练参数
        cfl_stats: 实验记录器
        malicious_client_ids: 恶意客户端ID列表

    返回:
        cfl_stats: 更新后的实验记录器
    """
    print_title("GB-CFLB改进系统启动")

    # 如果没有提供恶意客户端列表，初始化为空列表
    if malicious_client_ids is None:
        malicious_client_ids = []

    # 识别诚实客户端ID集合（用于准确率评估）
    honest_client_ids = set(range(len(clients))) - set(malicious_client_ids)

    # 初始化系统组件
    system_state = SystemState()
    temp_state = TempState()

    # 读取区块链配置中的候选池比例
    candidate_pool_percentage = args.blockchain_config['committee'].get(
        'candidate_pool_percentage', 0.7
    )

    committee_manager = CommitteeManager(
        committee_size=args.blockchain_config['committee']['size'],
        rotation_interval=args.blockchain_config['committee']['rotation_interval'],
        candidate_pool_percentage=candidate_pool_percentage
    )

    impeachment_manager = ImpeachmentManager(
        threshold=args.blockchain_config['committee']['consensus_threshold']
    )

    qc_manager = QCManager(
        threshold=args.blockchain_config['committee']['consensus_threshold']
    )

    reputation_manager = ReputationManager()
    chain_storage = ChainStorage(args.output_dir)

    # 创建验证器并设置依赖
    validator = ConsensusValidator()

    # 设置相互引用
    committee_manager.set_reputation_manager(reputation_manager)
    qc_manager.set_committee_manager(committee_manager)
    validator.set_qc_manager(qc_manager)
    validator.set_committee_manager(committee_manager)

    if cfl_stats is None:
        cfl_stats = ExperimentLogger()

    # 创建测试数据加载器
    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # 步骤1: 系统初始化
    print_stage("系统初始化")

    # 1.1 提取所有客户端ID
    client_ids = [client.id for client in clients]

    # 1.2 初始化所有客户端的持久密钥对
    committee_manager.initialize_all_client_keys(client_ids)

    # 1.3 初始化信誉分
    reputation_manager.initialize_scores(client_ids)

    # 1.4 创建初始粒球(ID=0，包含所有客户端ID)
    initial_ball = GranularBall([], ball_id=0)
    initial_ball.client_ids = client_ids.copy()

    # 1.5 初始化全局模型（使用模型构造函数创建初始模型）
    initial_model_instance = model_func().to(device)
    initial_model = {key: value.clone() for key, value in initial_model_instance.named_parameters()}

    # 1.6 初始化系统状态
    system_state.initialize(initial_ball, initial_model)

    # 1.7 初始化区块链
    blockchain = []
    latest_block = None

    # 1.8 选举初始委员会（基于信誉分）
    committee_info = committee_manager.initialize_committee(client_ids)

    print_complete()
    print_info(f"委员会初始化: {len(committee_info['committee'])}个成员")
    print_info(f"委员长ID: {committee_info['chair_id']}")

    # 输出恶意节点信息（如果有）
    if malicious_client_ids:
        malicious_in_committee = [mid for mid in malicious_client_ids if mid in committee_info['committee']]
        if malicious_in_committee:
            print_warning(f"委员会中有 {len(malicious_in_committee)} 个恶意节点: {malicious_in_committee}")
        print_info(f"诚实客户端数量: {len(honest_client_ids)}")

    # 初始化所有客户端模型（使用初始全局模型）
    for client in clients:
        client.synchronize_with_ball_model(initial_model)

    # 记录客户端的初始旋转信息
    for i, client in enumerate(clients):
        if hasattr(client, 'rotation_angle'):
            pass
        else:
            client.rotation_angle = 0
            client.rotation_cluster_id = -1

    # 训练阶段标记
    phase = 'standard'  # 'standard' 或 'granular'
    first_split_round = -1
    merge_basis_initialization_pending = False  # 标记是否待初始化合并依据

    # 标记是否已触发突发变化
    sudden_change_triggered = False

    # 获取攻击配置
    attack_config = args.attack_config if hasattr(args, 'attack_config') else {'enabled': False}

    # 主训练循环
    for round_num in range(1, args.communication_rounds + 1):
        print_title(f"[轮次 {round_num}/{args.communication_rounds}] 模式: {phase}")

        # ========== 突发数据变化处理（保持原有API） ==========
        if (not sudden_change_triggered and
            hasattr(args, 'sudden_change_config') and
            args.sudden_change_config.get('enabled', False) and
            round_num == args.sudden_change_config.get('trigger_round', 40)):

            print("")
            print("="*60)
            print(f"{Fore.YELLOW}{Style.BRIGHT}触发突发数据变化（第{round_num}轮）{Style.RESET_ALL}")
            print("="*60)

            # 获取配置
            affected_ratio = args.sudden_change_config.get('affected_ratio', 0.5)
            additional_rotation = args.sudden_change_config.get('additional_rotation', -1)

            # 统计所有旋转聚类
            rotation_clusters = {}
            for client in clients:
                cluster_id = getattr(client, 'rotation_cluster_id', -1)
                if cluster_id == -1:
                    if hasattr(client, 'rotation_angle'):
                        if client.rotation_angle == 0:
                            cluster_id = 0
                        elif client.rotation_angle == 90:
                            cluster_id = 1
                        elif client.rotation_angle == 180:
                            cluster_id = 2
                        elif client.rotation_angle == 270:
                            cluster_id = 3
                        else:
                            cluster_id = int(client.rotation_angle / 90)
                    else:
                        cluster_id = 0

                if cluster_id not in rotation_clusters:
                    rotation_clusters[cluster_id] = []
                rotation_clusters[cluster_id].append(client.id)

            # 计算旋转角度
            num_clusters = len(rotation_clusters)
            if additional_rotation == -1 and num_clusters > 0:
                additional_rotation = 360 // num_clusters
                print(f"自动计算额外旋转角度: 360/{num_clusters} = {additional_rotation}°")
            elif additional_rotation == -1:
                additional_rotation = 90
                print(f"使用默认额外旋转角度: {additional_rotation}°")
            else:
                print(f"使用配置的额外旋转角度: {additional_rotation}°")

            print(f"受影响比例: {affected_ratio * 100:.0f}%")
            print("")

            # 对每个旋转聚类应用突发变化
            all_affected_clients = []
            affected_clients_for_basis_update = []  # 需要更新合并依据的客户端

            for cluster_id in sorted(rotation_clusters.keys()):
                client_ids_in_cluster = rotation_clusters[cluster_id]
                original_rotation = cluster_id * (360 // num_clusters) if num_clusters > 0 else 0

                print(f"聚类{cluster_id} (原始旋转{original_rotation}°):")
                print(f"  客户端总数: {len(client_ids_in_cluster)}")

                num_affected = max(1, int(len(client_ids_in_cluster) * affected_ratio))

                import random
                affected_in_cluster = random.sample(client_ids_in_cluster, num_affected)
                affected_in_cluster.sort()

                print(f"  受影响客户端数: {num_affected}")
                if len(affected_in_cluster) <= 20:
                    print(f"  受影响客户端ID: {affected_in_cluster}")
                else:
                    print(f"  受影响客户端ID: {affected_in_cluster[:10]} ... {affected_in_cluster[-10:]}")

                # 应用额外旋转（使用原有API）
                for client_id in affected_in_cluster:
                    clients[client_id].apply_additional_rotation(additional_rotation)
                    all_affected_clients.append(client_id)
                    affected_clients_for_basis_update.append(client_id)

                new_rotation = (original_rotation + additional_rotation) % 360
                print(f"  新旋转角度: {original_rotation}° + {additional_rotation}° = {new_rotation}°")
                print("")

            # 如果已初始化合并依据，更新受影响客户端的合并依据
            if phase == 'granular' and chain_storage.load_anchor_model() is not None:
                print("="*60)
                print(f"{Fore.CYAN}更新受影响客户端的合并依据{Style.RESET_ALL}")

                anchor_model = chain_storage.load_anchor_model()
                merge_basis_dict = {}

                for client_id in affected_clients_for_basis_update:
                    client = clients[client_id]
                    if client.data_changed:
                        print(f"  客户端{client_id}: 使用锚点模型重新计算合并依据...")
                        merge_basis = client.compute_merge_basis_with_anchor(anchor_model)
                        if merge_basis is not None:
                            merge_basis_dict[client_id] = merge_basis

                # 批量保存更新后的合并依据
                if merge_basis_dict:
                    chain_storage.batch_save_merge_basis(merge_basis_dict, round_num)
                    print(f"  更新了{len(merge_basis_dict)}个客户端的合并依据")

            print("="*60)
            print(f"{Fore.GREEN}突发变化完成: 共{len(all_affected_clients)}个客户端受影响{Style.RESET_ALL}")
            print("="*60)
            print("")

            if cfl_stats:
                cfl_stats.log({
                    'sudden_change_event': {
                        'round': round_num,
                        'affected_clients': all_affected_clients,
                        'additional_rotation': additional_rotation,
                        'clusters_affected': list(rotation_clusters.keys()),
                        'affected_ratio': affected_ratio
                    }
                })

            sudden_change_triggered = True

        # ========== 输出当前聚类结构（完整输出所有客户端ID） ==========
        print(f"当前聚类结构:")
        for ball in system_state.balls:
            # 完整输出所有客户端ID，不省略
            client_list = ", ".join(map(str, ball.client_ids))
            print(f"  粒球{ball.ball_id}: {len(ball.client_ids)}个客户端 [{client_list}]")

        if system_state.unmatched_clients:
            # 完整输出所有未分配客户端ID
            unmatched_list = ", ".join(map(str, system_state.unmatched_clients))
            print(f"  未分配客户端: {len(system_state.unmatched_clients)}个 [{unmatched_list}]")

        # 步骤2: 委员会管理
        if committee_manager.should_rotate(round_num):
            print_stage("委员会轮换")
            committee_info = committee_manager.rotate_committee(client_ids)
            print_complete()
            print_info(f"新委员长ID: {committee_info['chair_id']}")

            # 输出恶意节点信息（如果有）
            if malicious_client_ids:
                malicious_in_committee = [mid for mid in malicious_client_ids if mid in committee_info['committee']]
                if malicious_in_committee:
                    print_warning(f"委员会中有 {len(malicious_in_committee)} 个恶意节点: {malicious_in_committee}")

        # 步骤3: 客户端本地训练
        print_stage("客户端本地训练")
        update_messages = []

        for client in clients:
            # 本地训练
            client.compute_weight_update(epochs=args.local_epochs)

            # 准备更新消息（传入当前轮次用于攻击判断）
            msg, update_params = client.prepare_update_message(current_round=round_num)
            msg['update_params'] = update_params
            msg['encrypted'] = False
            update_messages.append(msg)

            # 如果是合并依据待初始化且是阶段转换后第一轮，提取合并依据
            if merge_basis_initialization_pending and phase == 'granular':
                # 复用训练结果作为合并依据
                merge_basis = client.extract_merge_basis_from_update()
                # 暂时存储，稍后批量保存

            # 恢复训练前状态
            client.reset()
            # 保留更新参数
            client.dW = update_params

        print_complete()

        # 在阶段转换后第一轮初始化合并依据
        if merge_basis_initialization_pending and phase == 'granular':
            print_stage("初始化客户端合并依据")

            merge_basis_dict = {}
            for client in clients:
                # 复用本轮训练结果
                merge_basis = flatten(client.dW)
                merge_basis_dict[client.id] = merge_basis
                client.merge_basis = merge_basis
                client.merge_basis_initialized = True

            # 批量保存合并依据
            chain_storage.batch_save_merge_basis(merge_basis_dict, round_num)
            merge_basis_initialization_pending = False

            print_complete()
            print_info(f"已初始化{len(merge_basis_dict)}个客户端的合并依据")

        # 步骤4-6: 委员会独立执行验证、聚合、聚类
        print_stage("委员会独立执行操作")

        member_results = {}
        committee = committee_info['committee']

        # 确保委员长最后执行
        ordered_committee = [m for m in committee if m != committee_info['chair_id']]
        ordered_committee.append(committee_info['chair_id'])

        for member_id in ordered_committee:
            # 每个成员从系统状态克隆临时状态
            member_temp_state = TempState()
            member_temp_state.clone_from_system(system_state)

            # 创建独立操作器（传递chain_storage）
            operator = IndependentOperator(member_id, args, chain_storage)

            # 执行操作
            transactions, cluster_states, merkle_trans, merkle_cluster = \
                operator.execute_operations(member_temp_state, update_messages, phase, round_num)

            # 保存结果
            member_results[member_id] = {
                'transactions': transactions,
                'cluster_states': cluster_states,
                'merkle_trans': merkle_trans,
                'merkle_cluster': merkle_cluster,
                'temp_state': member_temp_state
            }

        print_complete()

        # 步骤5: 委员长生成区块提议
        print_stage("生成区块提议")

        chair_result = member_results[committee_info['chair_id']]

        # 生成区块提议
        block_proposal = {
            'proposer_id': committee_info['chair_id'],
            'view': round_num,
            'parent_hash': latest_block['hash'] if latest_block else 'genesis',
            'parent_qc': latest_block['header']['qc'] if latest_block else None,
            'timestamp': time.time(),
            'merkle_cluster_hash': MerkleTree.hash_data(chair_result['merkle_cluster']),
            'merkle_trans_hash': MerkleTree.hash_data(chair_result['merkle_trans'])
        }

        print_complete()

        # 步骤6: 委员会投票（包括恶意节点攻击逻辑）
        print_stage("委员会投票")
        votes = []
        impeachment_votes = []

        from blockchain.consensus.crypto_signer import CryptoSigner
        signer = CryptoSigner()

        # 检查是否启用投票攻击
        voting_attack_enabled = False
        if attack_config.get('enabled', False):
            voting_attack = attack_config.get('voting_attack', {})
            if voting_attack.get('enabled', False) and round_num >= voting_attack.get('start_round', 1):
                voting_attack_enabled = True
                print_warning(f"投票攻击已激活（轮次 {round_num}）")

        for member_id in committee:
            result = member_results[member_id]

            # 使用validator验证区块提议
            validation_result = validator.validate_block_proposal_with_merkle(
                block_proposal,
                result['merkle_cluster'],
                result['merkle_trans'],
                committee_info
            )

            # 检查是否为恶意节点
            is_malicious_member = member_id in malicious_client_ids and voting_attack_enabled

            # 决定投票（恶意节点反转投票）
            if is_malicious_member:
                # 恶意节点：反转投票逻辑
                should_vote_yes = not validation_result['valid']  # 反转
                if validation_result['valid']:
                    print_warning(f"恶意委员 {member_id} 对有效区块投反对票")
                else:
                    print_warning(f"恶意委员 {member_id} 对无效区块投赞成票")
            else:
                # 正常节点：按验证结果投票
                should_vote_yes = validation_result['valid']

            if should_vote_yes:
                # 投赞成票
                # 构建区块头
                block_header = {
                    'proposer_id': block_proposal['proposer_id'],
                    'view': block_proposal['view'],
                    'parent_hash': block_proposal['parent_hash'],
                    'parent_qc': block_proposal['parent_qc'],
                    'timestamp': block_proposal['timestamp'],
                    'merkle_cluster': result['merkle_cluster'],
                    'merkle_trans': result['merkle_trans']
                }

                # 获取成员的持久签名私钥
                private_key_hex = committee_manager.get_member_private_key(member_id)
                if private_key_hex:
                    # 计算区块头哈希
                    block_hash = signer.compute_block_hash(block_header)

                    # 生成签名
                    signature = signer.sign(block_hash, private_key_hex)

                    # 创建投票
                    vote = {
                        'member_id': member_id,
                        'block_hash': block_hash,
                        'signature': signature
                    }
                    votes.append(vote)
            else:
                # 投反对票（弹劾投票）
                impeachment_votes.append({
                    'member_id': member_id,
                    'type': 'proposal_invalid'
                })

        print_complete()
        print_info(f"收到 {len(votes)} 个有效投票, {len(impeachment_votes)} 个弹劾投票")

        # 步骤7: 共识达成
        print_stage("共识达成")

        # 委员长验证投票
        valid_votes = []
        chair_block_header = {
            'proposer_id': block_proposal['proposer_id'],
            'view': block_proposal['view'],
            'parent_hash': block_proposal['parent_hash'],
            'parent_qc': block_proposal['parent_qc'],
            'timestamp': block_proposal['timestamp'],
            'merkle_cluster': chair_result['merkle_cluster'],
            'merkle_trans': chair_result['merkle_trans']
        }

        # 区块哈希永远不包含QC
        chair_block_hash = signer.compute_block_hash(chair_block_header)

        for vote in votes:
            if validator.validate_vote(vote, chair_block_hash):
                valid_votes.append(vote)

        # 检查是否达到共识阈值
        consensus_reached = len(valid_votes) >= int(
            len(committee) * args.blockchain_config['committee']['consensus_threshold'])

        # 保存正式区块用于后续验证
        new_block = None
        valid_update_clients = []

        if consensus_reached:
            # 生成QC
            qc = qc_manager.generate_qc(
                chair_block_hash,
                round_num,
                valid_votes,
                committee_info['committee']
            )

            # 将QC加入区块头
            chair_block_header['qc'] = qc

            # 构建正式区块
            new_block = {
                'header': chair_block_header,
                'body': {
                    'cluster_states': chair_result['cluster_states'],
                    'transactions': chair_result['transactions']
                },
                'hash': chair_block_hash,
                'round': round_num
            }

            # 上链
            blockchain.append(new_block)
            latest_block = new_block

            # 存储区块和模型
            chair_temp_state = member_results[committee_info['chair_id']]['temp_state']
            chain_storage.save_block(new_block, round_num)
            chain_storage.save_models(chair_temp_state.ball_models, round_num)

            # 从交易中提取有效更新的客户端
            for tx in chair_result['transactions']:
                if tx['type'] == 'aggregation':
                    valid_update_clients.extend(tx.get('valid_clients', []))

            print_complete()
            print_success("区块上链成功")

            # 步骤8：区块验证
            print_stage("区块验证")
            block_verification_impeachments = []

            for member_id in committee:
                # 使用validator验证正式区块
                block_validation_result = validator.validate_block(
                    new_block,
                    committee_info,
                    blockchain,
                    round_num
                )

                if not block_validation_result['valid']:
                    block_verification_impeachments.append({
                        'member_id': member_id,
                        'type': 'block_invalid'
                    })

            # 将区块验证弹劾加入总弹劾列表
            impeachment_votes.extend(block_verification_impeachments)

            print_complete()
            if block_verification_impeachments:
                print_warning(f"{len(block_verification_impeachments)} 个委员认为区块无效")

            # 步骤9: 弹劾判定
            impeached = impeachment_manager.check_impeachment(impeachment_votes, len(committee))

            if impeached:
                print_warning("委员长被弹劾，重组委员会")
                committee_info = committee_manager.handle_impeachment(client_ids)
            else:
                # 记录交易事件
                for tx in chair_result['transactions']:
                    if tx['type'] == 'split':
                        cfl_stats.ball_events.append({
                            'round': round_num,
                            'type': 'Split',
                            'details': tx
                        })

                        # 检查是否是首次分裂（阶段转换）
                        if phase == 'standard' and tx.get('source_ball_id', tx.get('ball_id')) == 0:
                            # 保存锚点模型
                            print_stage("保存锚点模型")
                            chain_storage.save_anchor_model(
                                system_state.get_ball_model(0),
                                round_num
                            )
                            print_complete()
                            print_info("锚点模型已保存，将在下一轮初始化合并依据")
                            merge_basis_initialization_pending = True

                    elif tx['type'] == 'merge':
                        cfl_stats.ball_events.append({
                            'round': round_num,
                            'type': 'Merge',
                            'details': tx
                        })

                # 步骤10: 更新系统状态
                system_state.update_from_temp(chair_temp_state)

                # 更新最大粒球ID
                GranularBall.id_manager.set_next_id(system_state.max_ball_id + 1)

                # 步骤11：客户端同步
                print_stage("客户端同步")

                for client in clients:
                    # 获取新的粒球归属
                    new_ball_id = system_state.get_client_ball_id(client.id)

                    if new_ball_id is not None and new_ball_id >= 0:
                        # 分配到粒球：同步该粒球模型
                        client.update_ball_assignment(new_ball_id)
                        ball_model = system_state.get_ball_model(new_ball_id)
                        if ball_model is not None:
                            client.synchronize_with_ball_model(ball_model)

                    elif new_ball_id == -1:
                        # 未分配客户端的处理
                        client.update_ball_assignment(-1)

                        # 检查是否是分裂产生的未分配客户端
                        for tx in new_block['body']['transactions']:
                            if tx['type'] == 'split' and client.id in tx.get('unassigned_clients', []):
                                # 从任意子粒球获取继承的父模型（所有子粒球都继承了父模型）
                                if 'child_balls' in tx and len(tx['child_balls']) > 0:
                                    # 获取第一个子粒球的模型
                                    first_child = tx['child_balls'][0]
                                    first_child_id = first_child['ball_id']
                                    inherited_model = system_state.get_ball_model(first_child_id)

                                    if inherited_model:
                                        client.synchronize_with_ball_model(inherited_model)
                                        print(f"  客户端{client.id}（未分配）同步分裂前模型（来自子粒球{first_child_id}）")
                                    else:
                                        print_warning(f"  客户端{client.id}无法获取分裂前模型")
                                break

                print_complete()

                # 检查阶段转换
                if phase == 'standard' and len(system_state.balls) > 1:
                    phase = 'granular'
                    first_split_round = round_num
                    print_success(f"首次分裂发生在第 {round_num} 轮，进入粒球联邦学习阶段")
        else:
            print_warning("未达成共识，跳过本轮")

        # 步骤12: 信誉分更新
        print_stage("更新信誉分")

        # 准备信誉分更新的参数
        votes_for_reputation = valid_votes if consensus_reached else []

        reputation_manager.update_scores(
            update_messages,
            committee_info,
            votes_for_reputation,
            impeachment_votes,
            consensus_reached,
            args.blockchain_config['reward'],
            valid_update_clients
        )

        # 保存信誉分文件
        reputation_manager.save_reputation_scores(round_num)

        print_complete()

        # 步骤13: 评估与记录（改进：只评估诚实节点）
        print_stage("评估性能（诚实节点）")

        # 只评估诚实客户端的准确率
        acc_clients = []
        honest_acc_clients = []

        for client in clients:
            if client.id in honest_client_ids:
                # 诚实节点：进行评估
                accuracy = client.evaluate()
                acc_clients.append(accuracy)
                honest_acc_clients.append(accuracy)
            else:
                # 恶意节点：跳过评估，用NaN占位以保持数组形状一致
                acc_clients.append(float('nan'))

        # 计算诚实节点的平均准确率
        mean_acc = np.mean(honest_acc_clients) if honest_acc_clients else 0.0

        print_complete()
        print_info(f"诚实节点平均准确率: {mean_acc:.4f} (基于{len(honest_acc_clients)}个诚实节点)")
        print_info(f"粒球数量: {len(system_state.balls)}")
        print_info(f"未分配客户端: {len(system_state.unmatched_clients)}")

        # 记录到日志（传递诚实节点信息）
        cfl_stats.log({
            'rounds': round_num,
            'acc_clients': acc_clients,  # 保留完整数组（包含NaN）
            'honest_acc_clients': honest_acc_clients,  # 新增：只包含诚实节点准确率
            'honest_client_ids': list(honest_client_ids),  # 新增：诚实客户端ID列表
            'granular_balls': system_state.to_dict()['balls'],
            'unmatched_clients': len(system_state.unmatched_clients),
            'phase': phase
        })

        # 生成可视化图表
        if round_num % args.plot_interval == 0 or round_num == args.communication_rounds:
            from gbcfl.visualization.accuracy_plot import plot_accuracy_with_events

            events = []
            for event in cfl_stats.ball_events:
                events.append({
                    'round': event['round'],
                    'type': event['type']
                })

            plot_accuracy_with_events(
                cfl_stats,
                first_split_round,
                args.communication_rounds,
                args.output_dir,
                "accuracy_with_events.png",
                silent_export=True,
                events=events,
                use_honest_only=True,  # 新增参数：使用诚实节点准确率绘图
                generate_pdf=True,     # 同时生成PDF
                generate_eps=True      # 同时生成EPS
            )

        # 定期保存检查点
        if round_num % 10 == 0:
            chain_storage.save_checkpoint(system_state, round_num)

    # 训练完成
    print_title("训练完成")
    print_info(f"最终诚实节点准确率: {mean_acc:.4f}")
    print_info(f"最终粒球数: {len(system_state.balls)}")
    print_info(f"区块链长度: {len(blockchain)}")

    if first_split_round > 0:
        print_info(f"首次分裂轮次: {first_split_round}")

    # 保存最终结果
    chain_storage.save_final_results(system_state, blockchain, cfl_stats)

    return cfl_stats