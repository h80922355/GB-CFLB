"""
粒球分裂操作模块
基于参数更新的分裂算法，不依赖客户端对象
改进：确保种子选择和分配只在有效客户端中进行
"""
import torch
import numpy as np
from gbcfl.granular_ball.ball import GranularBall
from gbcfl.utils.operations import flatten
from gbcfl.utils.device_utils import get_device

device = get_device()


def should_split(ball, args, current_round=0):
    """
    判断粒球是否应该分裂

    分裂条件（必须同时满足）：
    1. 粒球纯度 < 纯度阈值（存在异质性）
    2. 收敛指标 >= 收敛阈值（模型稳定）
    3. 客户端数量 >= min_size（数量足够）

    参数:
        ball: 粒球对象
        args: 训练参数
        current_round: 当前轮次

    返回:
        (should_split, split_details): 是否分裂及详细信息
    """
    split_details = {
        'ball_id': ball.ball_id,
        'round': current_round,
        'convergence': None,
        'purity': None,
        'client_count': len(ball.client_ids),
        'conditions': {}
    }

    # 条件1: 纯度 < 阈值（存在异质性）
    average_purity, purity_valid = ball.get_average_purity(
        window_size=args.purity_window_size,
        min_history=1
    )

    split_details['average_purity'] = average_purity
    split_details['purity_valid'] = purity_valid

    if purity_valid:
        is_heterogeneous = average_purity < args.purity_threshold
    else:
        is_heterogeneous = ball.purity < args.purity_threshold

    split_details['is_heterogeneous'] = is_heterogeneous
    split_details['conditions']['heterogeneous'] = is_heterogeneous

    # 条件2: 收敛指标 >= 阈值（模型稳定）
    convergence_indicator, conv_valid = ball.calculate_convergence_indicator(
        window_size=args.convergence_window_size,
        min_history=args.convergence_window_size
    )

    split_details['convergence'] = convergence_indicator
    split_details['convergence_valid'] = conv_valid

    # 修正：收敛指标大于等于阈值表示稳定
    if conv_valid:
        is_stable = convergence_indicator >= args.convergence_threshold
    else:
        is_stable = False  # 历史数据不足，认为不稳定

    split_details['is_stable'] = is_stable
    split_details['conditions']['stable'] = is_stable

    # 条件3: 客户端数量 >= min_size
    has_enough_clients = len(ball.client_ids) >= args.min_size
    split_details['has_enough_clients'] = has_enough_clients
    split_details['conditions']['enough_clients'] = has_enough_clients

    # 所有条件都满足才能分裂
    can_split = is_heterogeneous and is_stable and has_enough_clients

    return can_split, split_details


def split_ball_with_params(ball, client_updates, min_size, similarity_threshold=0.0, current_round=0):
    """
    基于更新参数的粒球分裂算法
    改进：确保只使用有效客户端进行分裂

    参数:
        ball: 要分裂的粒球（只包含有效客户端ID）
        client_updates: 客户端更新参数字典 {client_id: {'params': update_params, 'data_amount': amount}}
                       注意：这个字典应该只包含有效客户端
        min_size: 最小粒球大小
        similarity_threshold: 相似度阈值
        current_round: 当前轮次

    返回:
        (split_result, unmatched_ids)
        split_result: 分裂结果粒球列表
        unmatched_ids: 未匹配客户端ID列表
    """
    # 验证输入：确保ball.client_ids中的客户端都在client_updates中
    valid_client_ids = [cid for cid in ball.client_ids if cid in client_updates]

    if len(valid_client_ids) != len(ball.client_ids):
        print(f"警告：粒球{ball.ball_id}包含{len(ball.client_ids)}个客户端，但只有{len(valid_client_ids)}个有效更新")
        # 更新粒球的客户端列表为有效客户端
        ball.client_ids = valid_client_ids

    # 检查客户端数量
    if len(ball.client_ids) < min_size * 2:
        print(f"粒球{ball.ball_id}有效客户端数量不足（{len(ball.client_ids)} < {min_size * 2}），无法分裂")
        return [ball], []

    # 选择种子客户端（只在有效客户端中选择）
    seed_id1, seed_id2 = find_seeds_with_params(ball, client_updates)

    if seed_id1 is None or seed_id2 is None:
        print(f"无法为粒球{ball.ball_id}选择有效的种子客户端")
        return [ball], []

    print(f"        选择种子客户端: {seed_id1}(低相似度) 和 {seed_id2}(高相似度)")

    # 创建两个新粒球
    ball1 = GranularBall([])
    ball1.client_ids = [seed_id1]
    ball2 = GranularBall([])
    ball2.client_ids = [seed_id2]

    # 初始化粒球中心（使用种子客户端的更新参数）
    if seed_id1 in client_updates:
        update_params = client_updates[seed_id1]
        if isinstance(update_params, dict) and 'params' in update_params:
            ball1.center = flatten(update_params['params'])
        else:
            ball1.center = flatten(update_params)

    if seed_id2 in client_updates:
        update_params = client_updates[seed_id2]
        if isinstance(update_params, dict) and 'params' in update_params:
            ball2.center = flatten(update_params['params'])
        else:
            ball2.center = flatten(update_params)

    # 记录操作轮次
    ball1.last_operation_round = current_round
    ball2.last_operation_round = current_round

    # 未匹配客户端列表
    unmatched_ids = []

    # 分配其他客户端（只分配有效客户端）
    for client_id in ball.client_ids:
        if client_id == seed_id1 or client_id == seed_id2:
            continue

        if client_id not in client_updates:
            # 这种情况不应该发生，因为我们已经过滤了
            print(f"警告：客户端{client_id}没有更新参数，标记为未匹配")
            unmatched_ids.append(client_id)
            continue

        # 获取客户端更新参数
        update_info = client_updates[client_id]
        if isinstance(update_info, dict) and 'params' in update_info:
            client_update = flatten(update_info['params'])
        else:
            client_update = flatten(update_info)

        # 计算与两个粒球中心的相似度
        sim1 = calculate_similarity(client_update, ball1.center)
        sim2 = calculate_similarity(client_update, ball2.center)

        # 分配到相似度更高的粒球
        if sim1 >= similarity_threshold and sim1 >= sim2:
            ball1.client_ids.append(client_id)
            # 动态更新粒球中心（使用加权平均）
            update_ball_center_with_params(ball1, client_updates)
        elif sim2 >= similarity_threshold and sim2 > sim1:
            ball2.client_ids.append(client_id)
            # 动态更新粒球中心（使用加权平均）
            update_ball_center_with_params(ball2, client_updates)
        else:
            unmatched_ids.append(client_id)

    # 检查分裂结果
    result_balls = []
    if len(ball1.client_ids) >= min_size:
        result_balls.append(ball1)
    else:
        # 将ball1的客户端标记为未匹配
        unmatched_ids.extend(ball1.client_ids)

    if len(ball2.client_ids) >= min_size:
        result_balls.append(ball2)
    else:
        # 将ball2的客户端标记为未匹配
        unmatched_ids.extend(ball2.client_ids)

    # 如果分裂失败，返回原球
    if len(result_balls) == 0:
        print(f"分裂失败：无法形成足够大的新粒球")
        return [ball], []

    print(f"        分裂结果: {len(result_balls)}个新粒球, {len(unmatched_ids)}个未分配客户端")

    return result_balls, unmatched_ids


def find_seeds_with_params(ball, client_updates):
    """
    基于更新参数选择种子客户端
    使用15%和85%分位点策略
    改进：只在有效客户端中选择种子

    参数:
        ball: 粒球对象（client_ids应该只包含有效客户端）
        client_updates: 客户端更新参数字典（应该只包含有效客户端）

    返回:
        (seed_id1, seed_id2): 两个种子客户端ID
    """
    # 首先验证：确保ball.client_ids中的所有客户端都有更新参数
    valid_client_ids = [cid for cid in ball.client_ids if cid in client_updates]

    if len(valid_client_ids) < 2:
        print(f"警告：有效客户端不足2个，无法选择种子")
        return None, None

    if ball.center is None:
        # 如果粒球没有中心，选择第一个和最后一个有效客户端
        return valid_client_ids[0], valid_client_ids[-1]

    # 计算每个有效客户端与粒球中心的相似度
    similarities = []
    for client_id in valid_client_ids:
        if client_id in client_updates:
            update_info = client_updates[client_id]
            if isinstance(update_info, dict) and 'params' in update_info:
                client_update = flatten(update_info['params'])
            else:
                client_update = flatten(update_info)
            sim = calculate_similarity(client_update, ball.center)
            similarities.append((client_id, sim))

    if len(similarities) < 2:
        print(f"警告：无法计算足够的相似度，使用默认种子")
        return valid_client_ids[0] if len(valid_client_ids) > 0 else None, \
               valid_client_ids[-1] if len(valid_client_ids) > 1 else None

    # 按相似度排序
    similarities.sort(key=lambda x: x[1])

    # 选择15%和85%分位点
    q1_idx = max(0, int(len(similarities) * 0.15))
    q3_idx = min(len(similarities) - 1, int(len(similarities) * 0.85))

    # 确保两个种子不同
    if q1_idx == q3_idx:
        q3_idx = min(q1_idx + 1, len(similarities) - 1)
        if q3_idx == q1_idx:  # 只有一个客户端的极端情况
            q1_idx = max(0, q3_idx - 1)

    return similarities[q1_idx][0], similarities[q3_idx][0]


def calculate_similarity(update_tensor, center):
    """
    计算更新向量与中心的余弦相似度

    参数:
        update_tensor: 更新向量
        center: 中心向量

    返回:
        相似度值 [-1, 1]
    """
    if center is None:
        return 0.0

    center_norm = torch.norm(center)
    update_norm = torch.norm(update_tensor)

    if center_norm < 1e-8 or update_norm < 1e-8:
        return 0.0

    sim = torch.dot(center, update_tensor) / (center_norm * update_norm)
    return sim.item()


def update_ball_center_with_params(ball, client_updates):
    """
    基于客户端更新参数动态更新粒球中心
    改进：使用数据量加权平均而不是简单均值

    参数:
        ball: 粒球对象
        client_updates: 客户端更新参数字典，格式可以是：
                       {client_id: update_params} 或
                       {client_id: {'params': update_params, 'data_amount': amount}}
    """
    if not ball.client_ids:
        ball.center = None
        return

    # 计算加权平均
    weighted_sum = None
    total_weight = 0

    for client_id in ball.client_ids:
        if client_id in client_updates:
            update_info = client_updates[client_id]

            # 提取更新参数和数据量
            if isinstance(update_info, dict) and 'params' in update_info:
                # 新格式：包含参数和数据量
                update_params = flatten(update_info['params'])
                data_amount = update_info.get('data_amount', 1)
            else:
                # 旧格式：直接是参数
                update_params = flatten(update_info)
                data_amount = 1  # 默认权重为1

            # 加权累加
            if weighted_sum is None:
                weighted_sum = torch.zeros_like(update_params).to(device)

            weighted_sum += update_params * data_amount
            total_weight += data_amount

    # 计算加权平均作为新的粒球中心
    if total_weight > 0:
        ball.center = weighted_sum / total_weight
    else:
        ball.center = None