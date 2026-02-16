"""
客户端重分配操作模块
基于参数更新的重分配算法，不依赖客户端对象
改进：确保只重分配验证通过的客户端
"""
import torch
from gbcfl.utils.operations import flatten
from gbcfl.utils.device_utils import get_device

device = get_device()


def reassign_clients_with_params(unmatched_ids, client_updates, balls, similarity_threshold=0.0):
    """
    基于更新参数的客户端重分配算法
    改进：只重分配有更新参数的客户端（即验证通过的客户端）

    参数:
        unmatched_ids: 未匹配客户端ID列表（可能包含未通过验证的客户端）
        client_updates: 客户端更新参数字典 {client_id: update_params}（只包含验证通过的客户端）
        balls: 粒球列表
        similarity_threshold: 相似度阈值

    返回:
        assignments: 分配结果 {ball_id: [client_ids]}
    """
    assignments = {}

    # 过滤出有更新参数的未匹配客户端（即验证通过的）
    valid_unmatched_ids = [cid for cid in unmatched_ids if cid in client_updates]
    invalid_unmatched_ids = [cid for cid in unmatched_ids if cid not in client_updates]

    print(f"        开始重分配{len(unmatched_ids)}个未匹配客户端")
    print(f"        其中：{len(valid_unmatched_ids)}个验证通过，{len(invalid_unmatched_ids)}个验证未通过")

    if not valid_unmatched_ids:
        print(f"        没有验证通过的未匹配客户端，跳过重分配")
        return assignments

    # 按客户端ID顺序处理，确保确定性
    sorted_client_ids = sorted(valid_unmatched_ids)

    successful_count = 0
    low_similarity_count = 0

    for client_id in sorted_client_ids:
        # 这里不需要再检查client_updates，因为我们已经过滤了
        client_update = flatten(client_updates[client_id])

        # 计算与所有粒球的相似度
        best_ball_id = None
        best_similarity = -1

        for ball in balls:
            if ball.center is not None:
                sim = calculate_similarity(client_update, ball.center)
                if sim > best_similarity:
                    best_similarity = sim
                    if best_similarity > similarity_threshold:
                        best_ball_id = ball.ball_id

        if best_similarity > similarity_threshold and best_ball_id is not None:
            print(f"            客户端{client_id}分配至粒球{best_ball_id}:相似度{best_similarity:.4f}")
        else:
            print(f"            客户端{client_id}未能重分配，最高相似度{best_similarity:.4f}")

        # 分配到最佳匹配的粒球
        if best_ball_id is not None:
            if best_ball_id not in assignments:
                assignments[best_ball_id] = []
            assignments[best_ball_id].append(client_id)
            successful_count += 1
        else:
            low_similarity_count += 1

    print(f"        重分配结果:")
    print(f"          成功分配: {successful_count}个")
    print(f"          相似度不足: {low_similarity_count}个")
    print(f"          验证未通过(不参与重分配): {len(invalid_unmatched_ids)}个")

    return assignments


def calculate_similarity(update_tensor, center):
    """
    计算更新向量与中心的余弦相似度

    参数:
        update_tensor: 更新向量
        center: 中心向量

    返回:
        相似度值
    """
    if center is None:
        return 0.0

    center_norm = torch.norm(center)
    update_norm = torch.norm(update_tensor)

    if center_norm < 1e-8 or update_norm < 1e-8:
        return 0.0

    sim = torch.dot(center, update_tensor) / (center_norm * update_norm)
    return sim.item()


def calculate_best_match(client_id, client_update, balls):
    """
    计算客户端的最佳匹配粒球

    参数:
        client_id: 客户端ID
        client_update: 客户端更新参数
        balls: 粒球列表

    返回:
        (best_ball_id, best_similarity): 最佳匹配的粒球ID和相似度
    """
    best_ball_id = None
    best_similarity = -1.0

    client_tensor = flatten(client_update)

    for ball in balls:
        if ball.center is not None:
            sim = calculate_similarity(client_tensor, ball.center)
            if sim > best_similarity:
                best_similarity = sim
                best_ball_id = ball.ball_id

    return best_ball_id, best_similarity


def batch_reassign_with_validation(unmatched_clients_info, balls, similarity_threshold=0.0):
    """
    批量重分配客户端（包含验证信息）

    参数:
        unmatched_clients_info: 未匹配客户端信息列表
                                [{client_id, update_params, is_valid}, ...]
        balls: 粒球列表
        similarity_threshold: 相似度阈值

    返回:
        assignments: 分配结果 {ball_id: [client_ids]}
        unassigned_valid: 未能分配的有效客户端ID列表
        unassigned_invalid: 未通过验证的客户端ID列表
    """
    assignments = {}
    unassigned_valid = []
    unassigned_invalid = []

    for client_info in unmatched_clients_info:
        client_id = client_info['client_id']

        if not client_info.get('is_valid', False):
            unassigned_invalid.append(client_id)
            continue

        update_params = client_info.get('update_params')
        if update_params is None:
            unassigned_invalid.append(client_id)
            continue

        # 计算最佳匹配
        best_ball_id, best_similarity = calculate_best_match(
            client_id, update_params, balls
        )

        if best_similarity > similarity_threshold and best_ball_id is not None:
            if best_ball_id not in assignments:
                assignments[best_ball_id] = []
            assignments[best_ball_id].append(client_id)
        else:
            unassigned_valid.append(client_id)

    return assignments, unassigned_valid, unassigned_invalid