"""
粒球合并操作模块
基于参数更新的三阶段合并算法，不依赖客户端对象
改进：确保只使用验证通过的客户端进行合并评估
"""
import torch
from gbcfl.granular_ball.ball import GranularBall
from gbcfl.utils.operations import flatten
from gbcfl.utils.device_utils import get_device

device = get_device()


def merge_balls_with_params(balls, balls_updates, args, current_round=0, chain_storage=None):
    """
    基于更新参数的三阶段粒球合并算法
    改进：只使用有效客户端进行合并评估

    三阶段合并策略：
    1. 筛选候选粒球：纯度>=阈值 且 收敛指标>=阈值
    2. 评估粒球对：模型中心相似度>=阈值
    3. 计算数据相似度：基于合并依据的相似度>=阈值，按相似度顺序执行合并

    参数:
        balls: 粒球列表
        balls_updates: 各粒球的客户端更新参数 {ball_id: {client_id: update_params}}
                      注意：只包含验证通过的客户端
        args: 训练参数
        current_round: 当前轮次
        chain_storage: 链下存储管理器（用于读取合并依据）

    返回:
        (has_merged, new_balls): 是否发生合并，新粒球列表
    """
    has_merged = False
    merged_ball_ids = set()  # 记录已经参与合并的粒球ID
    merge_operations = []  # 记录所有合并操作

    print(f"        开始合并判定（共{len(balls)}个粒球）")

    # 第一阶段：筛选候选粒球
    candidate_balls = []
    for ball in balls:
        print(f"          粒球{ball.ball_id}候选判定:")

        if ball.cooling_period > 0:
            print(f"            冷却期剩余{ball.cooling_period}轮 → 不可合并")
            continue

        # 检查粒球中是否有足够的有效客户端
        valid_client_count = len([cid for cid in ball.client_ids
                                 if ball.ball_id in balls_updates and
                                 cid in balls_updates[ball.ball_id]])

        if valid_client_count < args.min_size:
            print(f"            有效客户端不足({valid_client_count}个) → 不可合并")
            continue

        print(f"            纯度={ball.purity:.4f} (阈值={args.purity_threshold})")

        convergence, conv_valid = ball.calculate_convergence_indicator()
        if conv_valid:
            print(f"            收敛指标={convergence:.4f} (阈值={args.convergence_threshold})")
        else:
            print(f"            收敛指标=无效（历史数据不足）")

        if is_merge_candidate(ball, args):
            candidate_balls.append(ball)
            print(f"            → ✓ 满足候选条件")
        else:
            print(f"            → ✗ 不满足候选条件")

    if len(candidate_balls) < 2:
        print(f"        第一阶段结果：候选粒球不足（需要至少2个，当前{len(candidate_balls)}个）")
        return False, balls

    print(f"        第一阶段结果：找到{len(candidate_balls)}个候选粒球")

    # 第二阶段：两两配对评估模型相似度（只基于有效客户端）
    print(f"        第二阶段：评估粒球对（模型相似度）")
    model_similar_pairs = []

    for i in range(len(candidate_balls)):
        for j in range(i + 1, len(candidate_balls)):
            ball1 = candidate_balls[i]
            ball2 = candidate_balls[j]

            print(f"          评估粒球对({ball1.ball_id}, {ball2.ball_id}):")

            # 使用只包含有效客户端的更新重新计算中心
            ball1_valid_updates = balls_updates.get(ball1.ball_id, {})
            ball2_valid_updates = balls_updates.get(ball2.ball_id, {})

            # 临时计算基于有效客户端的中心
            ball1_center = calculate_center_from_updates(ball1.client_ids, ball1_valid_updates)
            ball2_center = calculate_center_from_updates(ball2.client_ids, ball2_valid_updates)

            # 评估模型中心相似度
            similarity_satisfied, direction_similarity = evaluate_model_similarity_with_centers(
                ball1_center, ball2_center, args)
            print(f"            模型方向相似度={direction_similarity:.4f} (阈值={args.direction_threshold})")

            if similarity_satisfied:
                model_similar_pairs.append((ball1, ball2))
                print(f"            → ✓ 满足模型相似度条件")
            else:
                print(f"            → ✗ 不满足模型相似度条件")

    if not model_similar_pairs:
        print(f"        第二阶段结果：没有满足模型相似度条件的粒球对")
        return False, balls

    print(f"        第二阶段结果：找到{len(model_similar_pairs)}个模型相似的粒球对")

    # 第三阶段：计算数据相似度并按相似度顺序执行合并
    print(f"        第三阶段：计算数据相似度并执行合并")

    # 如果没有提供chain_storage，无法进行数据相似度判定
    if chain_storage is None:
        print(f"        警告：未提供链下存储，无法进行数据相似度判定")
        return False, balls

    # 从更新消息中提取数据量信息（只计算有效客户端）
    data_amounts = {}
    for ball_id, updates in balls_updates.items():
        for client_id, update in updates.items():
            if isinstance(update, dict) and 'data_amount' in update:
                data_amounts[client_id] = update['data_amount']
            else:
                data_amounts[client_id] = 1  # 默认数据量

    # 计算所有满足条件的粒球对的数据相似度
    qualified_pairs = []
    for ball1, ball2 in model_similar_pairs:
        print(f"          评估数据相似度({ball1.ball_id}, {ball2.ball_id}):")

        # 只使用有效客户端计算数据相似度
        ball1_valid_clients = [cid for cid in ball1.client_ids
                               if ball1.ball_id in balls_updates and
                               cid in balls_updates[ball1.ball_id]]
        ball2_valid_clients = [cid for cid in ball2.client_ids
                               if ball2.ball_id in balls_updates and
                               cid in balls_updates[ball2.ball_id]]

        # 计算数据相似度
        data_similarity = calculate_data_similarity_for_valid_clients(
            ball1_valid_clients, ball2_valid_clients, chain_storage, data_amounts
        )

        print(f"            数据相似度={data_similarity:.4f} (阈值={args.data_similarity_threshold})")

        # 检查是否满足数据相似度阈值
        if data_similarity >= args.data_similarity_threshold:
            qualified_pairs.append((ball1, ball2, data_similarity))
            print(f"            → ✓ 满足数据相似度条件")
        else:
            print(f"            → ✗ 不满足数据相似度条件")

    if not qualified_pairs:
        print(f"        第三阶段结果：没有粒球满足数据相似度条件")
        return False, balls

    # 按数据相似度降序排序
    qualified_pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"        第三阶段：找到{len(qualified_pairs)}个满足条件的粒球对，按相似度排序处理")

    # 按相似度顺序执行合并，确保每个粒球只参与一次合并
    for ball1, ball2, similarity in qualified_pairs:
        # 检查两个粒球是否都未参与过合并
        if ball1.ball_id not in merged_ball_ids and ball2.ball_id not in merged_ball_ids:
            print(f"        执行合并: 粒球{ball1.ball_id}与粒球{ball2.ball_id} (相似度={similarity:.4f})")

            # 获取更新参数（只包含有效客户端）
            updates1 = balls_updates.get(ball1.ball_id, {})
            updates2 = balls_updates.get(ball2.ball_id, {})

            # 执行合并（传递数据量信息）
            merged_ball = execute_merge(ball1, ball2, updates1, updates2, current_round, args, data_amounts)

            # 记录合并操作
            merge_operations.append({
                'merged_ball': merged_ball,
                'source_balls': [ball1.ball_id, ball2.ball_id],
                'similarity': similarity
            })

            # 标记已参与合并的粒球
            merged_ball_ids.add(ball1.ball_id)
            merged_ball_ids.add(ball2.ball_id)
            has_merged = True

            print(f"        合并完成：新粒球临时ID={merged_ball.ball_id}，包含客户端{merged_ball.client_ids}")
        else:
            # 至少有一个粒球已经参与过合并
            if ball1.ball_id in merged_ball_ids:
                print(f"        跳过合并({ball1.ball_id}, {ball2.ball_id}): 粒球{ball1.ball_id}已参与合并")
            if ball2.ball_id in merged_ball_ids:
                print(f"        跳过合并({ball1.ball_id}, {ball2.ball_id}): 粒球{ball2.ball_id}已参与合并")

    # 构建最终粒球列表
    new_balls = []

    # 添加未参与合并的原有粒球
    for ball in balls:
        if ball.ball_id not in merged_ball_ids:
            new_balls.append(ball)

    # 添加所有新合并的粒球
    for merge_op in merge_operations:
        new_balls.append(merge_op['merged_ball'])

    if has_merged:
        print(f"        第三阶段结果：成功执行{len(merge_operations)}对粒球合并")
        for i, merge_op in enumerate(merge_operations):
            print(f"          合并{i+1}: {merge_op['source_balls']} → 新粒球(待分配ID), 相似度={merge_op['similarity']:.4f}")
    else:
        print(f"        第三阶段结果：没有执行任何合并")

    return has_merged, new_balls


def is_merge_candidate(ball, args):
    """
    第一阶段：判断粒球是否可以作为合并候选

    候选条件（必须同时满足）：
    1. 纯度 >= 纯度阈值（内部同质性良好）
    2. 收敛指标 >= 收敛阈值（模型稳定）
    3. 不在冷却期内

    参数:
        ball: 粒球对象
        args: 训练参数

    返回:
        是否可以作为合并候选
    """
    # 检查冷却期
    if ball.cooling_period > 0:
        return False

    # 条件1：纯度 >= 阈值
    if ball.purity < args.purity_threshold:
        return False

    # 条件2：收敛指标 >= 阈值
    convergence, conv_valid = ball.calculate_convergence_indicator()
    if not conv_valid:
        return False  # 历史数据不足

    if convergence < args.convergence_threshold:
        return False  # 不够稳定

    return True


def evaluate_model_similarity_with_centers(center1, center2, args):
    """
    第二阶段：评估两个中心的模型相似度

    参数:
        center1, center2: 两个中心向量
        args: 训练参数

    返回:
        (是否满足相似度条件, 方向相似度值)
    """
    if center1 is None or center2 is None:
        return False, 0.0

    # 计算方向相似度（余弦相似度）
    center_norm1 = torch.norm(center1)
    center_norm2 = torch.norm(center2)

    if center_norm1 < 1e-8 or center_norm2 < 1e-8:
        return False, 0.0

    direction_similarity = torch.dot(center1, center2) / (center_norm1 * center_norm2)
    direction_similarity = direction_similarity.item()

    # 判断是否满足阈值
    return direction_similarity >= args.direction_threshold, direction_similarity


def evaluate_model_similarity_with_details(ball1, ball2, args):
    """
    保留原有接口的兼容性
    """
    if ball1.center is None or ball2.center is None:
        return False, 0.0

    return evaluate_model_similarity_with_centers(ball1.center, ball2.center, args)


def calculate_center_from_updates(client_ids, updates):
    """
    从客户端更新计算中心

    参数:
        client_ids: 客户端ID列表
        updates: 更新参数字典

    返回:
        中心向量
    """
    center_sum = None
    count = 0

    for cid in client_ids:
        if cid in updates:
            update = flatten(updates[cid])
            if center_sum is None:
                center_sum = torch.zeros_like(update).to(device)
            center_sum += update
            count += 1

    if count > 0:
        return center_sum / count
    return None


def calculate_data_similarity_for_valid_clients(valid_clients1, valid_clients2,
                                                chain_storage, data_amounts):
    """
    计算两组有效客户端的数据相似度
    基于客户端合并依据的加权平均

    参数:
        valid_clients1, valid_clients2: 两组有效客户端ID列表
        chain_storage: 链下存储管理器
        data_amounts: 客户端数据量字典

    返回:
        数据相似度值 [-1, 1]
    """
    # 获取两组有效客户端的合并依据
    ball1_basis = chain_storage.load_ball_merge_basis(valid_clients1, data_amounts)
    ball2_basis = chain_storage.load_ball_merge_basis(valid_clients2, data_amounts)

    if ball1_basis is None or ball2_basis is None:
        print(f"            警告：无法加载合并依据")
        return 0.0

    # 计算余弦相似度
    norm1 = torch.norm(ball1_basis)
    norm2 = torch.norm(ball2_basis)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    similarity = torch.dot(ball1_basis, ball2_basis) / (norm1 * norm2)
    return similarity.item()


def calculate_data_similarity(ball1, ball2, chain_storage, data_amounts):
    """
    保留原有接口的兼容性
    """
    return calculate_data_similarity_for_valid_clients(
        ball1.client_ids, ball2.client_ids, chain_storage, data_amounts
    )


def execute_merge(ball1, ball2, updates1, updates2, current_round, args, data_amounts):
    """
    执行粒球合并
    改进：只合并有效客户端

    参数:
        ball1, ball2: 要合并的两个粒球
        updates1, updates2: 对应的更新参数（只包含有效客户端）
        current_round: 当前轮次
        args: 训练参数
        data_amounts: 客户端数据量字典

    返回:
        合并后的新粒球
    """
    # 只合并有更新的客户端
    valid_clients1 = [cid for cid in ball1.client_ids if cid in updates1]
    valid_clients2 = [cid for cid in ball2.client_ids if cid in updates2]

    # 创建合并后的粒球
    merged_ball = GranularBall([])
    merged_ball.client_ids = valid_clients1 + valid_clients2

    # 计算合并后的中心（只基于有效客户端）
    merged_ball.center = calculate_merged_center(
        valid_clients1, updates1,
        valid_clients2, updates2
    )

    # 使用配置文件的冷却期参数
    merged_ball.cooling_period = args.cooling_period
    merged_ball.last_operation_round = current_round

    # 计算合并后的纯度（基于模型更新的相似度）
    merged_ball.purity = estimate_merged_purity(
        valid_clients1, updates1,
        valid_clients2, updates2
    )

    return merged_ball


def calculate_merged_center(client_ids1, updates1, client_ids2, updates2):
    """
    计算合并后的粒球中心

    参数:
        client_ids1, client_ids2: 两个粒球的客户端ID列表
        updates1, updates2: 对应的更新参数

    返回:
        合并后的中心向量
    """
    center_sum = None
    count = 0

    # 处理第一个粒球的客户端
    for cid in client_ids1:
        if cid in updates1:
            update = flatten(updates1[cid])
            if center_sum is None:
                center_sum = torch.zeros_like(update).to(device)
            center_sum += update
            count += 1

    # 处理第二个粒球的客户端
    for cid in client_ids2:
        if cid in updates2:
            update = flatten(updates2[cid])
            if center_sum is None:
                center_sum = torch.zeros_like(update).to(device)
            center_sum += update
            count += 1

    if count > 0:
        return center_sum / count
    return None


def estimate_merged_purity(client_ids1, updates1, client_ids2, updates2):
    """
    估计合并后的粒球纯度（基于模型更新）

    参数:
        client_ids1, client_ids2: 两个粒球的客户端ID列表
        updates1, updates2: 对应的更新参数

    返回:
        估计的合并后纯度
    """
    # 计算合并后的中心
    merged_center = calculate_merged_center(
        client_ids1, updates1,
        client_ids2, updates2
    )

    if merged_center is None:
        return 0.0

    # 计算所有客户端与合并中心的相似度
    similarities = []

    for cid in client_ids1:
        if cid in updates1:
            update = flatten(updates1[cid])
            sim = calculate_similarity(update, merged_center)
            similarities.append(sim)

    for cid in client_ids2:
        if cid in updates2:
            update = flatten(updates2[cid])
            sim = calculate_similarity(update, merged_center)
            similarities.append(sim)

    if similarities:
        return sum(similarities) / len(similarities)
    return 0.0


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


def get_merged_model_params(ball1_model, ball2_model, data_amount1, data_amount2):
    """
    计算合并后的模型参数（基于数据量加权平均）

    参数:
        ball1_model: 粒球1的模型参数
        ball2_model: 粒球2的模型参数
        data_amount1: 粒球1的总数据量
        data_amount2: 粒球2的总数据量

    返回:
        合并后的模型参数
    """
    if ball1_model is None or ball2_model is None:
        return ball1_model if ball1_model is not None else ball2_model

    # 基于数据量计算权重
    total_data = data_amount1 + data_amount2
    if total_data == 0:
        # 如果没有数据量信息，回退到均等权重
        print(f"            警告：数据量为0，使用均等权重")
        w1 = 0.5
        w2 = 0.5
    else:
        w1 = data_amount1 / total_data
        w2 = data_amount2 / total_data
        print(f"            模型合并权重：w1={w1:.3f} (数据量:{data_amount1}), w2={w2:.3f} (数据量:{data_amount2})")

    # 加权平均
    merged_model = {}
    for key in ball1_model.keys():
        if key in ball2_model:
            merged_model[key] = w1 * ball1_model[key] + w2 * ball2_model[key]
        else:
            merged_model[key] = ball1_model[key]

    return merged_model