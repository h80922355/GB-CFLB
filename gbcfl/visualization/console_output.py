"""
控制台输出格式化模块
负责联邦学习相关的控制台输出
修改版：移除半径相关的输出内容，增加优化种子选择的输出
"""
from colorama import Fore, Style, init

# 初始化colorama
init(autoreset=True)


def print_title(text):
    """
    打印标题文本

    参数:
        text: 要打印的标题文本
    """
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")


def print_clusters(clusters, balls=None):
    """
    打印当前聚类信息

    参数:
        clusters: 客户端聚类列表
        balls: 粒球对象列表（可选），如果提供，将使用粒球实际ID而不是索引
    """
    print(f"{Fore.YELLOW}当前聚类配置:{Style.RESET_ALL}")

    if balls is not None:
        # 使用粒球对象的真实ID
        for i, (cluster, ball) in enumerate(zip(clusters, balls)):
            client_ids = [client.id for client in cluster]
            print(f"  {Fore.GREEN}• 粒球 {ball.ball_id}:{Style.RESET_ALL} 包含 {len(cluster)} 个客户端")
            if len(cluster) <= 10:
                print(f"    客户端: {client_ids}")
            else:
                print(
                    f"    客户端: [{client_ids[0]}, {client_ids[1]}, ..., {client_ids[-2]}, {client_ids[-1]}] (共{len(cluster)}个)")
    else:
        # 原有逻辑，使用索引作为粒球ID
        for idx, cluster in enumerate(clusters):
            client_ids = [client.id for client in cluster]
            print(f"  {Fore.GREEN}• 粒球 {idx}:{Style.RESET_ALL} 包含 {len(cluster)} 个客户端")
            if len(cluster) <= 10:
                print(f"    客户端: {client_ids}")
            else:
                print(
                    f"    客户端: [{client_ids[0]}, {client_ids[1]}, ..., {client_ids[-2]}, {client_ids[-1]}] (共{len(cluster)}个)")

    print()


def print_system_clustering(balls, unmatched_clients=None):
    """
    打印当前系统聚类状态

    参数:
        balls: 粒球列表
        unmatched_clients: 未匹配客户端列表
    """
    print(f"{Fore.YELLOW}当前聚类配置:{Style.RESET_ALL}")

    # 打印每个粒球的详细信息
    for ball in balls:
        client_ids = [client.id for client in ball.clients]

        if len(client_ids) <= 10:
            client_list = ", ".join(map(str, client_ids))
            print(f"  {Fore.GREEN}• 粒球 {ball.ball_id}:{Style.RESET_ALL} 包含 {len(ball.clients)} 个客户端")
            print(f"    客户端: [{client_list}]")
        else:
            client_list = ", ".join(map(str, client_ids[:5])) + ", ..., " + ", ".join(map(str, client_ids[-2:]))
            print(f"  {Fore.GREEN}• 粒球 {ball.ball_id}:{Style.RESET_ALL} 包含 {len(ball.clients)} 个客户端")
            print(f"    客户端: [{client_list}] (共{len(client_ids)}个)")

    # 打印未匹配客户端
    if unmatched_clients and len(unmatched_clients) > 0:
        unmatched_ids = [client.id for client in unmatched_clients]

        if len(unmatched_ids) <= 10:
            unmatched_list = ", ".join(map(str, unmatched_ids))
        else:
            unmatched_list = ", ".join(map(str, unmatched_ids[:5])) + ", ..., " + ", ".join(map(str, unmatched_ids[-2:]))

        print(f"  {Fore.YELLOW}• 未匹配客户端:{Style.RESET_ALL} [{unmatched_list}] (共{len(unmatched_ids)}个)")


def print_result(accuracy, mode="训练"):
    """
    打印当前轮次的结果

    参数:
        accuracy: 准确率列表
        mode: 模式字符串，"训练"或"测试"
    """
    import numpy as np
    avg_acc = np.mean(accuracy) * 100
    print(f"{Fore.BLUE}本轮{mode}结果:{Style.RESET_ALL}")
    print(f"  {Fore.MAGENTA}• 平均准确率:{Style.RESET_ALL} {avg_acc:.2f}%")


def print_ball_metrics(balls):
    """
    打印所有粒球的指标

    参数:
        balls: 粒球列表
    """
    print(f"{Fore.YELLOW}粒球指标:{Style.RESET_ALL}")
    for ball in balls:
        center_movement = ball.get_center_movement() if hasattr(ball, 'prev_center') and ball.prev_center is not None else float('inf')
        purity = ball.purity

        # 计算收敛指标
        convergence_indicator, is_calculable = ball.calculate_convergence_indicator(window_size=3, min_history=3)
        convergence_value = convergence_indicator if is_calculable else float('inf')

        print(
            f"  {Fore.GREEN}• 粒球 {ball.ball_id}:{Style.RESET_ALL} 客户端数量={len(ball.clients)}, 中心移动={center_movement:.4f}, 纯度={purity:.4f}, 收敛指标={convergence_value:.4f}")


def print_stage(text):
    """
    打印阶段信息

    参数:
        text: 阶段描述文本
    """
    print(f"{Fore.CYAN}{text}...{Style.RESET_ALL}", end="", flush=True)


def print_complete():
    """
    打印完成标记
    """
    print(f"{Fore.GREEN}✓ 完成{Style.RESET_ALL}")


def print_warning(text):
    """
    打印警告信息

    参数:
        text: 警告文本
    """
    print(f"{Fore.YELLOW}警告: {text}{Style.RESET_ALL}")


def print_error(text):
    """
    打印错误信息

    参数:
        text: 错误文本
    """
    print(f"{Fore.RED}错误: {text}{Style.RESET_ALL}")


def print_info(text):
    """
    打印信息

    参数:
        text: 信息文本
    """
    print(f"{Fore.BLUE}{text}{Style.RESET_ALL}")


def print_success(text):
    """
    打印成功信息

    参数:
        text: 成功文本
    """
    print(f"{Fore.GREEN}{text}{Style.RESET_ALL}")


def print_round_result(c_round, mean_acc, blockchain_height):
    """
    打印轮次结果

    参数:
        c_round: 当前轮次
        mean_acc: 平均准确率
        blockchain_height: 区块链高度
    """
    print(f"{Fore.BLUE}本轮训练结果:{Style.RESET_ALL}")
    print(f"  {Fore.MAGENTA}• 平均准确率:{Style.RESET_ALL} {mean_acc*100:.2f}%")
    print(f"  {Fore.MAGENTA}• 区块链高度:{Style.RESET_ALL} {blockchain_height}")


def print_split_analysis(ball, split_details):
    """
    打印粒球分裂分析结果

    参数:
        ball: 粒球对象
        split_details: 分裂分析详情
    """
    print(f"{Fore.YELLOW}• 粒球 {ball.ball_id} 分裂条件分析:{Style.RESET_ALL}")

    convergence = split_details.get('convergence_indicator', float('inf'))
    is_stable = split_details.get('is_stable', False)
    is_heterogeneous = split_details.get('is_heterogeneous', False)
    has_enough_clients = split_details.get('has_enough_clients', False)

    # 新增：窗口期平均纯度信息
    average_purity = split_details.get('average_purity', 0.0)
    purity_calculable = split_details.get('purity_calculable', False)

    print(f"      收敛指标 = {Fore.CYAN}{convergence:.4f}{Style.RESET_ALL}, 稳定性条件: {Fore.GREEN if is_stable else Fore.RED}{'√' if is_stable else '×'}{Style.RESET_ALL}")

    if purity_calculable:
        print(f"      当前纯度 = {Fore.CYAN}{ball.purity:.4f}{Style.RESET_ALL}, 窗口期平均纯度 = {Fore.CYAN}{average_purity:.4f}{Style.RESET_ALL}, 异质性条件: {Fore.GREEN if is_heterogeneous else Fore.RED}{'√' if is_heterogeneous else '×'}{Style.RESET_ALL}")
    else:
        print(f"      粒球纯度 = {Fore.CYAN}{ball.purity:.4f}{Style.RESET_ALL}, 异质性条件: {Fore.GREEN if is_heterogeneous else Fore.RED}{'√' if is_heterogeneous else '×'}{Style.RESET_ALL}")

    print(f"      客户端数量 = {Fore.CYAN}{len(ball.clients)}{Style.RESET_ALL}, 数量条件: {Fore.GREEN if has_enough_clients else Fore.RED}{'√' if has_enough_clients else '×'}{Style.RESET_ALL}")

    all_conditions_met = is_stable and is_heterogeneous and has_enough_clients
    print(f"      最终判定: {Fore.GREEN if all_conditions_met else Fore.RED}{'√ 满足分裂条件' if all_conditions_met else '× 不满足分裂条件'}{Style.RESET_ALL}")


def print_split_seeds(ball, seed_i, seed_j):
    """
    打印基于四分位点的分裂种子信息

    参数:
        ball: 粒球对象
        seed_i: 种子1的索引（Q1位置）
        seed_j: 种子2的索引（Q3位置）
    """
    seed1_id = ball.clients[seed_i].id
    seed2_id = ball.clients[seed_j].id

    total_clients = len(ball.clients)

    # 计算实际的四分位点位置
    q1_theoretical = int(total_clients * 0.25)
    q3_theoretical = int(total_clients * 0.75)

    print(" ")
    print(f"        种子选择:")
    print(f"        总客户端数: {total_clients}")
    print(f"        Q1位置(25%): 理论索引{q1_theoretical} → 客户端 {seed1_id}(索引{seed_i})")
    print(f"        Q3位置(75%): 理论索引{q3_theoretical} → 客户端 {seed2_id}(索引{seed_j})")
    print(
        f"        选中分裂种子: {Fore.GREEN}低相似度种子 {seed1_id}{Style.RESET_ALL} 和 {Fore.GREEN}高相似度种子 {seed2_id}{Style.RESET_ALL}")


def print_split_seeds_optimized(ball, seed_i, seed_j, seed_selection_range_1,
                                seed_selection_range_2, seed_candidates_per_range):
    """
    打印优化种子选择策略的分裂种子信息

    参数:
        ball: 粒球对象
        seed_i: 种子1的索引
        seed_j: 种子2的索引
        seed_selection_range_1: 第一个种子选择范围
        seed_selection_range_2: 第二个种子选择范围
        seed_candidates_per_range: 每个范围内候选种子数量
    """
    seed1_id = ball.clients[seed_i].id
    seed2_id = ball.clients[seed_j].id

    # 计算实际的范围索引以便显示
    import math
    total_clients = len(ball.clients)
    range1_lower = int(math.floor(total_clients * seed_selection_range_1[0]))
    range1_upper = int(math.ceil(total_clients * seed_selection_range_1[1]))
    range2_lower = int(math.floor(total_clients * seed_selection_range_2[0]))
    range2_upper = int(math.ceil(total_clients * seed_selection_range_2[1]))

    # 计算每个范围内的实际客户端数量
    range1_size = max(0, range1_upper - range1_lower)
    range2_size = max(0, range2_upper - range2_lower)

    # 计算实际选择的候选数量
    actual_candidates_1 = min(seed_candidates_per_range, range1_size) if seed_candidates_per_range > 1 else (1 if range1_size > 0 else 0)
    actual_candidates_2 = min(seed_candidates_per_range, range2_size) if seed_candidates_per_range > 1 else (1 if range2_size > 0 else 0)

    print(f"      优化种子选择策略:")
    print(f"        总客户端数: {total_clients}")
    print(f"        范围1: {seed_selection_range_1} → 索引[{range1_lower}, {range1_upper}), 可选{range1_size}个, 实选{actual_candidates_1}个")
    print(f"        范围2: {seed_selection_range_2} → 索引[{range2_lower}, {range2_upper}), 可选{range2_size}个, 实选{actual_candidates_2}个")
    print(f"        最大候选数/范围: {seed_candidates_per_range}")
    print(f"        选中分裂种子: {Fore.GREEN}客户端 {seed1_id}(索引{seed_i}){Style.RESET_ALL} 和 {Fore.GREEN}客户端 {seed2_id}(索引{seed_j}){Style.RESET_ALL}")


def print_merge_analysis(ball1, ball2, merge_details):
    """
    打印粒球合并分析结果 - 移除了交叉相似度，增加了纯度窗口期信息

    参数:
        ball1: 第一个粒球对象
        ball2: 第二个粒球对象
        merge_details: 合并分析详情
    """
    print(f"{Fore.YELLOW}• 粒球 {ball1.ball_id} 与 粒球 {ball2.ball_id} 合并条件分析:{Style.RESET_ALL}")

    direction_similarity = merge_details.get('direction_similarity', 0.0)
    merge_benefit = merge_details.get('merge_benefit', 0.0)

    condition1 = merge_details.get('condition1', False)
    condition2 = merge_details.get('condition2', False)

    print(f"      方向相似度 = {Fore.CYAN}{direction_similarity:.4f}{Style.RESET_ALL}, 条件1: {Fore.GREEN if condition1 else Fore.RED}{'√' if condition1 else '×'}{Style.RESET_ALL}")
    print(f"      合并收益 = {Fore.CYAN}{merge_benefit:.4f}{Style.RESET_ALL}, 条件2: {Fore.GREEN if condition2 else Fore.RED}{'√' if condition2 else '×'}{Style.RESET_ALL}")

    all_conditions_met = condition1 and condition2
    print(f"      最终判定: {Fore.GREEN if all_conditions_met else Fore.RED}{'√ 满足合并条件' if all_conditions_met else '× 不满足合并条件'}{Style.RESET_ALL}")


def print_reassign_analysis(client, balls, similarity_scores, similarity_threshold=0.0):
    """
    打印客户端重分配分析结果
    修复：使用传入的相似度阈值而不是硬编码为0

    参数:
        client: 客户端对象
        balls: 粒球列表
        similarity_scores: 相似度得分字典，键为粒球ID，值为相似度
        similarity_threshold: 相似度阈值，从配置文件读取
    """
    print(f"{Fore.YELLOW}• 客户端 {client.id} 重分配判定:{Style.RESET_ALL}")

    # 按相似度排序
    sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    # 判断最佳匹配 - 修复：使用传入的相似度阈值
    if sorted_scores:
        best_ball_id, best_score = sorted_scores[0]
        match_result = best_score > similarity_threshold  # 修复：使用传入的阈值而不是硬编码为0
        print(f"      粒球 {best_ball_id} 相似度: {Fore.CYAN}{best_score:.4f}{Style.RESET_ALL}")
        print(f"      分配结果: {Fore.GREEN if match_result else Fore.RED}{'√ 分配到粒球 ' + str(best_ball_id) if match_result else '× 未分配'}{Style.RESET_ALL}")
    else:
        print(f"      {Fore.RED}分配结果: × 未分配 (无相似度数据){Style.RESET_ALL}")