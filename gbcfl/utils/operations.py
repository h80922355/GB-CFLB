"""
设备共享工具函数模块 - CUDA优化版（修复版）
包含客户端和服务器共享的基本操作函数
针对RTX 4070和CUDA 12.1优化
修复PyTorch新版本API弃用警告
"""
import torch
import torch.nn.functional as F
from .device_utils import get_device

# 获取全局设备
device = get_device()


def train_op(model, loader, optimizer, epochs=1):
    """
    执行模型的训练操作 - CUDA优化版（修复版）

    参数:
        model: 要训练的模型
        loader: 数据加载器
        optimizer: 优化器
        epochs: 训练轮数

    返回:
        平均训练损失和一个包含每批次损失值的列表
    """
    # 确保模型在正确设备上
    model = model.to(device)
    model.train()

    running_loss, samples = 0.0, 0
    batch_losses = []

    # 启用混合精度训练(RTX 4070支持)
    use_amp = device.type == 'cuda'

    # 修复：使用新的API
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    for ep in range(epochs):
        for batch_idx, (x, y) in enumerate(loader):
            # 数据移动到设备
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 使用混合精度 - 修复：使用新的API
            # 损失函数：交叉熵
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(x)
                    loss = F.cross_entropy(outputs, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(x)
                loss = F.cross_entropy(outputs, y)
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            batch_losses.append(batch_loss)

            running_loss += batch_loss * y.shape[0]
            samples += y.shape[0]

            # 定期清理CUDA缓存
            if use_amp and batch_idx % 50 == 0:
                torch.cuda.empty_cache()

    return running_loss / samples, batch_losses


def eval_op(model, loader):
    """
    在验证集上评估模型性能 - CUDA优化版（修复版）

    参数:
        model: 要评估的模型
        loader: 数据加载器

    返回:
        分类准确率
    """
    # 确保模型在正确设备上
    model = model.to(device)
    model.eval()

    samples, correct = 0, 0
    use_amp = device.type == 'cuda'

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            # 数据移动到设备
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # 使用混合精度推理 - 修复：使用新的API
            if use_amp:
                with torch.amp.autocast('cuda'):
                    y_ = model(x)
            else:
                y_ = model(x)

            _, predicted = torch.max(y_.data, 1)

            samples += y.shape[0]
            correct += (predicted == y).sum().item()

            # 定期清理CUDA缓存
            if use_amp and batch_idx % 50 == 0:
                torch.cuda.empty_cache()

    return correct / samples


def copy(target, source):
    """
    将 source 中的参数值拷贝到 target - 设备感知版

    参数:
        target: 目标参数字典
        source: 源参数字典
    """
    for name in target:
        # 确保数据在同一设备上
        if isinstance(source[name], torch.Tensor):
            target[name].data = source[name].data.clone().to(target[name].device)
        else:
            target[name].data = source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    """
    计算 minuend 和 subtrahend 参数的差值并存储到 target - 设备感知版

    参数:
        target: 目标参数字典，存储差值
        minuend: 被减数参数字典
        subtrahend: 减数参数字典
    """
    for name in target:
        # 确保所有tensor在同一设备上
        minuend_data = minuend[name].data.to(target[name].device)
        subtrahend_data = subtrahend[name].data.to(target[name].device)

        target[name].data = minuend_data.clone() - subtrahend_data.clone()


def reduce_weighted_average(targets, sources, weights):
    """
    对sources的name参数取加权平均值并累加到targets的name参数 - CUDA优化版

    在联邦学习中，用于服务器按客户端数据量加权聚合客户端更新

    参数:
        targets: 目标参数字典列表
        sources: 源参数字典列表
        weights: 权重列表，表示每个源的权重（例如数据量）
    """
    # 确保有源参数且总权重不为0
    if len(sources) == 0 or sum(weights) == 0:
        return

    # 计算权重归一化系数
    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]

    # 对每个目标执行加权更新
    for target in targets:
        for name in target:
            try:
                # 获取目标设备
                target_device = target[name].device

                # 使用目标设备进行计算
                with torch.cuda.device(target_device) if target_device.type == 'cuda' else torch.no_grad():
                    # 创建加权和tensor
                    weighted_sum = torch.zeros_like(target[name].data)

                    for i, source in enumerate(sources):
                        # 将源数据移动到目标设备
                        source_data = source[name].data.to(target_device)
                        weighted_sum += source_data * normalized_weights[i]

                    # 更新目标参数
                    target[name].data += weighted_sum

            except Exception as e:
                print(f"Error in reduce_weighted_average for parameter {name}: {e}")
                print(f"Source shapes: {[source[name].data.shape for source in sources]}")
                print(f"Target shape: {target[name].data.shape}")
                print(f"Target device: {target[name].device}")

                # 使用CPU备选计算方法
                cpu_target = target[name].data.cpu()
                weighted_sum = torch.zeros_like(cpu_target)

                for i, source in enumerate(sources):
                    cpu_source = source[name].data.cpu()
                    weighted_sum += cpu_source * normalized_weights[i]

                target[name].data += weighted_sum.to(target[name].device)


def flatten(source):
    """
    将模型参数展开为一维向量 - 设备感知版

    参数:
        source: 参数字典

    返回:
        展平后的参数向量
    """
    # 获取第一个参数的设备
    first_param = next(iter(source.values()))
    target_device = first_param.device if hasattr(first_param, 'device') else device

    # 将所有参数移动到同一设备并展平
    flattened_params = []
    for value in source.values():
        if hasattr(value, 'flatten'):
            flattened_params.append(value.flatten().to(target_device))
        else:
            flattened_params.append(torch.tensor(value).flatten().to(target_device))

    return torch.cat(flattened_params)


def move_to_device(data, target_device=None):
    """
    将数据移动到指定设备

    参数:
        data: 要移动的数据（tensor、dict或list）
        target_device: 目标设备，默认使用全局设备

    返回:
        移动后的数据
    """
    if target_device is None:
        target_device = device

    if isinstance(data, torch.Tensor):
        return data.to(target_device, non_blocking=True)
    elif isinstance(data, dict):
        return {k: move_to_device(v, target_device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, target_device) for item in data)
    else:
        return data


def optimize_memory():
    """
    优化内存使用
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_memory_usage():
    """
    获取当前内存使用情况

    返回:
        内存使用信息字典
    """
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
            'reserved': torch.cuda.memory_reserved() / (1024**3),    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / (1024**3)  # GB
        }
    else:
        import psutil
        memory = psutil.virtual_memory()
        return {
            'allocated': (memory.total - memory.available) / (1024**3),
            'reserved': memory.total / (1024**3),
            'max_allocated': memory.total / (1024**3)
        }


# 为了向后兼容，保持原有的device变量
# 但现在它是动态获取的
def get_current_device():
    """获取当前设备"""
    return device