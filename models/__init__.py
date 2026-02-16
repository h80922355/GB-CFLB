"""
模型模块初始化文件
提供统一的模型接口
"""
from .cnn_model import CNN, SimpleConvNet
from .lstm_model import LSTM
from .model_factory import ModelFactory

# 导出主要类
__all__ = [
    'CNN',
    'SimpleConvNet',  # 新增简易CNN导出
    'LSTM',
    'ModelFactory'
]

# 版本信息
__version__ = '2.0.0'

# 模型注册表（用于快速查找）
MODEL_REGISTRY = {
    'cnn': CNN,
    'simple_cnn': SimpleConvNet,
    'lstm': LSTM
}

def get_model_class(model_name):
    """
    根据名称获取模型类

    参数:
        model_name: 模型名称

    返回:
        模型类或None
    """
    return MODEL_REGISTRY.get(model_name.lower())

def list_available_models():
    """
    列出所有可用的模型

    返回:
        模型名称列表
    """
    return list(MODEL_REGISTRY.keys())