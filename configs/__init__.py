"""
配置模块初始化文件
提供配置加载和管理功能
"""

from .loader import ConfigLoader

# 创建一个默认的配置加载器实例
_default_config = None


def get_config(config_path="configs/config.yaml"):
    """
    获取配置实例（单例模式）

    Args:
        config_path: 配置文件路径

    Returns:
        ConfigLoader实例
    """
    global _default_config
    if _default_config is None:
        _default_config = ConfigLoader(config_path)
    return _default_config


def load_config(config_path="configs/config.yaml"):
    """
    加载新的配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        新的ConfigLoader实例
    """
    return ConfigLoader(config_path)


__all__ = [
    'ConfigLoader',
    'get_config',
    'load_config'
]