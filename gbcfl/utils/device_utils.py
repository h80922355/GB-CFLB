"""
设备管理工具模块
用于CUDA设备检测、配置和信息输出
针对RTX 4070和CUDA 12.1优化
"""
import torch
import platform
import psutil
import subprocess
import os
from typing import Dict, Any, Optional, Tuple
from colorama import Fore, Style


class DeviceManager:
    """设备管理器，负责CUDA设备检测和配置"""

    def __init__(self):
        self.device_info = self._detect_device_info()
        self.device = self._select_optimal_device()

    def _detect_device_info(self) -> Dict[str, Any]:
        """检测设备信息"""
        info = {
            'system': platform.system(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': None,
            'cudnn_version': None,
            'gpu_count': 0,
            'gpu_devices': [],
            'cpu_info': self._get_cpu_info(),
            'memory_info': self._get_memory_info()
        }

        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['cudnn_version'] = torch.backends.cudnn.version()
            info['gpu_count'] = torch.cuda.device_count()

            # 获取每个GPU的详细信息
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_info = {
                    'id': i,
                    'name': gpu_props.name,
                    'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                    'total_memory': gpu_props.total_memory / (1024 ** 3),  # GB
                    'multi_processor_count': gpu_props.multi_processor_count,
                    'is_integrated': self._is_integrated_gpu(gpu_props.name),
                    'memory_usage': self._get_gpu_memory_usage(i)
                }
                info['gpu_devices'].append(gpu_info)

        return info

    def _get_cpu_info(self) -> Dict[str, Any]:
        """获取CPU信息"""
        return {
            'processor': platform.processor(),
            'architecture': platform.architecture()[0],
            'logical_cores': psutil.cpu_count(logical=True),
            'physical_cores': psutil.cpu_count(logical=False),
            'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown'
        }

    def _get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024 ** 3),  # GB
            'available': memory.available / (1024 ** 3),  # GB
            'used_percent': memory.percent
        }

    def _get_gpu_memory_usage(self, device_id: int) -> Dict[str, float]:
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)  # GB
            return {
                'allocated': allocated,
                'reserved': reserved,
                'free': torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3) - reserved
            }
        return {'allocated': 0, 'reserved': 0, 'free': 0}

    def _is_integrated_gpu(self, gpu_name: str) -> bool:
        """判断是否为集成显卡"""
        integrated_keywords = ['Intel', 'AMD Radeon', 'Iris', 'UHD', 'Vega']
        return any(keyword in gpu_name for keyword in integrated_keywords)

    def _select_optimal_device(self) -> torch.device:
        """选择最优计算设备"""
        if not torch.cuda.is_available():
            return torch.device('cpu')

        # 选择显存最大的独立显卡
        best_device_id = 0
        max_memory = 0

        for gpu in self.device_info['gpu_devices']:
            if not gpu['is_integrated'] and gpu['total_memory'] > max_memory:
                max_memory = gpu['total_memory']
                best_device_id = gpu['id']

        return torch.device(f'cuda:{best_device_id}')

    def print_device_info(self):
        """输出设备信息"""
        print(f"\n{Fore.CYAN}{'=' * 60}")
        print(f"{Style.BRIGHT}GB-CFLB 设备信息检测{Style.RESET_ALL}")
        print(f"{'=' * 60}{Style.RESET_ALL}")

        # 系统信息
        print(f"\n{Fore.YELLOW}系统信息:{Style.RESET_ALL}")
        print(f"  操作系统: {self.device_info['system']}")
        print(f"  Python版本: {self.device_info['python_version']}")
        print(f"  PyTorch版本: {self.device_info['pytorch_version']}")

        # CPU信息
        cpu = self.device_info['cpu_info']
        print(f"\n{Fore.YELLOW}CPU信息:{Style.RESET_ALL}")
        print(f"  处理器: {cpu['processor']}")
        print(f"  架构: {cpu['architecture']}")
        print(f"  物理核心: {cpu['physical_cores']}")
        print(f"  逻辑核心: {cpu['logical_cores']}")
        print(f"  频率: {cpu['frequency']} MHz")

        # 内存信息
        memory = self.device_info['memory_info']
        print(f"\n{Fore.YELLOW}内存信息:{Style.RESET_ALL}")
        print(f"  总内存: {memory['total']:.2f} GB")
        print(f"  可用内存: {memory['available']:.2f} GB")
        print(f"  使用率: {memory['used_percent']:.1f}%")

        # CUDA信息
        print(f"\n{Fore.YELLOW}CUDA信息:{Style.RESET_ALL}")
        if self.device_info['cuda_available']:
            print(f"  {Fore.GREEN}✓ CUDA可用{Style.RESET_ALL}")
            print(f"  CUDA版本: {self.device_info['cuda_version']}")
            print(f"  cuDNN版本: {self.device_info['cudnn_version']}")
            print(f"  GPU数量: {self.device_info['gpu_count']}")
        else:
            print(f"  {Fore.RED}✗ CUDA不可用{Style.RESET_ALL}")

        # GPU详细信息
        if self.device_info['gpu_devices']:
            print(f"\n{Fore.YELLOW}GPU设备信息:{Style.RESET_ALL}")
            for gpu in self.device_info['gpu_devices']:
                gpu_type = "集成显卡" if gpu['is_integrated'] else "独立显卡"
                selected = "← 已选择" if f"cuda:{gpu['id']}" == str(self.device) else ""

                print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu_type}) {Fore.GREEN}{selected}{Style.RESET_ALL}")
                print(f"    计算能力: {gpu['compute_capability']}")
                print(f"    总显存: {gpu['total_memory']:.2f} GB")
                print(f"    已分配显存: {gpu['memory_usage']['allocated']:.2f} GB")
                print(f"    可用显存: {gpu['memory_usage']['free']:.2f} GB")
                print(f"    多处理器数量: {gpu['multi_processor_count']}")

        # 选中的设备
        print(f"\n{Fore.YELLOW}训练设备:{Style.RESET_ALL}")
        if self.device.type == 'cuda':
            gpu = self.device_info['gpu_devices'][self.device.index]
            gpu_type = "集成显卡" if gpu['is_integrated'] else "独立显卡"
            print(f"  {Fore.GREEN}✓ 使用GPU: {gpu['name']} ({gpu_type}){Style.RESET_ALL}")
            print(f"    设备ID: {self.device}")
            print(f"    显存: {gpu['total_memory']:.2f} GB")
        else:
            print(f"  {Fore.YELLOW}使用CPU训练{Style.RESET_ALL}")

        print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")

    def get_device(self) -> torch.device:
        """获取选定的计算设备"""
        return self.device

    def get_device_name(self) -> str:
        """获取设备名称"""
        if self.device.type == 'cuda':
            gpu = self.device_info['gpu_devices'][self.device.index]
            return gpu['name']
        return "CPU"

    def optimize_cuda_settings(self):
        """优化CUDA设置"""
        if torch.cuda.is_available():
            # 启用cuDNN自动调优
            torch.backends.cudnn.benchmark = True

            # 为RTX 4070优化设置
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # 设置内存增长策略
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                # 为RTX 4070 8GB显存设置80%使用率
                torch.cuda.set_per_process_memory_fraction(0.8)

            print(f"{Fore.GREEN}✓ CUDA优化设置已启用{Style.RESET_ALL}")

    def clear_cuda_cache(self):
        """清理CUDA缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# 全局设备管理器实例
_device_manager = None


def get_device_manager() -> DeviceManager:
    """获取全局设备管理器实例"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


def get_device() -> torch.device:
    """获取推荐的计算设备"""
    return get_device_manager().get_device()


def print_device_info():
    """打印设备信息"""
    get_device_manager().print_device_info()


def optimize_cuda():
    """优化CUDA设置"""
    get_device_manager().optimize_cuda_settings()


def clear_cuda_cache():
    """清理CUDA缓存"""
    get_device_manager().clear_cuda_cache()