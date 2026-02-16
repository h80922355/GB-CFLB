"""
系统粒球状态管理
存储达成共识后的粒球状态和模型
"""
import torch
import copy
from typing import List, Dict, Any
from gbcfl.utils.device_utils import get_device

device = get_device()


class SystemState:
    """
    系统粒球状态管理器
    维护达成共识的粒球状态
    """

    def __init__(self):
        """初始化系统状态"""
        self.balls = []  # 系统粒球列表
        self.ball_models = {}  # 粒球模型参数 {ball_id: model_params}
        self.unmatched_clients = []  # 未分配客户端ID列表
        self.max_ball_id = 0  # 当前最大粒球ID

    def initialize(self, initial_ball, initial_model):
        """
        初始化系统状态
        参数:
            initial_ball: 初始粒球(包含所有客户端)
            initial_model: 初始全局模型
        """
        self.balls = [initial_ball]
        self.ball_models = {
            initial_ball.ball_id: {
                key: value.clone().detach().to(device)  # 修正：添加detach()
                for key, value in initial_model.items()
            }
        }
        self.unmatched_clients = []
        self.max_ball_id = initial_ball.ball_id

    def update_from_temp(self, temp_state):
        """
        从临时状态更新系统状态(共识达成后)
        参数:
            temp_state: 临时状态对象
        """
        # 深拷贝临时状态到系统状态
        self.balls = [ball.clone() for ball in temp_state.balls]

        # 深拷贝模型参数
        self.ball_models = {}
        for ball_id, model_params in temp_state.ball_models.items():
            self.ball_models[ball_id] = {
                key: value.clone().detach().to(device)  # 修正：添加detach()
                for key, value in model_params.items()
            }

        self.unmatched_clients = temp_state.unmatched_clients.copy()
        self.max_ball_id = temp_state.max_ball_id

    def get_ball_by_id(self, ball_id):
        """
        根据ID获取粒球
        参数:
            ball_id: 粒球ID
        返回:
            粒球对象或None
        """
        for ball in self.balls:
            if ball.ball_id == ball_id:
                return ball
        return None

    def get_ball_model(self, ball_id):
        """
        获取粒球模型参数
        参数:
            ball_id: 粒球ID
        返回:
            模型参数字典或None
        """
        return self.ball_models.get(ball_id, None)

    def get_client_ball_id(self, client_id):
        """
        获取客户端所属粒球ID
        参数:
            client_id: 客户端ID
        返回:
            粒球ID或-1(未分配)
        """
        for ball in self.balls:
            if client_id in ball.client_ids:
                return ball.ball_id

        if client_id in self.unmatched_clients:
            return -1

        return None  # 客户端不存在

    def to_dict(self):
        """转换为字典格式用于存储"""
        return {
            'balls': [ball.to_dict() for ball in self.balls],
            'unmatched_clients': self.unmatched_clients,
            'max_ball_id': self.max_ball_id,
            'num_balls': len(self.balls)
        }