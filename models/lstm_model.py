"""
LSTM模型定义模块
基于论文中的LSTM架构实现
"""
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    LSTM模型类
    用于序列建模和分类任务
    """

    def __init__(self, input_size=784, hidden_size=256, num_layers=2, num_classes=10, dropout=0.2):
        """
        初始化LSTM模型

        参数:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类类别数
            dropout: dropout概率
        """
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # 初始化分类层
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入数据

        返回:
            分类预测结果
        """
        batch_size = x.size(0)

        # 处理不同输入格式
        if len(x.shape) == 4:  # 图像输入
            x = x.view(batch_size, 1, -1)
        elif len(x.shape) == 2:  # 展平输入
            x = x.unsqueeze(1)

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM前向传播
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 分类
        output = self.classifier(last_output)

        return output

    def get_feature_dim(self):
        """获取特征维度"""
        return self.hidden_size