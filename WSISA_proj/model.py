# model.py

import torch
import torch.nn as nn

class DeepConvSurv(nn.Module):
    def __init__(self, in_channels=3, dropout_rate=0.3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
 
        self.fc = nn.Sequential(
            nn.Linear(32 * 10 * 10, 128),  # 增加隐藏层的维度，从 32 到 128
            nn.BatchNorm1d(128),            # 扩大 batch normalization，防止梯度消失
            nn.ReLU(),                      # 使用 ReLU 激活函数，避免负值激活
            nn.Dropout(dropout_rate),       # Dropout，继续防止过拟合
            nn.Linear(128, 64),             # 第二层：进一步减小维度（有助于特征压缩）
            nn.ReLU(),                      # 使用 ReLU 激活函数
            nn.Linear(64, 1)                # 输出层：1 表示单一的风险值预测
        )



        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        risk = self.fc(x)
        return risk
