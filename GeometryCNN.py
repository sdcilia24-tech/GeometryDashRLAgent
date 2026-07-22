import torch
import torch.nn as nn
from torchrl.modules import NoisyLinear
class GeometryCNN(nn.Module):
    def __init__(self, in_channels = 4, num_actions = 2):
        super(GeometryCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = 32,
                kernel_size = 8,
                stride = 4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 4,
                stride = 2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = 64,
                out_channels = 64,
                kernel_size = 2,
                stride = 1
            ),
            nn.ReLU(),
            nn.Flatten()
        )
        self.connected = nn.Sequential(
            NoisyLinear(64 * 10 * 10, 512), 
            nn.ReLU(),
            NoisyLinear(512, num_actions)
        )
    def forward(self, x):
        return self.connected(self.conv_block(x))
