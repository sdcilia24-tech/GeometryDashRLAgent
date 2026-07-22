import torch
import torch.nn as nn
import torchrl.modules
from torchrl.modules import NoisyLinear
class GeometryCNN(nn.Module):
    def __init__(self, in_channels = 4, num_actions = 2, input_shape = (4, 100, 100)):
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
        conv_output_dimensions = self._get_output_shape(input_shape)

        self.connected = nn.Sequential(
            NoisyLinear(conv_output_dimensions, 512), 
            nn.ReLU(),
            NoisyLinear(512, num_actions)
        )

    def _get_output_shape(self, shape):
        """dynamically calculates the output shape of the flattened layer"""
        dummy_tensor = torch.zeros(1, *shape)
        out = self.conv_block(dummy_tensor)
        return out
    
    def forward(self, x):
        x = x.float() / 255.0
        return self.connected(self.conv_block(x))

    def reset_all_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()