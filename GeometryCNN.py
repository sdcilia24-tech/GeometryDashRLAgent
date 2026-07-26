import torch
import torch.nn as nn

class GeometryCNN(nn.Module):
    def __init__(self, in_channels=4, num_actions=2):
        super(GeometryCNN, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),          
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           
            nn.ReLU(),
            nn.Flatten()                                          
        )
        
        self.connected = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),                                
        )
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)
        
    def forward(self, x):
        features = self.conv_block(x)
        x = self.connected(features)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value
