import random
from collections import deque
import torch
import numpy as np
class ReplayBuffer:
    def __init__(self, capacity = 10000):
        self.memory = deque(maxlen = capacity)

    def push(self, state, action, reward, next_state, dead ):
        self.memory.append((state, action, reward, next_state, dead))
    def sample(self, batch_size):
        state, action, reward, next_state, dead = zip(*random.sample(self.memory, batch_size))
        return (
            torch.tensor(np.array(state), dtype = torch.float32),
            torch.tensor(action, dtype = torch.long),
            torch.tensor(reward, dtype = torch.float32),
            torch.tensor(np.array(next_state, dtype = torch.float32)),
            torch.tensor(dead, dtype = torch.bool)
        )
    def __len__(self):
        return len(self.memory)