import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from GeometryCNN import GeometryCNN as CNN

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, dead):
        self.memory.append((state, action, reward, next_state, dead))

    def sample(self, batch_size):
        state, action, reward, next_state, dead = zip(*random.sample(self.memory, batch_size))
        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(dead, dtype=torch.float32) 
        )

    def __len__(self):
        return len(self.memory)


class DqnAgent:
    def __init__(self, 
                 in_channels=4,
                 num_actions=2,
                 learning_rate=1e-4,
                 gamma=0.99,
                 buffer_cap=50_000,
                 batch_size=32,
                 target_update_freq=1000):
        
        self.gamma = gamma
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.train_step_count = 0
        self.target_update_freq = target_update_freq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = CNN(in_channels=in_channels, num_actions=num_actions).to(self.device)
        self.target_net = CNN(in_channels=in_channels, num_actions=num_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(capacity=buffer_cap)
        self.loss_fn = nn.SmoothL1Loss()

    def choose_action(self, state):
        self.policy_net.reset_all_noise()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.policy_net(state_tensor)
            return q_vals.argmax(dim=1).item()
        
    def save_state_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state_tensor = state.to(self.device)
        action_tensor = action.to(self.device)
        reward_tensor = reward.to(self.device)
        next_state_tensor = next_state.to(self.device)
        done_tensor = done.to(self.device)

        q_vals = self.policy_net(state_tensor)
        state_action_vals = q_vals.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            self.target_net.reset_all_noise()
            next_q_vals = self.target_net(next_state_tensor).max(dim=1)[0]
            expected_state_action_vals = reward_tensor + (self.gamma * next_q_vals * (1.0 - done_tensor))

        loss = self.loss_fn(state_action_vals, expected_state_action_vals)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save_check_points(self, filepath: str):
        state_dict = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(state_dict, filepath)

    def load_checkpoints(self, filepath: str):
        check = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(check["policy_net"])
        self.target_net.load_state_dict(check["target_net"])
        self.optimizer.load_state_dict(check["optimizer"])