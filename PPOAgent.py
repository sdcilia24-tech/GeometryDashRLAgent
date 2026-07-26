import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from GeometryCNN import GeometryCNN

class PPOAgent:
    def __init__(self, in_channels=4, num_actions=2, device="cpu", lr=3e-4, gamma=0.98, clip_eps=0.2):
        self.device = device
        self.gamma = gamma
        self.clip_eps = clip_eps
        
        self.model = GeometryCNN(in_channels, num_actions).to(self.device)
        
        dummy = torch.zeros(1, in_channels, 64, 64).to(self.device)
        self.model(dummy)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps = 1e-5)

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value = self.model(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), value.item()

    def evaluate_actions(self, states, actions):
        logits, values = self.model(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        return log_probs, values.squeeze(-1), entropy

    def update(self, rollout_buffer, ppo_epochs=2, batch_size=256, entropy_coeff=0.01):
        states, actions, old_log_probs, returns, advantages, old_values = rollout_buffer
        
        dataset_size = states.size(0)
        
        for _ in range(ppo_epochs):
            permutation = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, batch_size):
                batch_indices = permutation[start_idx:start_idx + batch_size]
                
                b_states = states[batch_indices].to(self.device)
                b_actions = actions[batch_indices].to(self.device)
                b_old_log_probs = old_log_probs[batch_indices].to(self.device)
                b_returns = returns[batch_indices].to(self.device)
                b_advantages = advantages[batch_indices].to(self.device)
                b_old_values = old_values[batch_indices].to(self.device) 
                
                new_log_probs, values, entropy = self.evaluate_actions(b_states, b_actions)

                ratios = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                v_loss_unclipped = (values - b_returns) ** 2
                values_clipped = b_old_values + torch.clamp(values - b_old_values, -self.clip_eps, self.clip_eps)
                v_loss_clipped = (b_returns - values_clipped) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                
                loss = actor_loss + 0.5 * value_loss - entropy_coeff * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()