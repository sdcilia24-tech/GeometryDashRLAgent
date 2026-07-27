import time
import torch
import numpy as np
import pydirectinput as pdi
from PPOAgent import PPOAgent
import GeometryGym

torch.set_num_threads(4)
pdi.FAILSAFE = False
pdi.PAUSE = 0.0

CHANNELS = 4
NUM_ACTIONS = 2
BASE_STEPS_PER_ROLLOUT = 2048
MAX_STEPS_PER_ROLLOUT = 6144
MAX_EPISODES = 1000
BASE_ENTROPY = 0.01 
MAX_ENTROPY = 0.070
PLATEAU_THRESHOLD = 6
MAX_PROGRESS_PIXELS = 320.0 * 1.5 

gym = GeometryGym.GeometryGym()
agent = PPOAgent(in_channels=CHANNELS, num_actions=NUM_ACTIONS)

def compute_gae(rewards, values, dones, next_value, gamma=0.98, lam=0.95):
    advantages = []
    gae = 0.0
    
    # Ensure all value entries are clean float scalars
    clean_values = [v.item() if isinstance(v, torch.Tensor) else float(v) for v in values]
    clean_next_val = next_value.item() if isinstance(next_value, torch.Tensor) else float(next_value)
    
    local_values = clean_values + [clean_next_val] 
    
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * local_values[step + 1] * (1.0 - dones[step]) - local_values[step]
        gae = delta + gamma * lam * (1.0 - dones[step]) * gae
        advantages.insert(0, gae)
        
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(clean_values, dtype=torch.float32)
    return returns, advantages

def get_dynamic_rollout_steps(best_progress_pixels):
    true_progress_ratio = min(1.0, max(0.0, best_progress_pixels / MAX_PROGRESS_PIXELS))
    extra_steps = int(true_progress_ratio * (MAX_STEPS_PER_ROLLOUT - BASE_STEPS_PER_ROLLOUT))
    return ((BASE_STEPS_PER_ROLLOUT + extra_steps) // 256) * 256

def train():
    total_steps = 0
    best_all_time_progress = 0
    plateau_counter = 0
    current_entropy = BASE_ENTROPY
                                     
    state = gym.reset()
    
    for rollout in range(MAX_EPISODES):
        rollout_best_progress = 0
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        
        target_rollout_steps = get_dynamic_rollout_steps(best_all_time_progress)
        rollout_steps = 0
        rollout_start_time = time.time()
        total_rollout_reward = 0
        
        while rollout_steps < target_rollout_steps:
            total_steps += 1
            rollout_steps += 1
            
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done, info = gym.step(action, frame_skip=2)
            total_rollout_reward += reward
            
            states.append(state)
            actions.append(action if isinstance(action, (int, float)) else action.item())
            log_probs.append(log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob)
            rewards.append(reward)
            values.append(value.item() if isinstance(value, torch.Tensor) else value)
            dones.append(float(done))
            
            current_progress = info.get("progress", 0)
            rollout_best_progress = max(rollout_best_progress, current_progress)
            
            if done:
                state = gym.reset()
            else:
                state = next_state
        
        rollout_duration = time.time() - rollout_start_time
        fps = (target_rollout_steps * 2) / rollout_duration if rollout_duration > 0 else 0

        gym.update_stuck_state(rollout_best_progress)

        if gym.all_time_high < best_all_time_progress:
            best_all_time_progress = gym.all_time_high
            plateau_counter = 0
            current_entropy = BASE_ENTROPY
        
        if rollout_best_progress > best_all_time_progress:
            best_all_time_progress = rollout_best_progress
            plateau_counter = 0
            current_entropy = BASE_ENTROPY 
            print("resetting entropy")
        else:
            plateau_counter += 1
            if plateau_counter >= PLATEAU_THRESHOLD:
                ramp = (plateau_counter - PLATEAU_THRESHOLD + 1) * 0.005
                current_entropy = min(MAX_ENTROPY, BASE_ENTROPY + ramp)
        
        with torch.no_grad():
            _, _, next_val = agent.get_action(state)
            
        returns, advantages = compute_gae(rewards, values, dones, next_val)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        b_states = torch.as_tensor(np.stack(states), dtype=torch.float32)
        b_actions = torch.tensor(actions, dtype=torch.long)
        b_old_log_probs = torch.tensor(log_probs, dtype=torch.float32)
        b_old_values = torch.tensor(values, dtype=torch.float32)
        
        batch_size = 256 if target_rollout_steps >= 4096 else 128

        agent.update(
            (b_states, b_actions, b_old_log_probs, returns, advantages, b_old_values),
            batch_size=batch_size,
            ppo_epochs=2,
            entropy_coeff=current_entropy
        )
        
        true_pb_percent = (best_all_time_progress / MAX_PROGRESS_PIXELS) * 100.0
        rollout_pb_percent = (rollout_best_progress / MAX_PROGRESS_PIXELS) * 100.0

        print(
            f"Rollout {rollout + 1:3d} | "
            f"Steps: {target_rollout_steps:4d} | "
            f"FPS: {fps:5.1f} | "
            f"Rollout PB: {rollout_pb_percent:5.1f}% | "
            f"All-Time PB: {true_pb_percent:5.1f}% | "
            f"Stuck: {plateau_counter:2d} | "
            f"Entropy: {current_entropy:.4f} | " 
            f"Reward: {total_rollout_reward:.2f}"
        )

if __name__ == "__main__":
    print("training beginning")
    train()