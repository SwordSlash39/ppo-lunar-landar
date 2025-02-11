import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# Define the policy network
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.rnn = nn.Sequential(
             nn.Linear(8, 64),
             nn.Mish(),
             nn.Linear(64, 64),
             nn.Mish(),
             nn.Linear(64, 64),
             nn.Mish(),
             nn.Linear(64, 4),
             nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.rnn(x)

# Define the value network
class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.rnn = nn.Sequential(
             nn.Linear(8, 64),
             nn.Mish(),
             nn.Linear(64, 64),
             nn.Mish(),
             nn.Linear(64, 64),
             nn.Mish(),
             nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.rnn(x)

# Initialise the environment
env_name = "LunarLander-v3"
env = gym.make(env_name)

# Model
LEARNING_RATE = 1e-3
GAMMA = 0.99
EPSILON = 0.2
policy = Policy()
value = Value()
policy_optim = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
value_optim = torch.optim.Adam(value.parameters(), lr=LEARNING_RATE)

# Reset the environment to generate the first observation
states2, next_states2, old_policy2, actions2, rewards2, done2 = [], [], [], [], [], []
observation, info = env.reset(seed=42)
terminated = False
log = 50
i = 0
try:
    while True:
        i += 1
        states, next_states, old_policy, actions, rewards, done = [], [], [], [], [], []
        with torch.no_grad():
            while len(rewards) <= 500:
                # run the policy network on the observation
                log_action = policy(torch.tensor(observation, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                action = torch.exp(log_action)
                
                # Sample randomly to get action
                dist = torch.distributions.Categorical(action)
                action = dist.sample()
                
                # Append to states
                old_policy.append(log_action)
                states.append(observation)
                actions.append(action.item())

                # step (transition) through the environment with the action
                # receiving the next observation, reward and if the episode has terminated or truncated
                observation, reward, terminated, truncated, info = env.step(action.item())
                
                # Append to next_states
                next_states.append(observation)
                
                # If the episode has ended then we can reset to start a new episode
                if terminated:
                    rewards.append(0.0)
                    done.append(1.0)
                    terminated = False
                    break
                
                rewards.append(reward)
                done.append(0.0)
        
        # Reset
        observation, info = env.reset()
        
        # Only run if not on first iter
        if i == 1:
            states2, next_states2, old_policy2, actions2, rewards2, done2 = states, next_states, old_policy, actions, rewards, done
            continue
        
        # Convert to tensors
        states_tensor = torch.tensor(np.array(states2), dtype=torch.float32)
        next_states_tensor = torch.tensor(np.array(next_states2), dtype=torch.float32)
        actions_tensor = torch.tensor(actions2, dtype=torch.int64).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards2, dtype=torch.float32).unsqueeze(1)
        done_tensor = torch.tensor(done2, dtype=torch.float32).unsqueeze(1)
        old_policy_tensor = torch.stack(old_policy2, dim=0)
        
        # Run PPO 
        temporal_diff = rewards_tensor + (1-done_tensor) * GAMMA * value(next_states_tensor) - value(states_tensor)
        value_loss = temporal_diff.pow(2).mean()
        
        advantage = temporal_diff.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
        r_value = torch.exp(torch.gather(policy(states_tensor), 1, actions_tensor) - torch.gather(old_policy_tensor, 1, actions_tensor).detach())
        policy_loss = -(torch.min(r_value * advantage, advantage * torch.clamp(r_value, 1 - EPSILON, 1 + EPSILON))).mean()
        
        # Update the policy and value networks
        value_optim.zero_grad(set_to_none=True)
        policy_optim.zero_grad(set_to_none=True)
        
        value_loss.backward()
        policy_loss.backward()
        
        value_optim.step()
        policy_optim.step()
        
        # Set new data to old data
        states2, next_states2, old_policy2, actions2, rewards2, done2 = states, next_states, old_policy, actions, rewards, done
        
        # Log
        writer.add_scalar("Value Loss", value_loss.item(), i)
        writer.add_scalar("Policy Loss", policy_loss.item(), i)
        writer.add_scalar("Reward", sum(rewards), i)
        if i % log == 0:
            print(f"{i}: Value Loss: {value_loss.item()} | Policy Loss: {policy_loss.item()} | Reward: {sum(rewards)}")

except KeyboardInterrupt:
    env.close()
    writer.flush()
    writer.close()
