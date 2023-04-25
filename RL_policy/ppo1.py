import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

# ... (other imports and definitions) ...


class PPO:
    def __init__(self, policy, optimizer, clip_epsilon=0.2, num_epochs=10):
        self.policy = policy
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.num_epochs = num_epochs

    def update(self, states, actions, old_log_probs, returns, advantages):
        for _ in range(self.num_epochs):
            log_probs = self.policy(states).gather(2, actions.unsqueeze(-1)).squeeze(-1)
            ratios = torch.exp(log_probs - old_log_probs)
            clipped_ratios = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

            loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()


# ... (LSTMRL class definition) ...


# Update the LSTMRL class
class LSTMRL:
    # ... (other methods) ...

    def train(self, ppo, loader):
        for epoch in range(self.num_epochs):
            running_loss = 0.0

            for states_batch, actions_batch, old_log_probs_batch, returns_batch, advantages_batch in loader:
                ppo.update(states_batch.to(self.device), actions_batch.to(self.device),
                           old_log_probs_batch.to(self.device), returns_batch.to(self.device),
                           advantages_batch.to(self.device))

            if lr_schedule:
                self.scheduler.step()

            print(f"Loss: {epoch_loss:.4f}")

            return epoch_loss

    # ... (other methods) ...


# ... (main code) ...


# Create the PPO algorithm
ppo = PPO(model.model, model.optimizer)

# Change the training loop to use PPO
rewards_list, loss_list = [], []
for i in range(episodes):

    print("Episode " + str(i))

    states, actions, rewards, states_const = model.sample_trajectories(num_trajectories=num_trajectories,
                                                                       sequence_length=seq_len, num_inputs=input_size,
                                                                       env=env)

    j = 160

    print(actions[j], states[j, :, -1], torch.round(rewards[j] * 100) / 100, rewards[j].sum(),
          np.array(actions.cpu()).mean())
    mean_reward = rewards.cpu().mean().detach().item()
    print("Reward Mean: ", mean_reward)

    # Calculate advantages and returns
    returns = model.discount_rewards(rewards, gamma)
    advantages = returns - returns.mean(dim=1, keepdim=True)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    old_log_probs = model.model(states_const.to(model.device)).gather(2, actions.unsqueeze(-1)).squeeze(-1).detach()

    dataset = TensorDataset(states_const, actions, old_log_probs, returns, advantages)
    loader
