import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
_data_path = os.path.join(script_dir, '..', 'data')
sys.path.insert(0, _data_path)
from datafactory import DataSet
from policyGradient import Environment
from utils import plot_rewards_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class LSTMPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead=4, transformer_layers=2):
        super(LSTMPolicy, self).__init__()

        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, transformer_layers)

        self.fc0 = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
        x = self.fc0(x)
        out = self.transformer_encoder(x)

        out = self.fc1(out)
        out = torch.softmax(out, dim=-1)

        return out


    
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out.squeeze(-1)

    

class LSTMRLPPO:
    def __init__(self, input_size, hidden_size, output_size, critic_hidden_size, learning_rate, batch_size, num_epochs, seq_len, dataset, epsilon=0.1, c_lr=0.00001, ppo_epochs=4, ppo_clip=0.3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMPolicy(input_size, hidden_size, output_size).to(self.device)
        self.critic = Critic(input_size, critic_hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=c_lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_epochs = num_epochs
        self.seq_len = seq_len
        self.dataset = dataset
        self.ppo_epochs = ppo_epochs
        self.ppo_clip = ppo_clip
        self.c1 = 0.5
        self.c2 = 0.1


    def train(self, loader):
        for epoch in range(self.ppo_epochs):
            running_loss = 0.0
            running_critic_loss = 0.0

            for states_batch, actions_batch, rewards_batch, old_log_probs_batch, returns_batch, advs_batch in loader:
                # Forward pass
                self.model.zero_grad()
                self.critic.zero_grad()

                probs = self.model(states_batch.to(self.device))  
                values = self.critic(states_batch.to(self.device))

                log_probs = torch.zeros_like(probs)
                log_probs.scatter_add_(2, actions_batch.unsqueeze(2).to(self.device), torch.ones_like(probs).to(self.device))
                log_probs = log_probs.masked_fill(log_probs == 0, float('-inf'))
                log_probs = torch.log(probs) + log_probs
                log_probs = log_probs.masked_fill(log_probs == float('-inf'), 0)

                selected_log_probs = torch.sum(log_probs * actions_batch.unsqueeze(2).to(self.device), dim=-1)

                ratio = torch.exp(selected_log_probs - old_log_probs_batch.to(self.device))
                surr1 = ratio * advs_batch.to(self.device)
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advs_batch.to(self.device)
                policy_loss = -torch.min(surr1, surr2).mean()

                critic_loss = self.c1 * F.mse_loss(values, returns_batch.to(self.device))

                entropy = -(probs * log_probs).sum(dim=-1).mean()
                loss = policy_loss + critic_loss - self.c2 * entropy

                # Backpropagate
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.critic_optimizer.step()

                running_loss += policy_loss.item() * states_batch.shape[0]
                running_critic_loss += critic_loss.item() * states_batch.shape[0]

            epoch_loss = running_loss / len(dataset)
            epoch_critic_loss = running_critic_loss / len(dataset)

            print(f"Policy Loss: {epoch_loss:.4f}, Critic Loss: {epoch_critic_loss:.4f}")

        return epoch_loss, epoch_critic_loss
    

    def sample_trajectories(self, num_trajectories, sequence_length, num_inputs, env, epsilon, gamma=0.99, lambda_=0.95):
        states = torch.zeros((num_trajectories, sequence_length, num_inputs))
        actions = torch.zeros((num_trajectories, sequence_length), dtype=torch.int64)
        rewards = torch.zeros((num_trajectories, sequence_length))
        old_log_probs = torch.zeros((num_trajectories, sequence_length))
        values = torch.zeros((num_trajectories, sequence_length))

        # Sample the initial states for each trajectory
        self.data = self.sequentialize_dataset(num_trajectories=num_trajectories)
        states = self.data.to(self.device)
        states_const = states

        # Sample actions for each state in each trajectory
        lstm_output = self.model.forward(states)  # out: (num_traj, seq_len, num_actions)
        lstm_output = lstm_output.detach()

        # Calculate action probabilities and values
        probs = lstm_output.squeeze()
        action_dist = torch.distributions.Categorical(probs=probs)
        values = self.critic(states).squeeze()

        # Sample from Distribution of PolicyLSTM Outputs
        actions = action_dist.sample()

        uniform_dist = torch.distributions.Uniform(0, 1)
        random_actions = uniform_dist.sample((num_trajectories, sequence_length))

        for t in range(sequence_length):
            for trajectory in range(num_trajectories):

                # Epsilon-greedy action selection
                if random_actions[trajectory, t] < epsilon:
                    actions[trajectory, t] = torch.randint(0, probs.shape[-1], (1,)).item()

                # Update the states for the next time step
                if t < sequence_length - 1:
                    s_1 = env.step(s=states[trajectory, t, :], a=actions[trajectory, t])
                    states[trajectory, t+1, -1] = s_1

                rewards[trajectory, t] = env.reward(action=actions[trajectory, t], s=states[trajectory, t, :])

                # Calculate the log probability of the action taken
                old_log_probs[trajectory, t] = action_dist.log_prob(actions[trajectory, t])[trajectory, t]

        # Calculate the returns and advantages using GAE (Generalized Advantage Estimation)
        returns = self.discount_rewards(rewards, gamma)
        advantages = self.calculate_advantages(returns, values, rewards, gamma, lambda_, num_trajectories)

        return states, actions, rewards, old_log_probs, returns, advantages, states_const

    def calculate_advantages(self, returns, values, rewards, gamma, lambda_, num_trajectories):
        values = torch.cat([values, torch.zeros(num_trajectories, 1)], dim=-1)
        td_errors = rewards + gamma * values[:, 1:].detach() - values[:, :-1].detach()
        advantages = torch.zeros_like(td_errors)
        running_advantage = 0
        for t in reversed(range(td_errors.size(1))):
            running_advantage = td_errors[:, t] + gamma * lambda_ * running_advantage
            advantages[:, t] = running_advantage
        return advantages
    

    def sequentialize_dataset(self, num_trajectories):
        df = self.dataset.copy()

        df['date'] = pd.to_datetime(df['date'])

        # Filter the dataframe to include only rows with a time of 8:00 am
        df_8_am = df[df['date'].dt.time == pd.to_datetime('08:00:00').time()]

        # Sample indices from the filtered dataframe
        sampled_indices = np.random.choice(list(df_8_am.index), num_trajectories)

        index_list = df_8_am.index.to_list()

        del df["date"]


        upperBound = len(df) - self.seq_len
        sequences = np.zeros((num_trajectories, self.seq_len, self.input_size))
        for i in range(num_trajectories):
            u = index_list[i]

            sequences[i] = df.iloc[u:u+self.seq_len]
            ''' set storage state constant to first state '''
            sequences[i,:,-1] = sequences[i,0,-1]  


        return torch.tensor(sequences, dtype=torch.float32)
    

    def discount_rewards(self, rewards, gamma):
        n = rewards.size(1)
        discounts = torch.tensor([gamma**i for i in range(n)]).unsqueeze(0).to(self.device)
        rewards_discounted = rewards.to(self.device) * discounts
        rewards_discounted = torch.flip(rewards_discounted, [1]).cumsum(1)
        rewards_discounted = torch.flip(rewards_discounted, [1])
        return rewards_discounted




batch_size = 16
seq_len = 24
input_size= 5
hidden_size = 256
lr = 0.00001
output_size= 2
episodes = 500
num_trajectories = 300 # max days: ~ 430
epsilon = 0.00

'''
NUM_TRAJECTORIES: important parameter; case = 100: batch_size = 64 --> only 2 updates per epoch --> 64 + 36
'''

max_storage_tank = 18.52

args = {
    "max_storage_tank": max_storage_tank,
    "optimum_storage": max_storage_tank * 0.9,
    "gamma1": 0,    # financial
    "gamma2": 1,      # distance optimum
    "gamma3": 0.0,      # tank change
    "demand_price": 0.5,
    "feedin_price": 0.1
}



dataset = DataSet(start_date="2022-01-01", target="i_m1sum", scale_target=False, scale_variables=False, time_features=False, resample=None,dynamic_price=False, demand_price=args["demand_price"], feedin_price=args["feedin_price"]).pipeline()
dataset["excess"] = (dataset.i_m1sum - dataset.power_consumption_kwh).clip(lower=0)
dataset = dataset[["date", "excess", "demand_price", "feedin_price", "thermal_consumption_kwh", "kwh_eq_state"]]

print(dataset, dataset.kwh_eq_state.mean())

env = Environment(levels=seq_len, max_storage_tank=args["max_storage_tank"], optimum_storage=args["optimum_storage"], gamma1=args["gamma1"], gamma2=args["gamma2"], gamma3=args["gamma3"])
model = LSTMRLPPO(input_size=input_size, hidden_size=hidden_size, output_size=output_size, critic_hidden_size=hidden_size, learning_rate=lr, batch_size=batch_size, num_epochs=1, seq_len=seq_len, dataset=dataset, epsilon=epsilon)

rewards_list, loss_list = [], []
for i in range(episodes):

    print("Episode " + str(i))
    

    states, actions, rewards, old_log_probs, returns, advantages, states_const = model.sample_trajectories(num_trajectories=num_trajectories, sequence_length=seq_len, num_inputs=input_size, env=env, epsilon=epsilon)
    dataset = TensorDataset(states, actions, rewards, old_log_probs, returns, advantages)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    j = 250

    print(actions[j], states[j,:,-1], torch.round(rewards[j]*100) / 100, np.array(actions).mean())

    mean_reward = rewards.mean().detach().item()
    print("Reward Mean: ", mean_reward)

    dataset = TensorDataset(states, actions, rewards, old_log_probs, returns, advantages)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    loss = model.train(loader)

    loss_list.append(loss)
    rewards_list.append(mean_reward)

plot_rewards_loss(rewards_list, loss_list)

