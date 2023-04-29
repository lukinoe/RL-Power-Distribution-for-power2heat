import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
_data_path = os.path.join(script_dir, '..', 'data')
sys.path.insert(0, _data_path)
from datafactory import DataSet
from environments.ENV_policyGradient import Environment
from utils import plot_rewards_loss, plot_states
from sklearn.preprocessing import StandardScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead=8, transformer_layers=1):
        super(PolicyNet, self).__init__()

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


class CriticNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead=8, transformer_layers=1):
        super(CriticNet, self).__init__()

        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, transformer_layers)

        self.fc0 = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        
        x = self.fc0(x)
        out = self.transformer_encoder(x)
        out = self.fc1(out)

        return out



class MCPolicyGrad:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, batch_size, num_epochs, seq_len, dataset, env, epsilon=0.1, lr_schedule=False, scaler=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyNet(input_size, hidden_size, output_size).to(self.device)
        self.env = env
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.lr_schedule = lr_schedule
        self.loss_fn = nn.CrossEntropyLoss()
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_epochs = num_epochs
        self.seq_len = seq_len
        self.dataset = dataset
        self.seq_dataset = self.sequentialize_dataset(num_trajectories=300)

        self.critic = CriticNet(input_size, hidden_size, 1).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.00001)
        self.scheduler_critic = StepLR(self.critic_optimizer, step_size=100, gamma=0.1)
        self.value_loss_fn = nn.MSELoss()

        self.scaler = scaler

        print("Params: ", sum(p.numel() for p in self.model.parameters()))


    def train(self):

        running_loss = 0.0
        running_value_loss = 0.0
        reward_running = 0.0

        num_trajectories = 300
        batch_size = self.batch_size
        num_batches = math.ceil(num_trajectories/ batch_size)

        states_dataset = self.seq_dataset.to(self.device)

        all_actions, all_states, all_rewards = torch.zeros((num_trajectories, self.seq_len)), torch.zeros((num_trajectories, self.seq_len)), torch.zeros((num_trajectories, self.seq_len))

        for i in range(num_batches):

            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if len(states_dataset) - end_idx < batch_size:
                break

            states_batch = states_dataset[start_idx:end_idx]
            
            self.model.zero_grad()
            states_simulated, actions_batch, rewards_batch, probs = self.sample_trajectories(data=states_batch, gamma=0.99)

            '''
            Store batches
            '''
            all_actions[start_idx:end_idx], all_rewards[start_idx:end_idx], all_states[start_idx:end_idx] = actions_batch, rewards_batch, states_simulated[:,:,-1]

            '''
            Predict baseline and use it to align rewards
            '''
            state_values = self.critic(self.scale(states_batch).to(self.device)).squeeze()
            baseline = state_values.detach().to(device)

            advantage_batch = rewards_batch - baseline

            '''
            Calculate Policy Gradients
            '''

            policy_gradients = torch.zeros(batch_size, self.seq_len).to(self.device)
            for t in range(self.seq_len):
                actions = actions_batch[:, t]
                advantage = advantage_batch[:, t]
                probs_t = probs[:, t, :]
                probs_selected = probs_t.gather(1, actions.unsqueeze(1)).squeeze(1)    # get the probability of the chosen action
                policy_gradients[:, t] = -torch.log(probs_selected) * advantage.to(self.device)

            ''' 
            Sum over time and batch dimensions to get total loss
            Calculates a scalar of the loss which is passed to perform backprop via computational graph.
            '''
            loss = policy_gradients.mean(dim=1).sum()      


            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)   # gradient clipping to avoid vanishing or exploding gradients
            self.optimizer.step()

            
            '''
            Update Critic Network
            '''
            self.critic.zero_grad()
            value_loss = self.value_loss_fn(state_values, rewards_batch)
            value_loss.backward()
            self.critic_optimizer.step()
            

            running_loss += loss.item() * states_batch.shape[0]
            running_value_loss += value_loss.item() * states_batch.shape[0]
            reward_running += rewards_batch.mean().item()

        
        epoch_loss = running_loss / num_batches
        epoch_value_loss = running_value_loss / num_batches
        epoch_reward = reward_running / num_batches

        if self.lr_schedule:
            self.scheduler.step()
            self.scheduler_critic.step()

        print(f"Loss: {epoch_loss:.4f}, Value Loss: {epoch_value_loss:.4f}")

        return epoch_loss, all_states, all_actions, epoch_reward



    def discount_rewards(self, rewards, gamma):
        n = rewards.size(1)
        discounts = torch.tensor([gamma**i for i in range(n)]).unsqueeze(0).to(self.device)
        rewards_discounted = rewards.to(self.device) * discounts
        rewards_discounted = torch.flip(rewards_discounted, [1]).cumsum(1)
        rewards_discounted = torch.flip(rewards_discounted, [1])
        return rewards_discounted


    def sample_trajectories(self, data, gamma=0.99):
        """
        Samples trajectories using the given LSTM policy model.

        Args:
            data:  state tensor of shape (num_trajectories, sequence_length, input_size) from which the trajectories should be simulated
            gamma: Discount factor.

        Returns:
            A tuple (states, actions, rewards) containing the sampled trajectories.
            - states: A tensor of shape (num_trajectories, sequence_length, num_inputs)
            containing the input states for each trajectory.
            - actions: A tensor of shape (num_trajectories, sequence_length) containing
            the actions taken for each state in each trajectory.
            - rewards: A tensor of shape (num_trajectories, sequence_length) containing
            the discounted rewards obtained for each action in each trajectory.
            - lstm_output: A tensor of shape (num_trajectories, sequence_length, 2) containing
            the softmax probabilities of the actions predicted by the policy network.
        """
        sequence_length, num_inputs = self.seq_len, self.input_size
        num_trajectories = data.shape[0]


        states = torch.zeros((num_trajectories, sequence_length, num_inputs))
        actions = torch.zeros((num_trajectories, sequence_length), dtype=torch.int64)
        rewards = torch.zeros((num_trajectories, sequence_length))

        states = data

        # Sample actions for each state in each trajectory
        lstm_output = self.model.forward(self.scale(states)) # out: (num_traj, 2)

        probs = lstm_output.squeeze().detach()
        action_dist = torch.distributions.Categorical(probs=probs)

        # Sample from Distribution of PolicyLSTM Outputs
        actions = action_dist.sample()

        uniform_dist = torch.distributions.Uniform(0, 1)
        random_actions = uniform_dist.sample((num_trajectories, sequence_length))


        for t in range(sequence_length):
            for trajectory in range(num_trajectories):

                # Epsilon-greedy action selection
                if random_actions[trajectory, t] < self.epsilon:
                    actions[trajectory, t] = torch.randint(0, probs.shape[-1], (1,)).item()


                if t < sequence_length - 1:
                    s_1 = self.env.step(s=states[trajectory, t, :], a=actions[trajectory, t])
                    states[trajectory, t+1, -1] = s_1


                assert actions[trajectory, t] in [0,1]

                rewards[trajectory, t] = self.env.reward(action=actions[trajectory, t], s=states[trajectory, t, :])

        # rewards = self.discount_rewards(rewards, gamma)

        return states, actions, rewards, lstm_output
    

    
    def sequentialize_dataset(self, num_trajectories, sample=False):
        df = self.dataset.copy()

        df['date'] = pd.to_datetime(df['date'])
        df_8_am = df[df['date'].dt.time == pd.to_datetime('12:00:00').time()]
        del df["date"]


        # Sample indices from the filtered dataframe
        sampled_indices = np.random.choice(list(df_8_am.index)[:num_trajectories], num_trajectories)
        index_list = df_8_am.index.to_list()

        upperBound = len(df) - self.seq_len
        sequences = np.zeros((num_trajectories, self.seq_len, self.input_size))
        for i in range(num_trajectories):
            u = index_list[i]
            if sample:
                u = sampled_indices[i] 

            sequences[i] = df.iloc[u:u+self.seq_len]
            ''' set storage state constant to first state '''
            sequences[i,:,-1] = sequences[i,0,-1]  


        return torch.tensor(sequences, dtype=torch.float32)
    


    def scale(self,tensor):


        tensor_reshaped = tensor.view(-1, tensor.shape[-1])
        tensor_np = tensor_reshaped.cpu().numpy()
        tensor_scaled_np = self.scaler.transform(tensor_np)
        tensor_scaled = torch.tensor(tensor_scaled_np, dtype=tensor.dtype).to(device)
        tensor_scaled = tensor_scaled.view(tensor.shape)

        return tensor_scaled
    
    def predict(self, index):

        states = self.seq_dataset[index]
        probs = self.model(self.scale(states)).cpu().detach()
        actions = torch.argmax(probs, dim=-1)
        rewards = torch.zeros(self.seq_len)

        for i in range(self.seq_len):
          
          if i < self.seq_len - 1:
            s_1 = self.env.step(s=states[i], a=actions[i])
            states[i+1, -1] = s_1
            rewards[i] = self.env.reward(action=actions[i], s=states[i])

        return states.numpy(), actions.numpy(), rewards.sum()


batch_size = 16
seq_len = 96
input_size= 5
hidden_size = 256
lr = 1e-05
output_size= 2
epochs = 200
num_trajectories = 300 # max days: ~ 430
epsilon = 0.0
lr_schedule = True

resample = None
if seq_len == 24:
  resample = "h"


'''
NUM_TRAJECTORIES: important parameter; case = 100: batch_size = 64 --> only 2 updates per epoch --> 64 + 36
'''

max_storage_tank = 18.52

args = {
    "max_storage_tank": max_storage_tank,
    "optimum_storage": max_storage_tank * 0.8,
    "gamma1": 0,    # financial
    "gamma2": 1,      # distance optimum
    "gamma3": 0.0,      # tank change
    "demand_price": 0.5,
    "feedin_price": 0.1
}



dataset = DataSet(start_date="2022-01-01", target="i_m1sum", scale_target=False, scale_variables=False, time_features=False, resample=resample,dynamic_price=False, demand_price=args["demand_price"], feedin_price=args["feedin_price"]).pipeline()
dataset["excess"] = (dataset.i_m1sum - dataset.power_consumption_kwh).clip(lower=0)
dataset = dataset[["date", "excess", "demand_price", "feedin_price", "thermal_consumption_kwh", "kwh_eq_state"]]


SC = StandardScaler()
scaler = SC.fit(dataset[["excess", "demand_price", "feedin_price", "thermal_consumption_kwh", "kwh_eq_state"]].values)


env = Environment(levels=seq_len, max_storage_tank=args["max_storage_tank"], optimum_storage=args["optimum_storage"], gamma1=args["gamma1"], gamma2=args["gamma2"], gamma3=args["gamma3"])
model = MCPolicyGrad(input_size=input_size, hidden_size=hidden_size, output_size=output_size, learning_rate=lr, batch_size=batch_size, num_epochs=1, seq_len=seq_len, dataset=dataset, env=env ,epsilon=epsilon, lr_schedule=lr_schedule, scaler=scaler)
day = 160

rewards_list, loss_list = [], []
for i in range(epochs):


    loss, states, actions, rewards = model.train()
    mean_reward = rewards
    

    loss_list.append(loss)
    rewards_list.append(mean_reward)
    
    print("Episode " + str(i))
    print("Reward Mean: ",mean_reward)
    print(states[day], actions[day])


plot_rewards_loss(rewards_list, loss_list)


for i in [50,60,75,85,100,115,125,140,150,175,203,204,205,220,225,245,250,260,270]:
    _states, _actions, _reward_sum = model.predict(i)
    plot_states(_states[:,-1], _actions, args["optimum_storage"], id=i)


torch.save(model.model.state_dict(), script_dir + "/checkpoints/model_" + str(seq_len))


    
