import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
_data_path = os.path.join(script_dir, '..', 'data')
sys.path.insert(0, _data_path)
from datafactory import DataSet
from policyGradient import Environment

class LSTMPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMPolicy, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        # out = self.fc(out[:, :, :])

        out = self.fc(out)
        out = torch.softmax(out, dim=-1)


        return out
        # out = self.fc(out)
        # out = torch.softmax(out.view(-1, self.output_size, 2), dim=2)

        # print(out.shape)

        # return out
        # out, _ = self.lstm(x)
        # out = self.fc(out)  # shape: (batch_size, seq_len, output_size * 2)
        # # out = out.reshape(-1, out.size(2) // 2, 2)  # shape: (batch_size, seq_len, output_size)
        # out = torch.softmax(out, dim=2)
        # print("out", out.shape)
        # return out
    


class LSTMRL:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, batch_size, num_epochs, seq_len, dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMPolicy(input_size, hidden_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_epochs = num_epochs
        self.seq_len = seq_len
        self.dataset = dataset


    def train(self, loader):
        for epoch in range(self.num_epochs):
            running_loss = 0.0

            for states_batch, actions_batch, rewards_batch in loader:

                batch_size = states_batch.shape[0]
                seq_len = states_batch.shape[1]


                # Forward pass
                self.model.zero_grad()
                outputs = self.model(states_batch)
                log_probs = F.log_softmax(outputs, dim=1)


                # print("Log probs and action shapes",log_probs.shape, actions_batch.shape, rewards_batch.shape)

                policy_gradients = torch.zeros(batch_size, seq_len)
                for t in range(seq_len):
                    for i in range(batch_size):
                        action = actions_batch[i, t]
                        reward = rewards_batch[i, t]
                        log_prob = log_probs[i, t, action]
                        policy_gradients[i, t] = -log_prob * reward

                # Sum over time and batch dimensions to get total loss
                loss = policy_gradients.sum()

                # Calculate loss and backpropagate
                
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * states_batch.shape[0]

            epoch_loss = running_loss / len(dataset)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")


    def predict(self, num_samples):
        self.model.eval()
        with torch.no_grad():
            states = self.data[0:num_samples].to(self.device)
            outputs = self.model(states)
            _, predicted = torch.max(outputs.data, 1)
            print("predicted", predicted)
            return predicted.cpu().numpy(), states[:,:,-1]

    

    # def sample_trajectories(self, num_trajectories, sequence_length, num_inputs, env, gamma=0.99):
    #     """
    #     Samples trajectories using the given LSTM policy model.

    #     Args:
    #         lstm_model: The LSTM policy model to use for sampling.
    #         num_trajectories: The number of trajectories to sample.
    #         sequence_length: The length of each trajectory.
    #         gamma: Discount factor.

    #     Returns:
    #         A tuple (states, actions, rewards) containing the sampled trajectories.
    #         - states: A tensor of shape (num_trajectories, sequence_length, num_inputs)
    #         containing the input states for each trajectory.
    #         - actions: A tensor of shape (num_trajectories, sequence_length) containing
    #         the actions taken for each state in each trajectory.
    #         - rewards: A tensor of shape (num_trajectories, sequence_length) containing
    #         the discounted rewards obtained for each action in each trajectory.
    #     """
    #     states = torch.zeros((num_trajectories, sequence_length, num_inputs))
    #     actions = torch.zeros((num_trajectories, sequence_length), dtype=torch.int64)
    #     rewards = torch.zeros((num_trajectories, sequence_length))

    #     # Sample the initial states for each trajectory
        
    #     self.data = self.sequentialize_dataset(num_trajectories=num_trajectories)
    #     # initial_states = dset[:,0,:]

    #     # states[:, 0, :] = initial_states

    #     # initial_states = dset[:,0,:]
    #     states = self.data

    #     # Sample actions for each state in each trajectory
    #     for t in range(sequence_length):
    #         lstm_output = self.model(states[:, t, :].unsqueeze(1)) # out: (num_traj, 2)
    #         log_probs = F.log_softmax(lstm_output.squeeze(), dim=1)
    #         action_dist = torch.distributions.Categorical(logits=log_probs)


    #         '''
    #         Sample from Distribution of PolicyLSTM Outputs
    #         '''
    #         actions[:, t] = action_dist.sample()

    #         print(actions.shape)
            
    #         for trajectory in range(num_trajectories):
    #             # Update the states for the next time step
    #             if t < sequence_length - 1:
    #                 states[trajectory, t+1, :] = env.step(s=states[trajectory, t, :], a=actions[trajectory, t])

    #             # Compute the discounted rewards for the current actions
    #             if t == sequence_length - 1:
    #                 # If this is the last time step, set the reward to zero
    #                 rewards[trajectory, t] = 0
    #             else:
    #                 rewards[trajectory, t] = env.reward(action=actions[trajectory, t], s=states[trajectory, t, :])

                    

    #     # Compute the discounted rewards for each trajectory
    #     for trajectory in range(num_trajectories):
    #         discounted_rewards = []
    #         Gt = 0
    #         for reward in reversed(rewards[trajectory]):
    #             Gt = reward + gamma * Gt
    #             discounted_rewards.append(Gt)
    #         discounted_rewards.reverse()
    #         rewards[trajectory] = torch.tensor(discounted_rewards)

    #     return states, actions, rewards

    def sample_trajectories(self, num_trajectories, sequence_length, num_inputs, env, gamma=0.99):
        """
        Samples trajectories using the given LSTM policy model.

        Args:
            lstm_model: The LSTM policy model to use for sampling.
            num_trajectories: The number of trajectories to sample.
            sequence_length: The length of each trajectory.
            gamma: Discount factor.

        Returns:
            A tuple (states, actions, rewards) containing the sampled trajectories.
            - states: A tensor of shape (num_trajectories, sequence_length, num_inputs)
            containing the input states for each trajectory.
            - actions: A tensor of shape (num_trajectories, sequence_length) containing
            the actions taken for each state in each trajectory.
            - rewards: A tensor of shape (num_trajectories, sequence_length) containing
            the discounted rewards obtained for each action in each trajectory.
        """
        states = torch.zeros((num_trajectories, sequence_length, num_inputs))
        actions = torch.zeros((num_trajectories, sequence_length), dtype=torch.int64)
        rewards = torch.zeros((num_trajectories, sequence_length))

        # Sample the initial states for each trajectory
            
        self.data = self.sequentialize_dataset(num_trajectories=num_trajectories)
        # initial_states = dset[:,0,:]

        # states[:, 0, :] = initial_states

        states = self.data

        # Sample actions for each state in each trajectory
        lstm_output = self.model.forward(states) # out: (num_traj, 2)

        log_probs = F.log_softmax(lstm_output.squeeze(), dim=1)
        action_dist = torch.distributions.Categorical(logits=log_probs)


        # Sample from Distribution of PolicyLSTM Outputs
        actions = action_dist.sample()

        for t in range(sequence_length):
            for trajectory in range(num_trajectories):
                # Update the states for the next time step
                if t < sequence_length - 1:
                    states[trajectory, t+1, :] = env.step(s=states[trajectory, t, :], a=actions[trajectory, t])

                # Compute the discounted rewards for the current actions
                if t == sequence_length - 1:
                    # If this is the last time step, set the reward to zero
                    rewards[trajectory, t] = 0
                else:
                    rewards[trajectory, t] = env.reward(action=actions[trajectory, t], s=states[trajectory, t, :])

        # Compute the discounted rewards for each trajectory
        for trajectory in range(num_trajectories):
            discounted_rewards = []
            Gt = 0
            for reward in reversed(rewards[trajectory]):
                Gt = reward + gamma * Gt
                discounted_rewards.append(Gt)
            discounted_rewards.reverse()
            rewards[trajectory] = torch.tensor(discounted_rewards)

        return states, actions, rewards
    

    
    def sequentialize_dataset(self, num_trajectories):
        df = self.dataset
        sequences = np.zeros((num_trajectories, self.seq_len, self.input_size))
        for i in range(num_trajectories):
            sequences[i] = df.iloc[i:i+self.seq_len]

        return torch.tensor(sequences, dtype=torch.float32)



seq_len = 24
input_size= 6
output_size= 2

dataset = DataSet(start_date="2022-01-01", target="i_m1sum", scale_target=False, scale_variables=False, time_features=False, resample=None,dynamic_price=False, demand_price=0.5, feedin_price=0.5).pipeline()
dataset = dataset[["i_m1sum" , "demand_price", "feedin_price", "power_consumption_kwh", "thermal_consumption_kwh",  "kwh_eq_state"]]


env = Environment(levels=seq_len, max_storage_tank=16, optimum_storage=8, gamma1=0.5, gamma2=0.5, gamma3=0.5)
model = LSTMRL(input_size=input_size, hidden_size=500, output_size=output_size, learning_rate=0.001, batch_size=32, num_epochs=1, seq_len=seq_len, dataset=dataset)


for i in range(100):

    states, actions, rewards = model.sample_trajectories(num_trajectories=20000, sequence_length=seq_len,num_inputs=input_size, env=env)

    # print(states.shape,actions.shape, rewards.shape)
    # print(states[2,:,-1], actions[2])


    dataset = TensorDataset(states, actions, rewards)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)



    model.train(loader)




policy_actions, states = model.predict(num_samples=10)
print(policy_actions, states)