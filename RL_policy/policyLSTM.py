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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMPolicy, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size).to(self.device),
                torch.zeros(1, batch_size, self.hidden_size).to(self.device))
    
    def forward(self, x):
        hidden = self.init_hidden(x.size(0))
        out, _ = self.lstm(x,hidden)

        out = self.fc(out)
        out = torch.softmax(out, dim=-1)

        return out

    


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
                probs = self.model(states_batch.to(self.device))  


                policy_gradients = torch.zeros(batch_size, seq_len)
                for t in range(seq_len):
                    actions = actions_batch[:, t]
                    rewards = rewards_batch[:, t]
                    probs_t = probs[:, t, :]
                    probs_selected = probs_t.gather(1, actions.unsqueeze(1)).squeeze(1)
                    policy_gradients[:, t] = -torch.log(probs_selected) * rewards

                ''' 
                Sum over time and batch dimensions to get total loss
                Calculates a scalar of the loss which is passed to the LSTM where BPTT happens.
                '''
                loss = -policy_gradients.mean(dim=1).sum()      


                # Calculate loss and backpropagate
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)   # gradient clipping to avoid vanishing or exploding gradients
                self.optimizer.step()

                running_loss += loss.item() * states_batch.shape[0]

            epoch_loss = running_loss / len(dataset)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")


    def predict(self, num_samples):
        self.model.eval()
        with torch.no_grad():
            states = self.data[0:num_samples].to(self.device)
            probs = self.model(states)
            max_probs, max_indices = torch.max(probs, dim=2)

            actions = max_indices

            '''
            no simulation of states implemented !
            '''

            return actions, states[:,:,-1]

    def discount_rewards(self, rewards, gamma):
        n = rewards.size(1)
        discounts = torch.tensor([gamma**i for i in range(n)]).unsqueeze(0).to(self.device)
        rewards_discounted = rewards * discounts
        rewards_discounted = torch.flip(rewards_discounted, [1]).cumsum(1)
        rewards_discounted = torch.flip(rewards_discounted, [1])
        return rewards_discounted


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
        states = self.data.to(self.device)

        # Sample actions for each state in each trajectory
        lstm_output = self.model.forward(states) # out: (num_traj, 2)
        lstm_output = lstm_output.detach()

        probs = lstm_output.squeeze()
        action_dist = torch.distributions.Categorical(probs=probs)

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


        rewards = self.discount_rewards(rewards, gamma)

        return states, actions, rewards
    

    
    def sequentialize_dataset(self, num_trajectories):
        df = self.dataset
        sequences = np.zeros((num_trajectories, self.seq_len, self.input_size))
        for i in range(num_trajectories):
            sequences[i] = df.iloc[i:i+self.seq_len]

        return torch.tensor(sequences, dtype=torch.float32)


batch_size = 16
seq_len = 100
input_size= 6
output_size= 2
episodes = 100

dataset = DataSet(start_date="2022-01-01", target="i_m1sum", scale_target=False, scale_variables=False, time_features=False, resample=None,dynamic_price=False, demand_price=0.5, feedin_price=0.5).pipeline()
dataset = dataset[["i_m1sum" , "demand_price", "feedin_price", "power_consumption_kwh", "thermal_consumption_kwh",  "kwh_eq_state"]]


env = Environment(levels=seq_len, max_storage_tank=16, optimum_storage=8, gamma1=0.5, gamma2=0.5, gamma3=0.5)
model = LSTMRL(input_size=input_size, hidden_size=1000, output_size=output_size, learning_rate=0.001, batch_size=batch_size, num_epochs=1, seq_len=seq_len, dataset=dataset)


for i in range(episodes):

    states, actions, rewards = model.sample_trajectories(num_trajectories=500, sequence_length=seq_len,num_inputs=input_size, env=env)

    print(actions[0])

    dataset = TensorDataset(states, actions, rewards)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.train(loader)




policy_actions, states = model.predict(num_samples=10)
print(policy_actions, states)