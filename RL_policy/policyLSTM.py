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
script_dir = os.path.dirname(os.path.abspath(__file__))
_data_path = os.path.join(script_dir, '..', 'data')
sys.path.insert(0, _data_path)
from datafactory import DataSet
from policyGradient import Environment
from utils import plot_rewards_loss, plot_states


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNet, self).__init__()
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

# class PolicyNet(nn.Module):
#     def __init__(self, input_size, hidden_size=512, output_size=2, nhead=4, num_layers=2):
#         super(PolicyNet, self).__init__()
#         d_model = hidden_size

#         self.embedding = nn.Linear(input_size, d_model)
#         self.transformer = nn.Transformer(d_model, nhead, num_layers)
#         self.fc = nn.Linear(d_model, output_size)

#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.transpose(0, 1)

#         # Create target sequence by shifting input sequence by one time step
#         tgt = torch.zeros_like(x)
#         tgt[:, :-1, :] = x[:, 1:, :]

#         out = self.transformer(x, tgt)

#         out = out.transpose(0, 1)
#         out = self.fc(out)
#         out = torch.softmax(out, dim=-1)
#         return out

    
# class PolicyNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(PolicyNet, self).__init__()
#         self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1, dilation=1)
#         self.fc1 = nn.Linear(hidden_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.device = device

#     def forward(self, x):
#         x = x.transpose(1, 2)
#         out = F.relu(self.conv1(x))
#         out = out.transpose(1, 2)
#         out = F.relu(self.fc1(out))
#         out = self.fc2(out)
#         out = torch.softmax(out, dim=-1)

#         return out
    

# class PolicyNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(PolicyNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, hidden_size, kernel_size=(input_size, 3), padding=(0, 1), dilation=1)
#         self.fc1 = nn.Linear(hidden_size * 1, hidden_size)  # Adjust input dimensions based on conv1 output
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape
#         x = x.unsqueeze(1)  # Add channel dimension
#         out = F.relu(self.conv1(x))

#         # Adjust tensor dimensions for fully connected layers
#         _, _, h, w = out.shape
#         out = out.view(batch_size, w, -1)  # Reshape tensor for fully connected layers
#         out = F.relu(self.fc1(out))
#         out = self.fc2(out)
#         out = torch.softmax(out, dim=-1)

#         return out



# class PolicyNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, sequence_length=24):
#         super(PolicyNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, hidden_size, kernel_size=(input_size, 3), padding=(0, 1), dilation=1)
#         self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 3), padding=(0, 1), dilation=1)
#         self.conv3 = nn.Conv2d(hidden_size, output_size, kernel_size=(1, 1), padding=(0, 0), dilation=1)
        
#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape
#         x = x.unsqueeze(1)
#         out = F.relu(self.conv1(x))
#         out = F.relu(self.conv2(out))
#         out = self.conv3(out).squeeze(2)
        
#         out = torch.softmax(out, dim=-1)

#         return out







class LSTMRL:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, batch_size, num_epochs, seq_len, dataset, epsilon=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyNet(input_size, hidden_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.epsilon = epsilon
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

                #print(probs.shape, states_batch.shape, actions_batch.shape, rewards_batch.shape)

                '''
                Baseline implementation
                
                '''
                baseline = rewards_batch.mean(dim=1, keepdim=True)
                rewards_batch = rewards_batch - baseline


                policy_gradients = torch.zeros(batch_size, seq_len).to(self.device)
                for t in range(seq_len):
                    actions = actions_batch[:, t]
                    rewards = rewards_batch[:, t]
                    probs_t = probs[:, t, :]
                    probs_selected = probs_t.gather(1, actions.unsqueeze(1)).squeeze(1)    # get the probability of the chosen action
                    policy_gradients[:, t] = -torch.log(probs_selected) * rewards.to(self.device)

                ''' 
                Sum over time and batch dimensions to get total loss
                Calculates a scalar of the loss which is passed to the LSTM where BPTT happens.
                '''
                loss = policy_gradients.mean(dim=1).sum()      


                # Calculate loss and backpropagate
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)   # gradient clipping to avoid vanishing or exploding gradients
                self.optimizer.step()
                

                running_loss += loss.item() * states_batch.shape[0]

            epoch_loss = running_loss / len(dataset)
            self.scheduler.step()
        
            print(f"Loss: {epoch_loss:.4f}")

            return epoch_loss


    def predict(self, num_samples):
        self.model.eval()
        with torch.no_grad():
            states = self.data[0:num_samples].to(self.device)
            probs = self.model(states)
            max_probs, max_indices = torch.max(probs, dim=-1)

            actions = max_indices

            '''
            no simulation of states implemented !
            '''

            return actions, states[:,:,-1]

    def discount_rewards(self, rewards, gamma):
        n = rewards.size(1)
        discounts = torch.tensor([gamma**i for i in range(n)]).unsqueeze(0).to(self.device)
        rewards_discounted = rewards.to(self.device) * discounts
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
        states_const = states

        # Sample actions for each state in each trajectory
        lstm_output = self.model.forward(states) # out: (num_traj, 2)
        lstm_output = lstm_output.detach()

        probs = lstm_output.squeeze()
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


                # Update the states for the next time step
                if t < sequence_length - 1:
                    
                    s_1 = env.step(s=states[trajectory, t, :], a=actions[trajectory, t])
                    states[trajectory, t+1, -1] = s_1

                if actions[trajectory, t] not in [0,1]:
                    print("error")

                rewards[trajectory, t] = env.reward(action=actions[trajectory, t], s=states[trajectory, t, :])

        # rewards = self.discount_rewards(rewards, gamma)


        return states, actions, rewards, states_const
    

    
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


batch_size = 16
seq_len = 24
input_size= 5
hidden_size = 1048
lr = 0.000001
output_size= 2
episodes = 500
num_trajectories = 300 # max days: ~ 430
epsilon = 0.1

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
    "demand_price": 0.1,
    "feedin_price": 0.5
}



dataset = DataSet(start_date="2022-01-01", target="i_m1sum", scale_target=False, scale_variables=False, time_features=False, resample=None,dynamic_price=False, demand_price=args["demand_price"], feedin_price=args["feedin_price"]).pipeline()
dataset["excess"] = (dataset.i_m1sum - dataset.power_consumption_kwh).clip(lower=0)
dataset = dataset[["date", "excess", "demand_price", "feedin_price", "thermal_consumption_kwh", "kwh_eq_state"]]

print(dataset.excess[dataset.excess > 0])

env = Environment(levels=seq_len, max_storage_tank=args["max_storage_tank"], optimum_storage=args["optimum_storage"], gamma1=args["gamma1"], gamma2=args["gamma2"], gamma3=args["gamma3"])
model = LSTMRL(input_size=input_size, hidden_size=hidden_size, output_size=output_size, learning_rate=lr, batch_size=batch_size, num_epochs=1, seq_len=seq_len, dataset=dataset, epsilon=epsilon)


rewards_list, loss_list = [], []
for i in range(episodes):

    print("Episode " + str(i))

    states, actions, rewards, states_const = model.sample_trajectories(num_trajectories=num_trajectories, sequence_length=seq_len,num_inputs=input_size, env=env)
    
    j = 160

    print(actions[j], states[j,:,-1], torch.round(rewards[j]*100) / 100, rewards[j].sum(), np.array(actions.cpu()).mean())
    mean_reward = rewards.mean().detach().item()
    print("Reward Mean: ",mean_reward)

    dataset = TensorDataset(states_const, actions, rewards)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    loss = model.train(loader)

    loss_list.append(loss)
    rewards_list.append(mean_reward)

plot_rewards_loss(rewards_list, loss_list)
plot_states(states[j], actions[j], args["optimum_storage"])


''' Attention: wrong states; no implementation of step'''
policy_actions, states = model.predict(num_samples=1)
    
