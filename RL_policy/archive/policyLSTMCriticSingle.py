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
from environments.ENV_policyGradient import Environment
from utils import plot_rewards_loss, plot_states
from sklearn.preprocessing import StandardScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# class PolicyNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size=2):
#         super(PolicyNet, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_size *2 *24, output_size)
#         self.relu = nn.ReLU()
#         self.device = device

#     def init_hidden(self, batch_size):
#         return (torch.zeros(2, batch_size, self.hidden_size).to(self.device),
#                 torch.zeros(2, batch_size, self.hidden_size).to(self.device))
    
#     def forward(self, x):

#         print(x.shape)
#         hidden = self.init_hidden(x.size(0))
#         out, _ = self.lstm(x,hidden)
#         out = out.flatten()
#         out = self.fc(out)
#         out = torch.softmax(out, dim=-1)

#         return out
    

# class PolicyNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size=2):
#         super(PolicyNet, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_size *2 *24, output_size)
#         self.relu = nn.ReLU()
#         self.device = device

#     def init_hidden(self, batch_size):
#         return (torch.zeros(2, batch_size, self.hidden_size).to(self.device),
#                 torch.zeros(2, batch_size, self.hidden_size).to(self.device))
    
#     def forward(self, x):
#         x = x.reshape((1, x.shape[0], x.shape[1]))
#         batch_size = x.size(0)
#         seq_len = x.size(1)
        
#         # Pass the input through the LSTM layer
#         hidden = self.init_hidden(batch_size)
#         out, _ = self.lstm(x, hidden)  # out.shape: (batch_size, seq_len, hidden_size*2)
        
#         # Reshape the output to (batch_size, hidden_size*2*seq_len)
#         out = out.flatten()
        
#         # Pass the output through the fully-connected layer
#         out = self.fc(out)
        
#         # Apply the softmax activation function
#         out = torch.softmax(out, dim=-1)


#         return out


class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len, nhead=8, transformer_layers=1):
        super(PolicyNet, self).__init__()

        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, transformer_layers)

        self.fc0 = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size*seq_len, output_size)

    def forward(self, x):

        x = x.reshape((1, x.shape[0], x.shape[1]))
        
        x = self.fc0(x)
        out = self.transformer_encoder(x)

        out = out.flatten()

        out = F.relu(self.fc1(out))
        out = torch.softmax(out, dim=-1)

        return out
    

# class PolicyNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, nhead=8, transformer_layers=1):
#         super(PolicyNet, self).__init__()

#         encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, transformer_layers)

#         self.fc0 = nn.Linear(input_size, hidden_size)
#         self.fc1 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = x.reshape((1, x.shape[0], x.shape[1]))
        
#         x = self.fc0(x)
#         out = self.transformer_encoder(x)

#         # Instead of flattening, take the last state of the sequence
#         out = out[:, -1]

#         out = F.relu(self.fc1(out))
        
#         # Since it's a binary action, ensure the output size is 2
#         assert out.shape[-1] == 2, "Output size should be 2 for binary action"
#         out = torch.softmax(out, dim=-1)

#         return out

class CriticNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len, nhead=8, transformer_layers=1):
        super(CriticNet, self).__init__()

        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, transformer_layers)

        self.fc0 = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size*seq_len, 1)

    def forward(self, x):

        x = x.reshape((1, x.shape[0], x.shape[1]))
        
        x = self.fc0(x)
        out = self.transformer_encoder(x)

        out = out.flatten()
        out = self.fc1(out)

        return out



class LSTMRL:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, batch_size, num_epochs, seq_len, dataset, env, epsilon=0.1, lr_schedule=False, scaler=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.model = PolicyNet(input_size, hidden_size, output_size, seq_len=seq_len).to(self.device)
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
        self.critic = CriticNet(input_size, hidden_size, 1, seq_len=seq_len).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.00001)
        self.scheduler_critic = StepLR(self.critic_optimizer, step_size=100, gamma=0.1)
        self.value_loss_fn = nn.MSELoss()

        self.scaler = scaler

        print("Params: ", sum(p.numel() for p in self.model.parameters()))


    def run(self, loader):
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            running_value_loss = 0.0

            for states_batch, actions_batch, rewards_batch, probs in loader:

                batch_size = states_batch.shape[0]
                seq_len = states_batch.shape[1]



                # Forward pass
                self.model.zero_grad()
                # probs = self.model(self.scale(states_batch).to(self.device))  

                #print(probs.shape, states_batch.shape, actions_batch.shape, rewards_batch.shape)

                # Get the baseline using the critic network
                state_values = self.critic(self.scale(states_batch).to(self.device)).squeeze()
                baseline = state_values.detach()

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

                # Train the critic network
                self.critic.zero_grad()
                value_loss = self.value_loss_fn(state_values, rewards_batch)
                value_loss.backward()
                self.critic_optimizer.step()
                

                running_loss += loss.item() * states_batch.shape[0]
                running_value_loss += value_loss.item() * states_batch.shape[0]

            
            epoch_loss = running_loss / len(dataset)
            epoch_value_loss = running_value_loss / len(dataset)

            if self.lr_schedule:
                self.scheduler.step()
                self.scheduler_critic.step()

            print(f"Loss: {epoch_loss:.4f}, Value Loss: {epoch_value_loss:.4f}")

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


    def sample_trajectories(self, num_trajectories, sequence_length, num_inputs, gamma=0.99):
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
        lstm_output = self.model.forward(self.scale(states)) # out: (num_traj, 2)
        lstm_output = lstm_output.detach()

        probs = lstm_output.squeeze()
        action_dist = torch.distributions.Categorical(probs=probs)

        # Sample from Distribution of PolicyLSTM Outputs
        actions = action_dist.sample()

        uniform_dist = torch.distributions.Uniform(0, 1)
        random_actions = uniform_dist.sample((num_trajectories, sequence_length))



        for t in range(sequence_length):
            for trajectory in range(num_trajectories):


                lstm_output = self.model.forward(self.scale(states)) # out: (num_traj, 2)
                lstm_output = lstm_output.detach()

                probs = lstm_output.squeeze()
                action_dist = torch.distributions.Categorical(probs=probs)

                # Sample from Distribution of PolicyLSTM Outputs
                actions = action_dist.sample()


                # Epsilon-greedy action selection
                if random_actions[trajectory, t] < self.epsilon:
                    actions[trajectory, t] = torch.randint(0, probs.shape[-1], (1,)).item()

                # excess = states[trajectory, t, 0]
                # if excess == 0:
                #     actions[trajectory, t] = 0

                # Update the states for the next time step
                if t < sequence_length - 1:
                    
                    s_1 = self.env.step(s=states[trajectory, t, :], a=actions[trajectory, t])
                    states[trajectory, t+1, -1] = s_1

                if actions[trajectory, t] not in [0,1]:
                    print("error")

                rewards[trajectory, t] = self.env.reward(action=actions[trajectory, t], s=states[trajectory, t, :])

        # rewards = self.discount_rewards(rewards, gamma)


        return states, actions, rewards, states_const, probs
    

    def main(self):
        
        episodes = 20000
        steps = self.seq_len
        all_rewards = []

        self.data = self.sequentialize_dataset(num_trajectories=episodes)
        states = self.data.to(self.device)

        for episode in range(episodes):

            log_probs = []
            rewards = []
            actions = []
            eq_states = []
            state_values = torch.zeros(self.seq_len)
            loss_l = []
            value_loss_l = []


            for step in range(steps):

                state_step = states[episode,step:]
                state_step_padded = torch.nn.functional.pad(state_step, (0, 0, 0, steps - len(state_step)), mode='constant', value=0)
                state_step_padded = state_step_padded[:steps]  # crop the tensor to the fixed shape (24, 5)


                action, log_prob = self.get_action(state_step_padded)

                state_value = self.critic(self.scale_tensor(state_step_padded).to(self.device)).squeeze()
                
                

                s_1 = self.env.step(s=states[episode,step], a=action)
                reward = self.env.reward(action=action, s=states[episode,step])
                

                '''
                update state  | Attention: sets ALL storage_states to constant = s_1
                '''
                if step +1 < steps:
                    states[episode, :, -1] = s_1


                log_probs.append(log_prob)
                rewards.append(reward)
                actions.append(action)
                state_values[step] = state_value
                eq_states.append(states[episode,0,-1].item())


                # print(sum(rewards))

                if step == steps -1:
                    loss, value_loss = self.update_policy(rewards, log_probs, state_values)   # update PER episode
                    loss_l.append(loss)
                    value_loss_l.append(value_loss)
                    all_rewards.append(np.sum(rewards))

                
            upd = 100
            if episode % upd == 0:
                print(str(episode), " : \n","Reward: " ,sum(all_rewards[-upd:]), "Loss: ", sum(loss_l[-upd:]) , "Value Loss: ", sum(value_loss_l[-upd:]))
                print(actions, eq_states)
            
        return all_rewards, loss_l
            
    def update_policy(self, rewards, log_probs, state_values, GAMMA=0.99):
        discounted_rewards = []


        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
            
        discounted_rewards_initial = torch.tensor(discounted_rewards)
        discounted_rewards = discounted_rewards_initial - state_values.detach()


        # discounted_rewards = torch.tensor(discounted_rewards)
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards


        '''
        Update Actor
        '''

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
        
        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) 
        self.optimizer.step()

        '''
        Update Critic
        '''

        self.critic.zero_grad()
        value_loss = self.value_loss_fn(state_values, discounted_rewards_initial)
        value_loss.backward()
        self.critic_optimizer.step()

        return policy_gradient.detach().item(), value_loss.detach().item()




    def get_action(self, state):

        num_actions = 2
        probs = self.model.forward(self.scale_tensor(state))

        highest_prob_action = np.random.choice(num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])

        return highest_prob_action, log_prob
    


    
    def sequentialize_dataset(self, num_trajectories, sample=False):
        df = self.dataset.copy()

        df['date'] = pd.to_datetime(df['date'])

        # Filter the dataframe to include only rows with a time of 8:00 am
        df_8_am = df[df['date'].dt.time == pd.to_datetime('12:00:00').time()]

        # Sample indices from the filtered dataframe
        sampled_indices = np.random.choice(list(df_8_am.index)[:300], num_trajectories)
        index_list = df_8_am.index.to_list()

        del df["date"]


        upperBound = len(df) - self.seq_len
        sequences = np.zeros((num_trajectories, self.seq_len, self.input_size))
        for i in range(num_trajectories):
            u = sampled_indices[i]
            if sample:
                u = sampled_indices[i] 

            sequences[i] = df.iloc[u:u+self.seq_len]
            ''' set storage state constant to first state '''
            sequences[i,:,-1] = sequences[i,0,-1]  


        return torch.tensor(sequences, dtype=torch.float32)

    
    def scale_tensor(self,tensor):
        array = tensor.numpy()
        scaled_array = self.scaler.transform(array)
        scaled_tensor = torch.from_numpy(scaled_array).float()

        return scaled_tensor


batch_size = 16
seq_len = 96
input_size= 5
hidden_size = 256
lr = 0.00001
output_size= 2
episodes = 200
num_trajectories = 300 # max days: ~ 430
epsilon = 0.1
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


print(dataset.excess[dataset.excess > 0])

env = Environment(levels=seq_len, max_storage_tank=args["max_storage_tank"], optimum_storage=args["optimum_storage"], gamma1=args["gamma1"], gamma2=args["gamma2"], gamma3=args["gamma3"])
model = LSTMRL(input_size=input_size, hidden_size=hidden_size, output_size=output_size, learning_rate=lr, batch_size=batch_size, num_epochs=1, seq_len=seq_len, dataset=dataset, env=env, epsilon=epsilon, lr_schedule=lr_schedule, scaler=scaler)



rewards, losses = model.main()
plot_rewards_loss(rewards, losses)



# rewards_list, loss_list = [], []
# for i in range(episodes):

#     print("Episode " + str(i))

#     states, actions, rewards, states_const, probs = model.sample_trajectories(num_trajectories=num_trajectories, sequence_length=seq_len,num_inputs=input_size, env=env)
    
#     j = 25

#     print(actions[j], states[j,:,-1], torch.round(rewards[j]*100) / 100, rewards[j].sum(), np.array(actions.cpu()).mean())
#     mean_reward = rewards.cpu().mean().detach().item()
#     print("Reward Mean: ",mean_reward)

#     dataset = TensorDataset(states_const, actions, rewards, probs)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,)


#     loss = model.train(loader)

#     loss_list.append(loss)
#     rewards_list.append(mean_reward)

# plot_rewards_loss(rewards_list, loss_list)


# for i in [50,75,100,125,150,175,203,204,205,225,250,275,299]:
#     plot_states(states[i,:,-1].cpu(), actions[i].cpu(), args["optimum_storage"], id=i)


# ''' Attention: wrong states; no implementation of step'''
# policy_actions, states = model.predict(num_samples=1)
    
