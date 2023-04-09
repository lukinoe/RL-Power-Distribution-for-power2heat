import pandas as pd
import numpy as np
import torch
from environments.ENV_policyGradient import Environment






class Simulation(Environment):

    def __init__(self, levels, n_samples, dataset, args, start_date="2022-04-08 10:45:00", random=False, exploit=False) -> None:
        print(args)
        super().__init__(levels, args["max_storage_tank"], args["optimum_storage"], args["gamma1"], args["gamma2"], args["gamma3"])
        self.df = dataset
        self.start_date = start_date
        self.day_interval = 96          # 96 | 24

        self.n_samples = n_samples 
        self.exploit = exploit
        self.random = random


    def pipe(self, n_samples=30):

        random = self.random
        exploit = self.exploit
        n_samples = self.n_samples 
 
        reward_list = []
        states_list = []
        b_i_list = []


        _s_idx = self.df[self.df.date == self.start_date].index[0]

        for i in range(n_samples):
            _seq = self.df[_s_idx:_s_idx+100][["i_m1sum" ,"power_consumption", "thermal_consumption_kwh", "demand_price", "feedin_price", "kwh_eq_state"]]


            rewards, states, tree, f_level_idx = self.simulate(self.levels,_seq, exploit=exploit)
            f_rewards = rewards[f_level_idx:-3]
            b_i = f_rewards.argmax() 
            if random:
                b_i = np.random.choice(np.arange(f_rewards.size))


            reward_list.append(rewards[b_i + f_level_idx])
            states_list.append(states)
            b_i_list.append(b_i + f_level_idx)

            _s_idx += self.day_interval

        return tree, states_list, b_i_list, reward_list


    def results(self):

        res = []

        tree, states_list, b_i_list, reward_list = self.pipe(self.n_samples)

        for i in range(self.n_samples):
            rewards = reward_list[i]

            start_capacity = states_list[i][0]
            print(start_capacity)

            _seq, _states = self.backtrack_seq(tree, states_list[i], b_i_list[i], self.levels, start_capacity=start_capacity)
            res.append([_states[:-1], _seq, rewards])

        return res
        
