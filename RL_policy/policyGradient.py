import pandas as pd
import numpy as np
import torch


class Environment:

    def __init__(self, levels, max_storage_tank, optimum_storage, gamma1, gamma2, gamma3) -> None:
        self.levels = levels
        self.max_storage_tank = max_storage_tank
        self.optimum_storage = optimum_storage
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.cool_down = 0.1




    def step(self, s, a):
        '''
        action = [0; max_storage]
        '''

        state = s[-1]
        thermal_consumption = s[0]

        heat_increase = a - self.cool_down
        s_1 = state + heat_increase - thermal_consumption
        
        if s_1 > self.max_storage_tank:
            s_1 = self.max_storage_tank


        s_next = s
        s_next[-1] = s_1

         
        return s_next


    def reward(self, action, s):

    
        pv_excess, demand_price, feedin_price, power_consumption, thermal_consumption, state = s


        kwh_increase = action
        
        if state+kwh_increase > self.max_storage_tank:
            kwh_increase = self.max_storage_tank - state
        
        '''
        FINANCIAL PENALTY
        '''
        
        # diff = pv_excess - kwh_increase - power_consumption - thermal_consumption    
            
        # if diff < 0:
        #     reward = -(diff * demand_price + (kwh_increase - diff) * feedin_price)
        # else:
        #     reward = -(kwh_increase * feedin_price)  + diff * feedin_price
        
        consumption = power_consumption + thermal_consumption    
        diff = pv_excess - consumption    
            
        if diff < 0:
            reward = -(kwh_increase * demand_price) - (abs(diff) * demand_price) -  (pv_excess * feedin_price)              # - heat*d - abs(diff)*d - "consumption_via_pv"*f
        else:
            if diff - kwh_increase > 0:
                reward =  -(kwh_increase * feedin_price)  - (consumption * feedin_price) + (diff * feedin_price)            # - heat*f - consumption*f + diff*f
            else:
                reward = -(abs(diff - kwh_increase) * demand_price) - (diff * feedin_price) + (consumption* feedin_price)   # - heat_1 * d - heat_2*f + consumption*f
            
        reward = reward * self.gamma1
        
        '''
        PENALTY FOR DISTANCE TO OPTIMUM
        '''
        reward -= abs(state - self.optimum_storage)*self.gamma2


        '''
        REWARD/PENALTY FOR TANK STATE CHANGE (OPTIONAL)
        '''
        reward += (kwh_increase - self.cool_down)*self.gamma3
        
        
        return reward






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
        
