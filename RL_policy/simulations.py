import pandas as pd
import numpy as np


class Tree:

    def __init__(self, levels, max_storage_tank, optimum_storage, gamma1, gamma2, gamma3) -> None:
        self.levels = levels
        self.max_storage_tank = max_storage_tank
        self.optimum_storage = optimum_storage
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.cool_down = 0.1


    def build_trees(self, levels):

        tree = np.zeros(2**(levels+1)-1,)
        tree[::2] =1
        tree = np.insert(tree, 0, 0)
        tree = np.insert(tree, 0, 0)

        res_sum = np.zeros(len(tree))
        states = np.zeros(len(tree))

        return tree, states, res_sum


    def step(self, s, thermal_consumption, a):
        '''
        action = [0; max_storage]
        '''
        heat_increase = a - self.cool_down
        s_1 = s + heat_increase - thermal_consumption
        
        if s_1 > self.max_storage_tank:
            s_1 = self.max_storage_tank
            
        return s_1


    def reward(self, action, pv_excess, demand_price, feedin_price, power_consumption, thermal_consumption, state):

        
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



    def simulate(self, levels, seq, exploit=False):

        tree, states, res_sum = self.build_trees(levels)

        start_capacity = seq.kwh_eq_state.to_list()[0]
        states[0] = start_capacity


        nodes = 1
        r = 0
        level_idx = 0 

        for i in range(levels): 

            level_idx = r  
            print(i,level_idx)

            seq_row = seq.iloc[i]

            pv_excess = seq_row.i_m1sum
            demand_price = seq_row.demand_price
            feedin_price = seq_row.feedin_price
            power_consumption = seq_row.power_consumption
            thermal_consumption = abs(seq_row.thermal_consumption_kwh)


            for n in range(nodes):
                
                root = tree[r]
                result_root = res_sum[r]
                state_root = states[r]
                
                
                left = tree[r*2 +1]
                right = tree[r*2 +2]

                if exploit: 
                    right = pv_excess - power_consumption - thermal_consumption 
                    if right < 0: right = 0
                    
                
                '''
                reward(action, pv_excess, demand_price, feedin_price, power_consumption, thermal_consumption, state)
                '''
                
                res_sum[r*2 +1] = result_root + self.reward(left, pv_excess, demand_price, feedin_price, power_consumption, thermal_consumption, state_root)
                res_sum[r*2 +2] = result_root + self.reward(right,pv_excess, demand_price, feedin_price, power_consumption, thermal_consumption, state_root)
                
                '''
                step(s, thermal_consumption, a)
                '''
                
                states[r*2 +1] = self.step(state_root, thermal_consumption, left)
                states[r*2 +2] = self.step(state_root, thermal_consumption, right)
                
                
                r +=1
                
            nodes *= 2

        return res_sum, states, tree, level_idx

    def backtrack_seq(self, tree, states, idx, levels, start_capacity):
        seq = []
        state_seq = []
        for i in range(levels):
            action_tmp = tree[int(idx)]
            
            if i == 0:
                seq.append(action_tmp)
                state_seq.append(start_capacity)
            
            if action_tmp == 0:
                i_add = -1
            if action_tmp == 1:
                i_add = -2
            
            idx = (idx + i_add)/2 
            a = tree[int(idx)]
            s = states[int(idx)]

            seq.append(a)
            state_seq.append(s)
        
        return np.flip(seq), np.flip(state_seq)



class Experiment(Tree):

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
        


class Experiment_Concat(Experiment):

    def __init__(self, levels, n_samples, dataset, args, start_date="2022-04-08 10:45:00", random=False, exploit=False, n_trees=5) -> None:
        super().__init__(levels, n_samples, dataset, args, start_date, random, exploit)

        self.n_trees = n_trees

    def pipe_concat(self):

        random = self.random
        exploit = self.exploit
        n_samples = self.n_samples 
        n_trees = self.n_trees

        reward_list = []
        b_i_list = []
        trees_list = []
        states_list = []

        _s_idx = self.df[self.df.date == self.start_date].index[0]

        for i in range(n_samples):
            _seq = self.df[_s_idx:_s_idx+100][["i_m1sum" ,"power_consumption", "thermal_consumption_kwh", "demand_price", "feedin_price", "kwh_eq_state"]]
            #_seq.i_m1sum /= 1000
            #_seq.power_consumption /= 1000

            seq_splits = np.array_split(_seq, n_trees)

            for t in range(n_trees):

                seq_split = seq_splits[t]
                if t != 0:
                    seq_split.kwh_eq_state.iloc[0] = last_state

                rewards, states, tree, f_level_idx = self.simulate(self.levels, seq_split, exploit=exploit)
                f_rewards = rewards[f_level_idx:-3]
                b_i = f_rewards.argmax() 
                if random:
                    b_i = np.random.choice(np.arange(f_rewards.size))

                b_i_list.append(b_i + f_level_idx)
                reward_list.append(rewards[b_i + f_level_idx])
                trees_list.append(tree)
                states_list.append(states)
                last_state = states[b_i + f_level_idx]

            _s_idx +=self.day_interval

        return np.array(reward_list).reshape(n_samples, n_trees ), np.array(b_i_list).reshape(n_samples, n_trees), np.array(trees_list).reshape(n_samples, n_trees, len(tree)), np.array(states_list).reshape(n_samples, n_trees, len(tree))        

    def results_concat(self):

        rewards, bi_list, trees_list, states_list = self.pipe_concat()

        states = []
        sequences = []
        for n in range(self.n_samples):
            for i in range(self.n_trees):
                start_capacity = states_list[n][i][0]
                _seq, _states = self.backtrack_seq(trees_list[n][i], states_list[n][i], bi_list[n][i], self.levels, start_capacity=start_capacity)
                states.append(_states[:-1])
                sequences.append(_seq)

        return states, sequences


# class Dataset:

#     def __init__(self) -> None:
#         df = pd.read_csv("../data/dset_08-12.csv")
#         #df.i_m1sum /= 1000
#         #df.power_consumption /= 1000
#         print(df)
#         self.df = df

#     def get_data(self):
#         return self.df

#     def set_prices(self, demand_price, feedin_price):
#         self.df.demand_price = demand_price
#         self.df.feedin_price = feedin_price

#     def get_seq(self, size=20, start_index="2022-04-08 10:45:00"):
#         _s_idx = self.df[self.df.date == start_index].index[0]

#         return self.df[_s_idx:_s_idx+size][["i_m1sum" ,"power_consumption", "thermal_consumption_kwh", "demand_price", "feedin_price", "kwh_eq_state"]]
