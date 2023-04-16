import pandas as pd
import numpy as np
from environments.ENV_extensiveSearch import Environment



class Tree:

    def __init__(self, levels, max_storage_tank, optimum_storage, gamma1, gamma2, gamma3) -> None:
        self.levels = levels
        self.max_storage_tank = max_storage_tank
        self.optimum_storage = optimum_storage
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.cool_down = 0.1

        self.env = Environment(levels, max_storage_tank, optimum_storage, gamma1, gamma2, gamma3)


    def build_trees(self, levels):

        tree = np.zeros(2**(levels+1)-1,)
        tree[::2] =1
        tree = np.insert(tree, 0, 0)
        tree = np.insert(tree, 0, 0)

        res_sum = np.zeros(len(tree))
        states = np.zeros(len(tree))

        return tree, states, res_sum



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
            power_consumption = seq_row.power_consumption_kwh
            thermal_consumption = abs(seq_row.thermal_consumption_kwh)


            for n in range(nodes):
                
                root = tree[r]
                result_root = res_sum[r]
                state_root = states[r]
                
                
                left = tree[r*2 +1]
                right = tree[r*2 +2]

                if exploit: 
                    right = pv_excess - power_consumption
                    if right < 0: right = 0
                    
                
                '''
                reward(action, pv_excess, demand_price, feedin_price, power_consumption, thermal_consumption, state)
                '''
                
                res_sum[r*2 +1] = result_root + self.env.reward(left, pv_excess, demand_price, feedin_price, power_consumption, thermal_consumption, state_root)
                res_sum[r*2 +2] = result_root + self.env.reward(right,pv_excess, demand_price, feedin_price, power_consumption, thermal_consumption, state_root)
                
                '''
                step(a, s, thermal_consumption)
                '''
                
                states[r*2 +1] = self.env.step(left, state_root, thermal_consumption)
                states[r*2 +2] = self.env.step(right, state_root, thermal_consumption)
                
                
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


    def pipe(self, n_samples=30, random=False):

        random = self.random
        exploit = self.exploit
        n_samples = self.n_samples 
 
        reward_list = []
        states_list = []
        b_i_list = []


        _s_idx = self.df[self.df.date == self.start_date].index[0]

        for i in range(n_samples):
            _seq = self.df[_s_idx:_s_idx+100][["i_m1sum" ,"power_consumption_kwh", "thermal_consumption_kwh", "demand_price", "feedin_price", "kwh_eq_state"]]


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

        return np.array(res)
    


    def pipe_mypv(self):

        df = self.df
        levels = self.levels


        reward_list = []

        _s_idx = df[df.date == self.start_date].index[0]

        for i in range(self.n_samples):
            _seq = df[_s_idx:_s_idx+100][["i_m1sum" ,"power_consumption_kwh", "thermal_consumption_kwh", "demand_price", "feedin_price", "kwh_eq_state"]]

            reward_ = 0
            state = _seq["kwh_eq_state"].iloc[0]

            for i in range(levels):

                seq_row = _seq.iloc[i]

                pv_excess = seq_row.i_m1sum
                demand_price = seq_row.demand_price
                feedin_price = seq_row.feedin_price
                power_consumption = seq_row.power_consumption_kwh
                thermal_consumption = abs(seq_row.thermal_consumption_kwh)


                action_ =  pv_excess - power_consumption 

                if action_ < 0: 
                    action_ = 0

                # if state + action_ > self.max_storage_tank:
                #     action_ = self.max_storage_tank - state
                
                reward_tmp = self.env.reward(action_, pv_excess, demand_price, feedin_price, power_consumption, thermal_consumption, state)
                
                reward_ += reward_tmp
                
                state = self.env.step(action_, state, thermal_consumption)

            reward_list.append(reward_)

            _s_idx += 96
        
        return np.array(reward_list)
        


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
            _seq = self.df[_s_idx:_s_idx+100][["i_m1sum" ,"power_consumption_kwh", "thermal_consumption_kwh", "demand_price", "feedin_price", "kwh_eq_state"]]

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




        
        
    