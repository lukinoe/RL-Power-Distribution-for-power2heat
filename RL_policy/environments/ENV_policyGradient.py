import torch


def clip(scalar, min=None, max=None):       # way faster than numpy array.clip()
    if min is not None and scalar < min:
        scalar = min
    elif max is not None and scalar > max:
        scalar = max
    
    return scalar



class Environment:

    def __init__(self, levels, max_storage_tank, optimum_storage, gamma1, gamma2, gamma3) -> None:
        self.levels = levels
        self.max_storage_tank = max_storage_tank
        self.optimum_storage = optimum_storage
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.gaussian_a = 0.05 #0.25     # the lower the parameter, the broader the gaussian curve
        self.cool_down = 0.1


    def step(self, s, a):
        '''
        action = element of [0; max_storage]
        s = ["excess", "demand_price", "feedin_price", "power_consumption_kwh", "thermal_consumption_kwh",  "kwh_eq_state"]
        '''
        
        s_0 = s[-1]
        thermal_consumption = s[-2]
        excess = s[0]

        if a == 1:
            a = excess

        s_1 = s_0 + a - thermal_consumption - self.cool_down
        s_1 = clip(s_1, min=0, max=self.max_storage_tank)
         
        return s_1


    def reward(self, action, s):

    
        excess, demand_price, feedin_price, thermal_consumption, state = s

        reward = 0

        max_increase = self.max_storage_tank - state


        availableExcess = clip((excess), min=0) 
        potentialHeat = clip(excess, max=max_increase)

        feedInAdvantage = clip((demand_price - feedin_price), min=0)
        

        '''
        FINANCIAL REWARD
        '''

        if action == 0:
            reward += (availableExcess * feedin_price)

        elif action == 1:
            reward += (availableExcess * feedInAdvantage)

        else: 
            print("undefined action!", action, s)


        reward = reward * self.gamma1

        

        '''
        REWARD FOR DISTANCE TO OPTIMUM
        '''

        # distance_to_optimum = abs(state - self.optimum_storage)
        # reward += torch.exp(-self.gaussian_a * distance_to_optimum**2)*self.gamma2

        reward_o = 0
        

        if state < self.optimum_storage and action == 1 and availableExcess > 0:
            reward_o += 1
        if state > self.optimum_storage and action == 0 and availableExcess > 0:
            reward_o += 1

        if availableExcess == 0 and action == 0:
            reward_o += 1


        distance_to_optimum = abs(state - self.optimum_storage)
        reward_o += torch.exp(-self.gaussian_a * distance_to_optimum**2) * 5

        reward += reward_o * self.gamma2


        
        return reward

