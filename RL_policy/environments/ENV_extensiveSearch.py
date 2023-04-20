def clip(scalar, min=None, max=None):       # way faster than numpy array.clip()
    if min and scalar < min:
        return min
    elif max and scalar > max:
        return max
    else:
        return scalar



class Environment:

    def __init__(self, levels, max_storage_tank, optimum_storage, gamma1, gamma2, gamma3) -> None:
        self.levels = levels
        self.max_storage_tank = max_storage_tank
        self.optimum_storage = optimum_storage
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.cool_down = 0.1
        
        
    def step(self, a, s, thermal_consumption):
        '''
        action = [0; max_storage]
        '''

        s_1 = s + a - thermal_consumption - self.cool_down
        s_1 = clip(s_1, min=0, max=self.max_storage_tank)
            
        return s_1


    def reward(self, action, pv_excess, demand_price, feedin_price, power_consumption, thermal_consumption, state):

        reward = 0

        max_increase = self.max_storage_tank - state
        action = clip(action, max=max_increase)


        consumption = power_consumption   
        availableExcess = clip((pv_excess - consumption), min=0) 
        feedInAdvantage = clip((demand_price - feedin_price), min=0)
        
        
        '''
        FINANCIAL REWARD
        '''

        if action == 0:
            reward += (availableExcess * feedin_price)

        else:
            if availableExcess > 0: # no excess power
                reward += (availableExcess * feedInAdvantage)

        reward = reward * self.gamma1

        
        '''
        PENALTY FOR DISTANCE TO OPTIMUM
        '''
        reward -= abs(state - self.optimum_storage)*self.gamma2
        
        
        return reward