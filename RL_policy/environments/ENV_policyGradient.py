class Environment:

    def __init__(self, levels, max_storage_tank, optimum_storage, gamma1, gamma2, gamma3) -> None:
        self.levels = levels
        self.max_storage_tank = max_storage_tank
        self.optimum_storage = optimum_storage
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.cool_down = 0.1


    def step(self, s, s_new, a, exploit=True):
        '''
        action = element of [0; max_storage]
        s = ["i_m1sum" , "demand_price", "feedin_price", "power_consumption_kwh", "thermal_consumption_kwh",  "kwh_eq_state"]
        '''

        state = s[-1]
        thermal_consumption = s[-2]
        power_consumption = s[-3]
        pv_excess = s[0]
        
        if a == 1 and exploit:
            a = pv_excess - power_consumption - thermal_consumption 
            a = a.clip(min=0)

        heat_increase = a
        
        s_1 = state + heat_increase - thermal_consumption - self.cool_down
        s_1 = s_1.clip(min=0, max=self.max_storage_tank)


        s_next = s_new
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
        
        consumption = power_consumption    # not thermal consumption because heat will be fed in via the action  
        availableExcess = (pv_excess - consumption).clip(min=0) 
        feedInAdvantage = (demand_price - feedin_price).clip(min=0)

        reward = 0

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


