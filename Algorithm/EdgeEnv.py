import gym
import math
import numpy as np
from gym import spaces
from shareInformation import server_capacities, ACC, dnn_capacities, dnn_cost, Server_cap

class RealTimeNormalizer:
    def __init__(self):
        self.min = float('inf')
        self.max = float('-inf')

    def update(self, new_value):
        self.min = min(self.min, new_value)
        self.max = max(self.max, new_value)

    def normalize(self, value):
        if self.max == self.min:
            return 0 
        return (value - self.min) / (self.max - self.min)
    
normalizer_accLoss = RealTimeNormalizer()
normalizer_dnncost = RealTimeNormalizer()

class EdgeEnv:
    def __init__(self):
        self.n_tasktype = 5 
        self.n_dnntype = 3  
        self.server_capacities = server_capacities 
        self.state = None            
        self.task_allocation = None  
        
        self.min_action = -1.0
        self.max_action = 1.0
        self.high_state = np.finfo(np.float32).max 

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(15,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low= -self.high_state, high=self.high_state, shape=(18,), dtype=np.float32
        )
    def reset(self):
        import random
        self.state = np.zeros(self.n_tasktype * self.n_dnntype, dtype=float)
        
        for n_type in range(self.n_tasktype):
            point1 = random.uniform(0, 1)
            point2 = random.uniform(0, 1)

            point1, point2 = min(point1, point2), max(point1, point2)

            num1 = point1
            self.state[self.n_dnntype*n_type] = num1
            
            num2 = point2 - point1
            self.state[self.n_dnntype*n_type + 1] = num2
            
            num3 = 1 - point2
            self.state[self.n_dnntype*n_type + 2] = num3

        return self.state, self.server_capacities

    def step(self, action, task_num, avg_reward, EdgeServerID):
        bool_flag = action > 0

        action_value = abs(action)
        action_value[action_value > 1] = 1
        res_action_value = 1 - action_value

        temp_state_trans = self.state * action_value
        temp_state_local = self.state * res_action_value
        
        temp_state_0 = np.zeros(self.n_tasktype * self.n_dnntype, dtype=float) # 
        temp_state_1 = np.zeros(self.n_tasktype * self.n_dnntype, dtype=float) # 
        temp_state_2 = np.zeros(self.n_tasktype * self.n_dnntype, dtype=float) # 
    
        
        for n_type in range(self.n_tasktype):

            if bool_flag[self.n_dnntype*n_type + 0]: 
                temp_state_0[self.n_dnntype*n_type + 1] = temp_state_trans[self.n_dnntype*n_type + 0]
            else: 
                temp_state_0[self.n_dnntype*n_type + 2] = temp_state_trans[self.n_dnntype*n_type + 0]

            if bool_flag[self.n_dnntype*n_type + 1]:
                temp_state_1[self.n_dnntype*n_type + 2] = temp_state_trans[self.n_dnntype*n_type + 1]
            else:
                temp_state_1[self.n_dnntype*n_type + 0] = temp_state_trans[self.n_dnntype*n_type + 1]

            if bool_flag[self.n_dnntype*n_type + 2]:
                temp_state_2[self.n_dnntype*n_type + 0] = temp_state_trans[self.n_dnntype*n_type + 2]
            else:
                temp_state_2[self.n_dnntype*n_type + 1] = temp_state_trans[self.n_dnntype*n_type + 2]

        self.state = temp_state_local + temp_state_0 + temp_state_1 + temp_state_2
    
        accLoss_sum = 0.0
        for n_type in range(self.n_tasktype):
            accLoss_0 = ACC[n_type][0]*self.state[self.n_dnntype*n_type + 0]*task_num[n_type]

            accLoss_1 = ACC[n_type][1]*self.state[self.n_dnntype*n_type + 1]*task_num[n_type]

            accLoss_2 = ACC[n_type][2]*self.state[self.n_dnntype*n_type + 2]*task_num[n_type]
            
            accLoss_sum += (accLoss_0 + accLoss_1 + accLoss_2)
        avg_accLoss_sum = accLoss_sum/sum(task_num)
       
        dnn_cost_sum = 0.0
        Utilization_value = np.zeros(3, dtype=float)
        for n_type in range(self.n_dnntype):
            task_dnn_num_0 = self.state[n_type + self.n_dnntype * 0]*task_num[0]
            task_dnn_num_1 = self.state[n_type + self.n_dnntype * 1]*task_num[1]
            task_dnn_num_2 = self.state[n_type + self.n_dnntype * 2]*task_num[2]
            task_dnn_num_3 = self.state[n_type + self.n_dnntype * 3]*task_num[3]
            task_dnn_num_4 = self.state[n_type + self.n_dnntype * 4]*task_num[4]

            task_dnn_num_sum = task_dnn_num_0 + task_dnn_num_1 + task_dnn_num_2 + task_dnn_num_3 + task_dnn_num_4
            
            a = 1  
            b = 2 
            x = math.ceil(task_dnn_num_sum/dnn_capacities[n_type])/Server_cap[EdgeServerID][n_type]
            Utilization_value[n_type] = x
            x = min(x, 1) 
            f = a * np.exp(b * x)
            
            dnn_cost_sum += dnn_cost[n_type] * math.ceil(task_dnn_num_sum/dnn_capacities[n_type]) * f
        avg_dnn_cost_sum = dnn_cost_sum/sum(task_num)

        normalizer_accLoss.update(avg_accLoss_sum)
        norm_accLoss_sum = normalizer_accLoss.normalize(avg_accLoss_sum)
        
        normalizer_dnncost.update(avg_dnn_cost_sum)
        norm_dnncost_sum = normalizer_dnncost.normalize(avg_dnn_cost_sum)
        
        self.reward = -(norm_accLoss_sum + norm_dnncost_sum)
       
        has_non_positive = bool(abs(avg_reward) < abs(self.reward))
        done = has_non_positive 

        ratio = self.state
        ratio_reshaped = ratio.reshape(5, 3)

        state = (task_num[:, np.newaxis] * ratio_reshaped)
        consumedCap = np.sum(state, axis=0)
        remainCap = self.server_capacities[EdgeServerID] - consumedCap
    
        state = state.reshape(1, 15)[0]
        return state, self.reward, done, remainCap

    def close(self):
        return None

