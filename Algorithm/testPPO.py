import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from PPO import PPO
file_path = "/home/yourpath/CoMS" # your absolute path
from EdgeEnv import EdgeEnv

class RealTimeMinMaxNormalizer:
    def __init__(self):
        self.min = float('inf')
        self.max = float('-inf')

    def update(self, new_data):
        self.min = min(self.min, np.min(new_data))
        self.max = max(self.max, np.max(new_data))

    def normalize(self, data):
        if self.max == self.min:
            return np.zeros(data.shape) 
        return 2 * (data - self.min) / (self.max - self.min) - 1
    

#################################### Testing ###################################
def test():
    print("============================================================================================")
    env_name = 'EdgeEnv'
    has_continuous_action_space = True
    
    max_ep_len = 100 # 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################
    env = EdgeEnv()

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory
    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num
    
    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    Req_pd = np.load(os.path.join(file_path,"OriginData", "PEMSD8.npz"))
    flow = Req_pd['truth'][:, 0, :, 0]
    flow = np.transpose(flow).astype('int')

    flow_train = flow[:,:2252] # 80%
    flow_test = flow[:,2252:] # 20%
    # key index
    OptimalPlacement =  np.loadtxt(os.path.join(file_path, "Dataset", "AdaptiveBestDNA_pesm08.txt"), dtype=np.int)
    typefortask =  np.loadtxt(os.path.join(file_path, "Dataset", "typefortask.txt"), dtype=np.int)

    OptimalPlacement[OptimalPlacement == 4] = 3
    
    max_training_timesteps = flow_test.shape[1]*4
    Server_0_State = []
    Server_1_State = []
    Server_2_State = []
    Server_3_State = []

    time_step_while = 0
    while time_step_while < max_training_timesteps:
        Place_indices3 = np.where(OptimalPlacement == int(time_step_while/flow_test.shape[1]))[0]
        EdgeServerID = int(time_step_while/flow_test.shape[1]) 
        temp_typefortask = typefortask[Place_indices3]
        temp_flow = flow_test[Place_indices3, time_step_while%flow_test.shape[1]]
        temp_task_num = np.zeros(5, dtype=int)
        for task_item in range(5):
            type_indices = np.where(temp_typefortask == task_item)[0]
            temp_task_num[task_item] = sum(temp_flow[type_indices])
        
        # init
        ratio, server_capacities = env.reset()
        ratio_reshaped = ratio.reshape(5, 3)
        state = (temp_task_num[:, np.newaxis] * ratio_reshaped)
        consumedCap = np.sum(state, axis=0)
        remainCap = server_capacities[EdgeServerID] - consumedCap
        state = state.reshape(1, 15)[0]
        ep_reward = 0
        avg_current_ep_reward = float('inf') 
        min_reward = float('inf') 
        normalizer_state = RealTimeMinMaxNormalizer()
        normalizer_remainCap = RealTimeMinMaxNormalizer()

        for t in range(1, max_ep_len+1):

            normalizer_state.update(state)
            state = normalizer_state.normalize(state)
            
            normalizer_remainCap.update(remainCap)
            remainCap = normalizer_remainCap.normalize(remainCap)

            state = np.concatenate((state, remainCap))

            action = ppo_agent.select_action(state)

            state, reward, done, remainCap = env.step(action, temp_task_num, avg_current_ep_reward, EdgeServerID)
            
            if min_reward > abs(reward):
                min_reward = abs(reward)
                goal_state = np.concatenate((state, remainCap))
            ep_reward += reward
            
            if t == 1:
                avg_current_ep_reward = reward 
            if done:
                break
        
        if int(time_step_while/flow_test.shape[1]) == 0: 
            Server_0_State.append(goal_state)
           
        elif int(time_step_while/flow_test.shape[1]) == 1:
            Server_1_State.append(goal_state)
            
        elif int(time_step_while/flow_test.shape[1]) == 2:
            Server_2_State.append(goal_state)
           
        elif int(time_step_while/flow_test.shape[1]) == 3:
            Server_3_State.append(goal_state)
            
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(time_step_while, round(ep_reward, 2)))
        ep_reward = 0
        time_step_while += 1

    path0 = f'/home/yourpath/Result/Server_0_state.txt'
    path1 = f'/home/yourpath/Result/Server_1_state.txt'
    path2 = f'/home/yourpath/Result/Server_2_state.txt'
    path3 = f'/home/yourpath/Result/Server_3_state.txt'
    np.savetxt(path0, Server_0_State, fmt='%0.4f')
    np.savetxt(path1, Server_1_State, fmt='%0.4f')
    np.savetxt(path2, Server_2_State, fmt='%0.4f')
    np.savetxt(path3, Server_3_State, fmt='%0.4f')
    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()