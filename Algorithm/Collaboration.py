from shareInformation import server_capacities, ACC, dnn_capacities, dnn_cost, Server_cap
import numpy as np
import os

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

DNN_dependence = np.zeros([3, 5], dtype=int)
DNN_dependence[0] = np.argsort(ACC[:, 0])[::-1] 
DNN_dependence[1] = np.argsort(ACC[:, 1])[::-1]
DNN_dependence[2] = np.argsort(ACC[:, 2])[::-1]

task_dependence = np.zeros([5, 3], dtype=int)
task_dependence[0] = np.argsort(ACC[0, :]) 
task_dependence[1] = np.argsort(ACC[1, :])
task_dependence[2] = np.argsort(ACC[2, :])
task_dependence[3] = np.argsort(ACC[3, :])
task_dependence[4] = np.argsort(ACC[4, :])

file_path = "/home/yourpath/CoMS/opensource" # your absolute path
Server_0_state =  np.loadtxt(os.path.join(file_path, "Result", "Server_0_state.txt"), dtype=np.float)
Server_1_state =  np.loadtxt(os.path.join(file_path, "Result", "Server_1_state.txt"), dtype=np.float)
Server_2_state =  np.loadtxt(os.path.join(file_path, "Result", "Server_2_state.txt"), dtype=np.float)
Server_3_state =  np.loadtxt(os.path.join(file_path, "Result", "Server_3_state.txt"), dtype=np.float)

Server_state = np.array([Server_0_state, Server_1_state, Server_2_state, Server_3_state])

total_avg_accLoss_sum = 0.0
total_avg_dnncost_sum = 0.0
total_tradeoff_accLoss_dnncost = 0.0

avg_accLoss_sum_list = []
avg_dnncost_sum_list = []
tradeoff_accLoss_dnncost_list = []

T = Server_0_state.shape[0]

for time_slot in range(0, T):
    HomeLessFlag = False 

    bool_ServerFlag = np.array([False, False, False, False]) 
    bool_ServerFlag[0] = np.any(Server_state[0, time_slot, 15:] < 0)
    bool_ServerFlag[1] = np.any(Server_state[1, time_slot, 15:] < 0)
    bool_ServerFlag[2] = np.any(Server_state[2, time_slot, 15:] < 0)
    bool_ServerFlag[3] = np.any(Server_state[3, time_slot, 15:] < 0)
    
    Utilization = np.zeros((4, 3))
    Utilization[0, :] = np.sum(Server_state[0, time_slot, 0:15].reshape(5, 3), axis=0)/server_capacities[0, :]
    Utilization[1, :] = np.sum(Server_state[1, time_slot, 0:15].reshape(5, 3), axis=0)/server_capacities[1, :]
    Utilization[2, :] = np.sum(Server_state[2, time_slot, 0:15].reshape(5, 3), axis=0)/server_capacities[2, :]
    Utilization[3, :] = np.sum(Server_state[3, time_slot, 0:15].reshape(5, 3), axis=0)/server_capacities[3, :]
    
    if np.any(bool_ServerFlag):
        HomeLessFlag = True

    if HomeLessFlag:
        while np.any(bool_ServerFlag):
            Id = np.where(bool_ServerFlag)[0]
            for Server_index in range(len(Id)):
                Server_state_DNN = Server_state[Id[Server_index], time_slot, 15:]
                IndexLessThan0 = np.where(Server_state_DNN < 0)[0]
                for DNN_index in range(len(IndexLessThan0)):
                    taskqueue = np.zeros((1,5))
                    IndexLessThan0[DNN_index]
                    Server_state_DNN[IndexLessThan0[DNN_index]] 
                    for task_index_sort in range(5): 
                        taskIndex = 3 * DNN_dependence[IndexLessThan0[DNN_index]][task_index_sort] + IndexLessThan0[DNN_index]
                        
                        if (Server_state[Id[Server_index], time_slot, taskIndex]) >= abs(Server_state_DNN[IndexLessThan0[DNN_index]]):
                            Server_state[Id[Server_index], time_slot, taskIndex] = Server_state[Id[Server_index], time_slot, taskIndex] - abs(Server_state_DNN[IndexLessThan0[DNN_index]])
                            taskqueue[0][DNN_dependence[IndexLessThan0[DNN_index]][task_index_sort]] = abs(Server_state_DNN[IndexLessThan0[DNN_index]])
                            Server_state_DNN[IndexLessThan0[DNN_index]] = 0
                            Server_state[Id[Server_index], time_slot, 15 + IndexLessThan0[DNN_index]] = Server_state_DNN[IndexLessThan0[DNN_index]]
                            break
                        else: 
                            Server_state_DNN[IndexLessThan0[DNN_index]] = (Server_state_DNN[IndexLessThan0[DNN_index]] + Server_state[Id[Server_index], time_slot, taskIndex])
                            taskqueue[0][DNN_dependence[IndexLessThan0[DNN_index]][task_index_sort]] = Server_state[Id[Server_index], time_slot, taskIndex]
                            Server_state[Id[Server_index], time_slot, taskIndex] = 0
                            
                    Utilization[0, :] = np.sum(Server_state[0, time_slot, 0:15].reshape(5, 3), axis=0)/server_capacities[0, :]
                    Utilization[1, :] = np.sum(Server_state[1, time_slot, 0:15].reshape(5, 3), axis=0)/server_capacities[1, :]
                    Utilization[2, :] = np.sum(Server_state[2, time_slot, 0:15].reshape(5, 3), axis=0)/server_capacities[2, :]
                    Utilization[3, :] = np.sum(Server_state[3, time_slot, 0:15].reshape(5, 3), axis=0)/server_capacities[3, :]
                    
                    for it_task_index in range(5): 
                        if taskqueue[0][it_task_index] != 0:
                            taskqueue[0][it_task_index]

                            for it_task_dependence in range(len(task_dependence[it_task_index])):
                                UtilizationFlag = True 
                                It_flag = True 
                                while UtilizationFlag:
                                    minUtilization = float('inf') 
                                    minIndex = 0
                                    for it_server in range(4):
                                        if Id[Server_index] != it_server:                                            
                                            if Utilization[it_server][task_dependence[it_task_index][it_task_dependence]] < minUtilization:                          
                                                minUtilization = Utilization[it_server][task_dependence[it_task_index][it_task_dependence]]
                                                minIndex = it_server
                                    
                                    if abs(1 - minUtilization) <= 0.000001:
                                        UtilizationFlag = False
                                    else:
                                        taskIndex = 3 * it_task_index + task_dependence[it_task_index][it_task_dependence]
                                        if Server_state[minIndex, time_slot, 15 + task_dependence[it_task_index][it_task_dependence]] >= taskqueue[0][it_task_index]:
                                            Server_state[minIndex, time_slot, taskIndex] += taskqueue[0][it_task_index]
                                            Server_state[minIndex, time_slot, 15 + task_dependence[it_task_index][it_task_dependence]] -= taskqueue[0][it_task_index]
                                            taskqueue[0][it_task_index] = 0
                                            UtilizationFlag = False
                                            It_flag = False
                                        else:
                                            taskqueue[0][it_task_index] = taskqueue[0][it_task_index] - Server_state[minIndex, time_slot, 15 + task_dependence[it_task_index][it_task_dependence]]
                                            Server_state[minIndex, time_slot, taskIndex] = Server_state[minIndex, time_slot, taskIndex] + Server_state[minIndex, time_slot, 15 + task_dependence[it_task_index][it_task_dependence]]
                                            Server_state[minIndex, time_slot, 15 + task_dependence[it_task_index][it_task_dependence]] = 0
                                        Utilization[minIndex, :] = np.sum(Server_state[minIndex, time_slot, 0:15].reshape(5, 3), axis=0)/server_capacities[minIndex, :]
                                        
                                if It_flag == False:
                                   break
                        if np.sum(taskqueue) == 0:
                            break
            bool_ServerFlag[0] = np.any(Server_state[0, time_slot, 15:] < 0)
            bool_ServerFlag[1] = np.any(Server_state[1, time_slot, 15:] < 0)
            bool_ServerFlag[2] = np.any(Server_state[2, time_slot, 15:] < 0)
            bool_ServerFlag[3] = np.any(Server_state[3, time_slot, 15:] < 0)
    
    Server_0_DNN =np.sum(Server_state[0, time_slot, 0:15].reshape(5, 3), axis=0)
    Server_0_DNN_Sum = np.ceil(Server_0_DNN/dnn_capacities)
    Utilization_Server0 = Server_0_DNN_Sum/Server_cap[0, :]

    Server_1_DNN =np.sum(Server_state[1, time_slot, 0:15].reshape(5, 3), axis=0)
    Server_1_DNN_Sum = np.ceil(Server_1_DNN/dnn_capacities)
    Utilization_Server1 = Server_1_DNN_Sum/Server_cap[1, :]

    Server_2_DNN =np.sum(Server_state[2, time_slot, 0:15].reshape(5, 3), axis=0)
    Server_2_DNN_Sum = np.ceil(Server_2_DNN/dnn_capacities)
    Utilization_Server2 = Server_2_DNN_Sum/Server_cap[2, :]
    
    Server_3_DNN =np.sum(Server_state[3, time_slot, 0:15].reshape(5, 3), axis=0)
    Server_3_DNN_Sum = np.ceil(Server_3_DNN/dnn_capacities)
    Utilization_Server3 = Server_3_DNN_Sum/Server_cap[3, :]

    Server_DNN_Sum = Server_0_DNN_Sum + Server_1_DNN_Sum + Server_2_DNN_Sum + Server_3_DNN_Sum
    Server_sum_DNN = Server_0_DNN + Server_1_DNN + Server_2_DNN + Server_3_DNN
    
    DNN_Sum = np.ceil(Server_sum_DNN/dnn_capacities)
    
    Server_cap_sum = np.sum(Server_cap, axis=0)
    DNN_number = np.floor(DNN_Sum * Server_cap/Server_cap_sum)
    DNN_number_sum = np.sum(DNN_number, axis=0)
    DNN_number_remain = DNN_Sum - DNN_number_sum
    if sum(DNN_number_remain) != 0:
        for dnn_i in range(3): 
            while DNN_number_remain[dnn_i]:
                minU = 1
                minIndex = 0
                for server_i in range(4):
                    if (DNN_number[server_i][dnn_i])/Server_cap[server_i][dnn_i] <= minU:
                        minU = (DNN_number[server_i][dnn_i])/Server_cap[server_i][dnn_i]
                        minIndex = server_i
                DNN_number_remain[dnn_i] -=1
                DNN_number[minIndex][dnn_i] += 1
    Utilization_Server = DNN_number/Server_cap
    a = 1  
    b = 2 
    x = Utilization_Server
    f = a * np.exp(b * x)
    final_DNNCost = f * (dnn_cost * DNN_number)
    DNN_number_sum = np.sum(DNN_number, axis=0)
    
    accLoss_sum = 0.0
    for Server_id in range(4):
        for n_type in range(5):
            accLoss_0 = ACC[n_type][0]*Server_state[Server_id, time_slot, 3*n_type + 0]
            accLoss_1 = ACC[n_type][1]*Server_state[Server_id, time_slot, 3*n_type + 1]
            accLoss_2 = ACC[n_type][2]*Server_state[Server_id, time_slot, 3*n_type + 2]

            accLoss_sum += (accLoss_0 + accLoss_1 + accLoss_2) 
    
    avg_accLoss_sum = accLoss_sum/np.sum(Server_0_DNN + Server_1_DNN + Server_2_DNN + Server_3_DNN)
    avg_dnncost_sum = np.sum(final_DNNCost)/np.sum(Server_0_DNN + Server_1_DNN + Server_2_DNN + Server_3_DNN)
   
    normalizer_accLoss.update(avg_accLoss_sum)
    norm_accLoss_sum = normalizer_accLoss.normalize(avg_accLoss_sum)
    
    normalizer_dnncost.update(avg_dnncost_sum)
    norm_dnncost_sum = normalizer_dnncost.normalize(avg_dnncost_sum)
    
    tradeoff_accLoss_dnncost = norm_accLoss_sum + norm_dnncost_sum
    
    total_avg_dnncost_sum += norm_dnncost_sum
    total_avg_accLoss_sum += norm_accLoss_sum
    total_tradeoff_accLoss_dnncost += tradeoff_accLoss_dnncost

    avg_accLoss_sum_list.append(norm_accLoss_sum)
    avg_dnncost_sum_list.append(norm_dnncost_sum)
    tradeoff_accLoss_dnncost_list.append(tradeoff_accLoss_dnncost)

print("total_avg_dnncost_sum =", total_avg_dnncost_sum)
print("total_avg_accLoss_sum =", total_avg_accLoss_sum)
print("total_tradeoff_accLoss_dnncost =", total_tradeoff_accLoss_dnncost)


path0 = f'/home/yourpath/CoMS/opensource/Result/avg_dnncost_sum_list.txt'
path1 = f'/home/yourpath/CoMS/opensource/Result/avg_accLoss_sum_list.txt'
path2 = f'/home/yourpath/CoMS/opensource/Result/tradeoff_accLoss_dnncost_list.txt'

np.savetxt(path0, avg_dnncost_sum_list, fmt='%0.4f')
np.savetxt(path1, avg_accLoss_sum_list, fmt='%0.4f')
np.savetxt(path2, tradeoff_accLoss_dnncost_list, fmt='%0.4f')
