import numpy as np

Server_cap = np.array([[40, 30, 20], 
                       [39, 30, 43], 
                       [30, 50, 10],
                       [50, 100, 20]])

dnn_capacities = np.array([180, 200, 190])

server_capacities = dnn_capacities * Server_cap

dnn_cost = np.array([10, 20, 30]) 

# accloss
ACC = np.array([[0.287, 0.206, 0.242],
       [0.329, 0.239, 0.272],
       [0.300, 0.206, 0.267],
       [0.281, 0.199, 0.251],
       [0.216, 0.130, 0.167]])