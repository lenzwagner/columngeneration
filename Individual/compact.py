from compactsolver import *
import time
import random
from setup import *
import numpy as np
from plots import optimality_plot
from test import *
from demand import *

I, T, K = list(range(1,101)), list(range(1,29)), [1, 2, 3]

# **** Solve ****
data = pd.DataFrame({
    'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
    'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
    'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
})

# Remove unused files
for file in os.listdir():
    if file.endswith('.lp') or file.endswith('.sol') or file.endswith('.log'):
        os.remove(file)

# Demand
random.seed(-2677)
demand_dict = {(1, 1): 1, (1, 2): 86, (1, 3): 0, (2, 1): 27, (2, 2): 71, (2, 3): 3, (3, 1): 1, (3, 2): 57, (3, 3): 53, (4, 1): 26, (4, 2): 53, (4, 3): 32, (5, 1): 0, (5, 2): 47, (5, 3): 28, (6, 1): 2, (6, 2): 73, (6, 3): 49, (7, 1): 50, (7, 2): 51, (7, 3): 1, (8, 1): 55, (8, 2): 29, (8, 3): 24, (9, 1): 25, (9, 2): 21, (9, 3): 57, (10, 1): 28, (10, 2): 2, (10, 3): 71, (11, 1): 5, (11, 2): 83, (11, 3): 1, (12, 1): 32, (12, 2): 78, (12, 3): 14, (13, 1): 2, (13, 2): 103, (13, 3): 17, (14, 1): 7, (14, 2): 65, (14, 3): 34, (15, 1): 7, (15, 2): 47, (15, 3): 46, (16, 1): 23, (16, 2): 50, (16, 3): 22, (17, 1): 5, (17, 2): 40, (17, 3): 45, (18, 1): 7, (18, 2): 97, (18, 3): 3, (19, 1): 20, (19, 2): 87, (19, 3): 2, (20, 1): 39, (20, 2): 24, (20, 3): 18, (21, 1): 3, (21, 2): 83, (21, 3): 10, (22, 1): 60, (22, 2): 13, (22, 3): 36, (23, 1): 9, (23, 2): 105, (23, 3): 4, (24, 1): 100, (24, 2): 2, (24, 3): 2, (25, 1): 24, (25, 2): 47, (25, 3): 28, (26, 1): 17, (26, 2): 37, (26, 3): 46, (27, 1): 34, (27, 2): 45, (27, 3): 14, (28, 1): 4, (28, 2): 18, (28, 3): 89}


# Parameter
time_Limit = 3600
max_itr = 30
output_len = 98
mue = 1e-4
eps = 0

# **** Compact Solver ****
problem_t0 = time.time()
problem = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i)
problem.buildLinModel()
problem.updateModel()
#problem.model.Params.LogFile = "./test.log"
problem.model.Params.TimeLimit = time_Limit
problem.model.optimize()

### Logs
#file='./test.log'
#results, timeline = glt.get_dataframe([file], timelines=True)
# Plot
#default_run = timeline["nodelog"]
#print(default_run)
#plot(default_run, 3600, 'opt_pl')