from compactsolver import *
import time
import random
from setup import *
import numpy as np
from plots import optimality_plot
from test import *
from demand import *

I, T, K = list(range(1,15)), list(range(1,15)), [1, 2, 3]

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
demand_dict = demand_dict_fifty(len(T), 1.0, len(I), 2, 0.25)

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

a = problem.model.getAttr("X", problem.sc)
print(f"c: {a}")
b = problem.model.getAttr("X", problem.x)
print(f"x: {b}")