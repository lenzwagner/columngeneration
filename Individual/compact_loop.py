import pandas as pd
import time
import os
from compactsolver import *
from setup import *
from plots import optimality_plot
from test import *
from demand import *
import numpy as np

# Parameterdefinitionen
I_values = [3, 4]
prob_values = [1.0, 1.1, 1.2]
patterns = [1, 2, 3, 4]
T = list(range(1, 14))
K = [1, 2, 3]

prob_mapping = {1.0: 'Low', 1.1: 'Medium', 1.2: 'High'}
pattern_mapping = {1: 'Split_Early', 2: 'Split_Noon', 3: 'Split_Evening', 4: 'Equal'}

# Ergebnisse DataFrame initialisieren
results = pd.DataFrame(columns=['I', 'T', 'K', 'prob', 'pattern', 'time', 'gap', 'lb', 'ub', 'obj'])

time_Limit = 3600
eps = 0.05

for I_len in I_values:
    I = list(range(1, I_len + 1))
    for prob in prob_values:
        for pattern in patterns:
            if pattern == 4:
                demand_dict = demand_dict_third(len(T), prob, len(I))
            else:
                demand_dict = demand_dict_fifty(len(T), prob, len(I), pattern)

            data = pd.DataFrame({
                'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
                'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
                'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
            })

            problem = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i)
            problem.buildLinModel()
            problem.updateModel()
            problem.model.Params.TimeLimit = time_Limit

            problem_t0 = time.time()
            problem.model.optimize()
            problem_t1 = time.time()

            runtime = round(problem_t1 - problem_t0, 1)
            mip_gap = round(problem.model.MIPGap, 2)
            lower_bound = round(problem.model.ObjBound, 2)
            upper_bound = round(problem.model.ObjVal, 2)
            objective_value = round(problem.model.ObjVal, 2)

            result = pd.DataFrame([{
                'I': I_len,
                'T': len(T),
                'K': len(K),
                'prob': prob_mapping[prob],
                'pattern': pattern_mapping[pattern],
                'time': runtime,
                'gap': mip_gap,
                'lb': lower_bound,
                'ub': upper_bound,
                'obj': objective_value
            }])

            results = pd.concat([results, result], ignore_index=True)

# Ergebnisse in eine CSV-Datei speichern
results.to_csv('compact.csv', index=False)