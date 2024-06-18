from masterproblem import *
import time
from setup import Min_WD_i, Max_WD_i
from gcutil import *
from subproblem import *
from compactsolver import Problem
from demand import *
import os

# **** Prerequisites ****
# Create Dataframes
I_values = [50, 100]
prob_values = [1.0]
patterns = [2]
T = list(range(1, 29))
K = [1, 2, 3]

prob_mapping = {1.0: 'Low', 1.1: 'Medium', 1.2: 'High'}
pattern_mapping = {2: 'Noon'}

# Ergebnisse DataFrame initialisieren
results = pd.DataFrame(columns=['I', 'T', 'K', 'prob', 'pattern', 'time', 'gap', 'lb', 'ub', 'obj'])

time_Limit = 7200
eps = 0.1

## Dataframe

# Parameter
for I_len in I_values:
    I = list(range(1, I_len + 1))
    for prob in prob_values:
        for pattern in patterns:
            if pattern == 4:
                demand_dict = demand_dict_third(len(T), prob, len(I))
            else:
                demand_dict = demand_dict_fifty2(len(T), prob, len(I), pattern, 0.25)

            seed = 12

            max_itr = 20
            output_len = 98
            mue = 1e-4
            threshold = 5e-7
            eps = 0.1

            # Demand Dict
            demand_dict = demand_dict_fifty2(len(T), 1, len(I), 2, 0.1)

            data = pd.DataFrame({
                'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
                'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
                'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
            })

            # **** Compact Solver ****
            problem_t0 = time.time()
            problem = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i)
            problem.buildLinModel()
            problem.model.Params.TimeLimit = time_Limit
            problem.updateModel()
            problem_t0 = time.time()
            problem.solveModel()
            problem_t1 = time.time()

            bound = problem.model.ObjBound
            print(f"Bound {bound}")

            obj_val_problem = round(problem.model.objval, 3)
            time_problem = time.time() - problem_t0
            vals_prob = problem.get_final_values()


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





results.to_csv('compact.csv', index=False)
results.to_excel('compact.xlsx', index=False)
