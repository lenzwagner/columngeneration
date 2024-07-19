from setup import Min_WD_i, Max_WD_i
from cg_naive import *
from cg_behavior import *
from subproblem import *
from demand import *
from datetime import datetime
import os
import pandas as pd

# **** Prerequisites ****
# Create Dataframes
eps_ls = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
chi_ls = [2, 3, 4, 5, 6, 7, 8, 9, 10]
T = list(range(1, 29))
I = list(range(1, 101))
K = [1, 2, 3]

# DataFrame
results = pd.DataFrame(columns=['I', 'pattern', 'epsilon', 'chi', 'objval', 'lbound', 'iteration', 'undercover', 'undercover_norm', 'cons', 'cons_norm', 'perf', 'perf_norm', 'max_auto', 'min_auto', 'mean_auto', 'lagrange'])

# Times and Parameter
time_Limit = 7200
time_cg = 7200
time_cg_init = 60

# Datanames
current_time = datetime.now().strftime('%Y-%m-%d_%H')
file = f'study_results_mulit_bam_{current_time}'
file_name_csv = f'.{os.sep}results{os.sep}study{os.sep}{file}.csv'
file_name_xlsx = f'.{os.sep}results{os.sep}study{os.sep}{file}.xlsx'


# Loop
for epsilon in eps_ls:
    for chi in chi_ls:

        eps = epsilon
        print(f"")
        print(f"Iteration: {epsilon}-{chi}")
        print(f"")

        seed1 = 123 - math.floor(len(I)*len(T))
        print(seed1)
        random.seed(seed1)
        demand_dict = demand_dict_fifty(len(T), 1, len(I), 2, 0.25)
        max_itr = 200
        output_len = 98
        mue = 1e-4
        threshold = 4e-5

        data = pd.DataFrame({
            'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
            'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
            'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
        })

        # Column Generation
        undercoverage, understaffing, perfloss, consistency, consistency_norm, undercoverage_norm, understaffing_norm, perfloss_norm, results_sc, results_r, autocorell, final_obj, final_lb, itr, lagrangeB = column_generation_behavior(data, demand_dict, eps, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi,
                                    threshold, time_cg, I, T, K)

        # Data frame
        result = pd.DataFrame([{
            'I': len(I),
            'pattern': "Medium",
            'epsilon': epsilon,
            'chi': chi,
            'objval': final_obj,
            'lbound': final_lb,
            'iteration': itr,
            'undercover': undercoverage,
            'undercover_norm': undercoverage_norm,
            'cons': consistency,
            'cons_norm': consistency_norm,
            'perf': perfloss,
            'perf_norm': perfloss_norm,
            'max_auto': round(max(autocorell), 5),
            'min_auto': round(min(autocorell), 5),
            'mean_auto': round(np.mean(autocorell), 5),
            'lagrange': lagrangeB
        }])

        results = pd.concat([results, result], ignore_index=True)

results.to_csv(file_name_csv, index=False)
results.to_excel(file_name_xlsx, index=False)