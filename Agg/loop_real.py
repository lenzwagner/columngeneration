from setup import Min_WD_i, Max_WD_i
from cg_naive import *
from cg_behavior import *
from subproblem import *
from demand import *
from datetime import datetime
import os

# **** Prerequisites ****
# Create Dataframes
eps_ls = [0.025, 0.05, 0.1]
chi_ls = [7]
T = list(range(1, 29))
I = list(range(1, 101))
K = [1, 2, 3]

# DataFrame
results = pd.DataFrame(columns=['epsilon', 'chi', 'undercover', 'undercover_norm', 'cons', 'cons_norm', 'perf', 'perf_norm', 'undercover_n', 'undercover_norm_n', 'cons_n', 'cons_norm_n', 'perf_n', 'perf_norm_n'])

# Times and Parameter
time_Limit = 7200
time_cg = 7200
time_cg_init = 60
time_cg_init_npm = 5

# Datanames
current_time = datetime.now().strftime('%Y-%m-%d_%H')
file = f'study_results_mulit_new_{current_time}'
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
        undercoverage_n, understaffing_n, perfloss_n, consistency_n, consistency_norm_n, undercoverage_norm_n, understaffing_norm_n, perfloss_norm_n, results_sc_n, results_r_n, gini_sc_n, gini_r_n = column_generation_naive(data, demand_dict, 0, Min_WD_i, Max_WD_i, time_cg_init_npm, max_itr, output_len, chi,
                                    threshold, time_cg, I, T, K, eps)

        undercoverage, understaffing, perfloss, consistency, consistency_norm, undercoverage_norm, understaffing_norm, perfloss_norm, results_sc, results_r, gini_sc, gini_r = column_generation_behavior(data, demand_dict, eps, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi,
                                    threshold, time_cg, I, T, K)

        # Data frame
        result = pd.DataFrame([{
            'epsilon': epsilon,
            'chi': chi,
            'undercover': undercoverage,
            'undercover_norm': undercoverage_norm,
            'cons': consistency,
            'cons_norm': consistency_norm,
            'perf': perfloss,
            'perf_norm': perfloss_norm,
            'undercover_n': undercoverage_n,
            'undercover_norm_n': undercoverage_norm_n,
            'cons_n': consistency_n,
            'cons_norm_n': consistency_norm_n,
            'perf_n': perfloss_n,
            'perf_norm_n': perfloss_norm_n
        }])

        results = pd.concat([results, result], ignore_index=True)

results.to_csv(file_name_csv, index=False)
results.to_excel(file_name_xlsx, index=False)

