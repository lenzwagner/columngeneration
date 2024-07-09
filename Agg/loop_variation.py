from setup import Min_WD_i, Max_WD_i
from cg_naive import *
from cg_behavior import *
from subproblem import *
from demand import *
from datetime import datetime
import os

# **** Prerequisites ****
# Create Dataframes
eps_ls = [0.025]
chi_ls = [3]
T = list(range(1, 29))
I = list(range(1, 101))
K = [1, 2, 3]

# DataFrame
results = pd.DataFrame(columns=['epsilon', 'chi', 'mean_r', 'min_r', 'max_r', 'std_r', 'mean_sc', 'min_sc', 'max_sc', 'std_sc', 'mean_r_n', 'min_r_n', 'max_r_n', 'std_r_n', 'mean_sc_n', 'min_sc_n', 'max_sc_n', 'std_sc_n'])


# Times and Parameter
time_Limit = 7200
time_cg = 7200
time_cg_init = 10
time_cg_init_npm = 4

# Datanames
current_time = datetime.now().strftime('%Y-%m-%d_%H')
file = f'study_results_variation_{current_time}'
file_name_csv = f'.{os.sep}results{os.sep}study{os.sep}{file}.csv'

# Loop
for epsilon in eps_ls:
    for chi in chi_ls:

        eps = epsilon
        print(f"")
        print(f"Iteration: {epsilon}-{chi}")
        print(f"")

        seed1 = 123 - math.floor(len(I)*len(T)*1.0)
        print(seed1)
        random.seed(seed1)
        demand_dict = demand_dict_fifty(len(T), 1.0, len(I), 2, 0.25)
        print(demand_dict)
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
        understaffing_n, u_results_n, sum_all_doctors_n, consistency_n, consistency_norm_n, understaffing_norm_n, u_results_norm_n, sum_all_doctors_norm_n, results_sc_n, results_r_n = column_generation_naive(data, demand_dict, 0, Min_WD_i, Max_WD_i, time_cg_init_npm, max_itr, output_len, chi,
                                    threshold, time_cg, I, T, K, eps)

        print(f"Res: {results_sc_n}")
        print(f"Res1: {results_sc_n[0]}")


        understaffing, u_results, sum_all_doctors, consistency, consistency_norm, understaffing_norm, u_results_norm, sum_all_doctors_norm, results_sc, results_r = column_generation_behavior(data, demand_dict, eps, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi,
                                    threshold, time_cg, I, T, K)

        # Data frame
        result = pd.DataFrame([{
            'epsilon': epsilon,
            'chi': chi,
            'mean_r': results_r[0],
            'min_r': results_r[1],
            'max_r': results_r[2],
            'std_r': results_r[3],
            'mean_sc': results_sc[0],
            'min_sc': results_sc[1],
            'max_sc': results_sc[2],
            'std_sc': results_sc[3],
            'mean_r_n': results_r_n[0],
            'min_r_n': results_r_n[1],
            'max_r_n': results_r_n[2],
            'std_r_n': results_r_n[3],
            'mean_sc_n': results_sc_n[0],
            'min_sc_n': results_sc_n[1],
            'max_sc_n': results_sc_n[2],
            'std_sc_n': results_sc_n[3]
        }])

        results = pd.concat([results, result], ignore_index=True)

results.to_csv(file_name_csv, index=False)
