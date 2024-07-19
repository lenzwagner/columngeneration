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


# DataFrame
results = pd.DataFrame(columns=['I', 'pattern', 'epsilon', 'chi', 'undercover_n', 'undercover_norm_n', 'cons_n', 'cons_norm_n', 'perf_n', 'perf_norm_n', 'max_auto_n', 'min_auto_n', 'mean_auto_n', 'lagrange_n'])

# Times and Parameter
time_Limit = 7200
time_cg = 7200
time_cg_init = 60
time_cg_init_npm = 30

# Datanames
current_time = datetime.now().strftime('%Y-%m-%d_%H')
file = f'study_results_mulit_npm_{current_time}'
file_name_csv = f'.{os.sep}results{os.sep}study{os.sep}{file}.csv'
file_name_xlsx = f'.{os.sep}results{os.sep}study{os.sep}{file}.xlsx'


# **** Column Generation ****
    # Prerequisites
modelImprovable = True

# Get Starting Solutions
problem_start = Problem(data, demand_dict, 0, Min_WD_i, Max_WD_i, 0)
problem_start.buildLinModel()
problem_start.model.Params.MIPFocus = 1
problem_start.model.Params.Heuristics = 1
problem_start.model.Params.RINS = 10
problem_start.model.Params.TimeLimit = time_cg_init
problem_start.model.update()
problem_start.model.optimize()

# Schedules
# Create
start_values_perf = {(t, s): problem_start.perf[1, t, s].x for t in T for s in K}
start_values_p = {(t): problem_start.p[1, t].x for t in T}
start_values_x = {(t, s): problem_start.x[1, t, s].x for t in T for s in K}
start_values_c = {(t): problem_start.sc[1, t].x for t in T}

# Initialize iterations
itr = 0
last_itr = 0

# Create empty results lists
histories = ["objValHistSP", "timeHist", "objValHistRMP", "avg_rc_hist", "lagrange_hist", "sum_rc_hist", "avg_sp_time", "rmp_time_hist", "sp_time_hist"]
histories_dict = {}
for history in histories:
    histories_dict[history] = []
objValHistSP, timeHist, objValHistRMP, avg_rc_hist, lagrange_hist, sum_rc_hist, avg_sp_time, rmp_time_hist, sp_time_hist = histories_dict.values()

X_schedules = {}
for index in I:
    X_schedules[f"Physician_{index}"] = []

Perf_schedules = create_schedule_dict(start_values_perf, 1, T, K)
Cons_schedules = create_schedule_dict(start_values_c, 1, T)
P_schedules = create_schedule_dict(start_values_p, 1, T)
X1_schedules = create_schedule_dict(start_values_x, 1, T, K)

master = MasterProblem(data, demand_dict, max_itr, itr, last_itr, output_len, start_values_perf)
master.buildModel()

# Initialize and solve relaxed model
master.setStartSolution()
master.updateModel()
master.solveRelaxModel()

# Retrieve dual values
duals_i0 = master.getDuals_i()
duals_ts0 = master.getDuals_ts()
print(f"{duals_i0, duals_ts0}")

# Start time count
t0 = time.time()
previous_reduced_cost = float('inf')

while modelImprovable and itr < max_itr:
    print("*{:^{output_len}}*".format(f"Begin Column Generation Iteration {itr}", output_len=output_len))

    # Start
    itr += 1

    # Solve RMP
    rmp_start_time = time.time()
    master.current_iteration = itr + 1
    master.solveRelaxModel()
    rmp_end_time = time.time()
    rmp_time_hist.append(rmp_end_time - rmp_start_time)

    objValHistRMP.append(master.model.objval)
    current_obj = master.model.objval

    # Get and Print Duals
    duals_i = master.getDuals_i()
    duals_ts = master.getDuals_ts()

    # Solve SPs
    modelImprovable = False

    # Build SP
    subproblem = Subproblem(duals_i, duals_ts, data, 1, itr, 0, Min_WD_i, Max_WD_i, 0)
    subproblem.buildModel()

    # Save time to solve SP
    sub_start_time = time.time()
    if previous_reduced_cost < -0.001:
        print("*{:^{output_len}}*".format(f"Use MIP-Gap > 0 in Iteration {itr}", output_len=output_len))
        subproblem.solveModelNOpt(time_cg)
    else:
        print("*{:^{output_len}}*".format(f"Use MIP-Gap = 0 in Iteration {itr}", output_len=output_len))
        subproblem.solveModelOpt(time_cg)
    sub_end_time = time.time()
    sp_time_hist.append(sub_end_time - sub_start_time)

    sub_totaltime = sub_end_time - sub_start_time
    timeHist.append(sub_totaltime)
    index = 1

    keys = ["X", "Perf", "P", "C", "X1"]
    methods = ["getOptX", "getOptPerf", "getOptP", "getOptC", "getOptX"]
    schedules = [X_schedules, Perf_schedules, P_schedules, Cons_schedules, X1_schedules]

    for key, method, schedule in zip(keys, methods, schedules):
        value = getattr(subproblem, method)()
        schedule[f"Physician_{index}"].append(value)

    # Check if SP is solvable
    status = subproblem.getStatus()
    if status != 2:
        raise Exception("*{:^{output_len}}*".format("Pricing-Problem can not reach optimality!", output_len=output_len))

    # Save ObjVal History
    reducedCost = subproblem.model.objval
    objValHistSP.append(reducedCost)

    # Update previous_reduced_cost for the next iteration
    previous_reduced_cost = reducedCost
    print("*{:^{output_len}}*".format(f"Reduced Costs in Iteration {itr}: {reducedCost}", output_len=output_len))

    # Increase latest used iteration
    last_itr = itr + 1

    # Generate and add columns with reduced cost
    if reducedCost < -threshold:
        Schedules = subproblem.getNewSchedule()
        master.addColumn(itr, Schedules)
        master.addLambda(itr)
        master.updateModel()
        modelImprovable = True

    # Update Model
    master.updateModel()

    # Calculate Metrics
    avg_rc = sum(objValHistSP) / len(objValHistSP)
    lagrange = avg_rc + current_obj
    sum_rc = sum(objValHistSP)
    avg_rc_hist.append(avg_rc)
    sum_rc_hist.append(sum_rc)
    lagrange_hist.append(lagrange)
    objValHistSP.clear()
    avg_time = sum(timeHist) / len(timeHist)
    avg_sp_time.append(avg_time)
    timeHist.clear()

    if not modelImprovable:
        print("*" * (output_len + 2))
        break

if modelImprovable and itr == max_itr:
    max_itr *= 2

# Solve Master Problem with integrality restored
master.model.setParam('PoolSearchMode', 2)
master.model.setParam('PoolSolutions', 100)
master.model.setParam('PoolGap', 0.05)
master.finalSolve(time_cg)

status = master.model.Status
if status in (gu.GRB.INF_OR_UNBD, gu.GRB.INFEASIBLE, gu.GRB.UNBOUNDED):
    print("The model cannot be solved because it is infeasible or unbounded")
    gu.sys.exit(1)

if status != gu.GRB.OPTIMAL:
    print(f"Optimization was stopped with status {status}")
    gu.sys.exit(1)

nSolutions = master.model.SolCount
print(f"Number of solutions found: {nSolutions}")

# Print objective values of solutions
for e in range(nSolutions):
    master.model.setParam(gu.GRB.Param.SolutionNumber, e)
    print(f"{master.model.PoolObjVal:g} ", end="")
    if e % 15 == 14:
        print("")
print("")

objValHistRMP.append(master.model.objval)

lagranigan_bound = round((objValHistRMP[-2] + sum_rc_hist[-1]), 3)



for epsilon in eps_ls:
    for chi in chi_ls:

        eps = epsilon
    # Calc Stats
        undercoverage_pool = []
        understaffing_pool = []
        perf_pool = []
        cons_pool = []
        undercoverage_pool_norm = []
        understaffing_pool_norm = []
        perf_pool_norm = []
        cons_pool_norm = []

        sol = master.printLambdas()

        ls_sc1 = plotPerformanceList(Cons_schedules, sol)
        ls_p1 = plotPerformanceList(Perf_schedules, sol)
        ls_x1 = plotPerformanceList(X_schedules, sol)
        ls_r1 = process_recovery(ls_sc1, chi, len(T))

        undercoverage_ab, understaffing_ab, perfloss_ab, consistency_ab, consistency_norm_ab, undercoverage_norm_ab, understaffing_norm_ab, perfloss_norm_ab = master.calc_naive(
            ls_p1, ls_sc1, ls_r1, eps)

        undercoverage_pool.append(undercoverage_ab)
        understaffing_pool.append(understaffing_ab)
        perf_pool.append(perfloss_ab)
        cons_pool.append(consistency_ab)
        undercoverage_pool_norm.append(undercoverage_norm_ab)
        understaffing_pool_norm.append(understaffing_norm_ab)
        perf_pool_norm.append(perfloss_norm_ab)
        cons_pool_norm.append(consistency_norm_ab)

        print(f"Solcount: {master.model.SolCount}")
        for k in range(master.model.SolCount):
            master.model.setParam(gu.GRB.Param.SolutionNumber, k)
            vals = master.model.getAttr("Xn", master.lmbda)

            solution = {key: round(value) for key, value in vals.items()}
            sum_lambda = sum(solution.values())
            if abs(sum_lambda - len(I)) > 1e-6:
                print(f"Skipping infeasible solution {k}: sum of lambda = {sum_lambda}")
                continue

            print(f"Processing feasible solution {k}")

            ls_sc = plotPerformanceList(Cons_schedules, solution)
            print(f"LsSc {ls_sc}")
            ls_p = plotPerformanceList(Perf_schedules, solution)
            ls_r = process_recovery(ls_sc, chi, len(T))
            ls_x = plotPerformanceList(X_schedules, solution)

            undercoverage_a, understaffing_a, perfloss_a, consistency_a, consistency_norm_a, undercoverage_norm_a, understaffing_norm_a, perfloss_norm_a = master.calc_naive(
                ls_p, ls_sc, ls_r, eps)

            undercoverage_pool.append(undercoverage_a)
            understaffing_pool.append(understaffing_a)
            perf_pool.append(perfloss_a)
            cons_pool.append(consistency_a)
            undercoverage_pool_norm.append(undercoverage_norm_a)
            understaffing_pool_norm.append(understaffing_norm_a)
            perf_pool_norm.append(perfloss_norm_a)
            cons_pool_norm.append(consistency_norm_a)

        # Nach der Schleife, geben Sie die Anzahl der zulässigen Lösungen aus
        print(f"Total feasible solutions processed: {len(undercoverage_pool)}")
        print(f"Under-List: {undercoverage_pool}")
        print(f"Perf-List: {perf_pool}")
        print(f"Cons-List: {cons_pool}")



        undercoverage = sum(undercoverage_pool) / len(undercoverage_pool)
        understaffing = sum(understaffing_pool) / len(understaffing_pool)
        perfloss = sum(perf_pool) / len(perf_pool)
        consistency = sum(cons_pool) / len(cons_pool)
        undercoverage_norm = sum(undercoverage_pool_norm) / len(undercoverage_pool_norm)
        understaffing_norm = sum(understaffing_pool_norm) / len(understaffing_pool_norm)
        perfloss_norm = sum(perf_pool_norm) / len(perf_pool_norm)
        consistency_norm = sum(cons_pool_norm) / len(cons_pool_norm)

        print(undercoverage_norm)

        # Coefficients
        sums, mean_value, min_value, max_value, indices_list = master.average_nr_of(ls_sc, len(master.nurses))
        variation_coefficients = [master.calculate_variation_coefficient(indices) for indices in indices_list]
        mean_variation_coefficient = (round(np.mean(variation_coefficients) * 100, 4))
        min_variation_coefficient = (round(np.min(variation_coefficients) * 100, 4))
        max_variation_coefficient = (round(np.max(variation_coefficients) * 100, 4))
        std_variation_coefficient = (round(np.std(variation_coefficients) * 100, 4))
        results_sc = [mean_variation_coefficient, min_variation_coefficient, max_variation_coefficient,
                      std_variation_coefficient]

        sums_r, mean_value_r, min_value_r, max_value_r, indices_list_r = master.average_nr_of(ls_r, len(master.nurses))
        variation_coefficients_r = [master.calculate_variation_coefficient(indices) for indices in indices_list_r]
        mean_variation_coefficient_r = (round(np.mean(variation_coefficients_r) * 100, 4))
        min_variation_coefficient_r = (round(np.min(variation_coefficients_r) * 100, 4))
        max_variation_coefficient_r = (round(np.max(variation_coefficients_r) * 100, 4))
        std_variation_coefficient_r = (round(np.std(variation_coefficients_r) * 100, 4))
        results_r = [mean_variation_coefficient_r, min_variation_coefficient_r, max_variation_coefficient_r,
                     std_variation_coefficient_r]

        # Gini
        autocorrel = master.autoccorrel(ls_sc, len(master.nurses), 2)

        result = pd.DataFrame([{
            'I': len(I),
            'pattern': "Medium",
            'epsilon': epsilon,
            'chi': chi,
            'undercover_n': undercoverage,
            'undercover_norm_n': undercoverage_norm,
            'cons_n': consistency,
            'cons_norm_n': consistency_norm,
            'perf_n': perfloss,
            'perf_norm_n': perfloss_norm,
            'max_auto_n': round(max(autocorrel), 5),
            'min_auto_n': round(min(autocorrel), 5),
            'mean_auto_n': round(np.mean(autocorrel), 5),
            'lagrange_n': lagranigan_bound
        }])

        results = pd.concat([results, result], ignore_index=True)


results.to_csv(file_name_csv, index=False)
results.to_excel(file_name_xlsx, index=False)

