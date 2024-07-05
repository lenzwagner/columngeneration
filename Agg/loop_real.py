import math
from masterproblem import *
from setup import Min_WD_i, Max_WD_i
from gcutil import *
from subproblem import *
from compactsolver import Problem
from demand import *
from datetime import datetime
import os
import time


# **** Prerequisites ****
# Create Dataframes
eps_ls = [0.02501]
chi_ls = [3, 5, 7]
T = list(range(1, 29))
I = list(range(1, 101))
K = [1, 2, 3]


# Ergebnisse DataFrame initialisieren
results = pd.DataFrame(columns=['epsilon', 'chi', 'obj', 'undercover', 'undercover_norm', 'cons', 'cons_norm', 'perf', 'perf_norm' ,'undercover_n', 'undercover_norm_n', 'cons_n', 'cons_norm_n', 'perf_n', 'perf_norm_n'])

# Times and Parameter
time_Limit = 7200
time_cg = 7200
time_cg_init = 2

## Dataframe
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
file = f'study_results_{current_time}'
file_name_csv = f'.{os.sep}results{os.sep}study{os.sep}{file}.csv'
file_name_xlsx = f'.{os.sep}results{os.sep}study{os.sep}{file}.xlsx'

# Parameter
for epsilon in eps_ls:
    for chi in chi_ls:

        eps = epsilon
        print(f"")
        print(f"Iteration: {epsilon}-{chi}")
        print(f"")

        seed1 = 123 - math.floor(len(I)*len(T)*1.0)
        random.seed(seed1)
        demand_dict = demand_dict_fifty(len(T), 1.0, len(I), 2, 0.25)

        max_itr = 200
        output_len = 98
        mue = 1e-4
        threshold = 4e-5

        data = pd.DataFrame({
            'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
            'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
            'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
        })

        # **** Column Generation ****
        # Prerequisites
        modelImprovable = True
        reached_max_itr = False

        # Get Starting Solutions
        problem_start = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i)
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
        start_values_r = {(t): problem_start.r[1, t].x for t in T}
        start_values_eup = {(t): problem_start.e[1, t].x for t in T}
        start_values_elow = {(t): problem_start.b[1, t].x for t in T}

        # Initialize iterations
        itr = 0
        last_itr = 0

        # Create empty results lists
        histories = ["objValHistSP", "timeHist", "objValHistRMP", "avg_rc_hist", "lagrange_hist", "sum_rc_hist",
                     "avg_sp_time", "rmp_time_hist", "sp_time_hist"]
        histories_dict = {}
        for history in histories:
            histories_dict[history] = []
        objValHistSP, timeHist, objValHistRMP, avg_rc_hist, lagrange_hist, sum_rc_hist, avg_sp_time, rmp_time_hist, sp_time_hist = histories_dict.values()

        X_schedules = {}
        for index in I:
            X_schedules[f"Physician_{index}"] = []

        Perf_schedules = create_schedule_dict(start_values_perf, 1, T, K)
        Cons_schedules = create_schedule_dict(start_values_c, 1, T)
        Recovery_schedules = create_schedule_dict(start_values_r, 1, T)
        EUp_schedules = create_schedule_dict(start_values_eup, 1, T)
        ELow_schedules = create_schedule_dict(start_values_elow, 1, T)
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
            current_bound = master.model.objval

            # Get and Print Duals
            duals_i = master.getDuals_i()
            duals_ts = master.getDuals_ts()

            # Solve SPs
            modelImprovable = False

            # Build SP
            subproblem = Subproblem(duals_i, duals_ts, data, 1, itr, eps, Min_WD_i, Max_WD_i, chi)
            subproblem.buildModel()

            # Save time to solve SP
            sub_start_time = time.time()
            subproblem.solveModel(time_cg)
            sub_end_time = time.time()
            sp_time_hist.append(sub_end_time - sub_start_time)

            sub_totaltime = sub_end_time - sub_start_time
            timeHist.append(sub_totaltime)
            index = 1

            keys = ["X", "Perf", "P", "C", "R", "EUp", "Elow", "X1"]
            methods = ["getOptX", "getOptPerf", "getOptP", "getOptC", "getOptR", "getOptEUp", "getOptElow", "getOptX"]
            schedules = [X_schedules, Perf_schedules, P_schedules, Cons_schedules, Recovery_schedules, EUp_schedules,
                         ELow_schedules, X1_schedules]

            for key, method, schedule in zip(keys, methods, schedules):
                value = getattr(subproblem, method)()
                schedule[f"Physician_{index}"].append(value)

            # Check if SP is solvable
            status = subproblem.getStatus()
            if status != 2:
                raise Exception(
                    "*{:^{output_len}}*".format("Pricing-Problem can not reach optimality!", output_len=output_len))

            # Save ObjVal History
            reducedCost = subproblem.model.objval
            objValHistSP.append(reducedCost)
            print("*{:^{output_len}}*".format(f"Reduced Costs in Iteration {itr}: {reducedCost}",
                                              output_len=output_len))

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
        master.finalSolve(time_cg)
        objValHistRMP.append(master.model.objval)

        # Calc Stats
        ls_sc = plotPerformanceList(Cons_schedules, master.printLambdas())
        ls_r = plotPerformanceList(Recovery_schedules, master.printLambdas())
        ls_e = plotPerformanceList(EUp_schedules, master.printLambdas())
        ls_b = plotPerformanceList(ELow_schedules, master.printLambdas())
        ls_x = plotPerformanceList(X_schedules, master.printLambdas())
        understaffing1, u_results, sum_all_doctors, consistency2, consistency2_norm, understaffing1_norm, u_results_norm, sum_all_doctors_norm = master.calc_behavior(plotPerformanceList(Perf_schedules, master.printLambdas()), ls_sc)



        # **** Naive ****
        # Prerequisites
        modelImprovable = True
        reached_max_itr = False

        # Get Starting Solutions
        problem_start_n = Problem(data, demand_dict, 0, Min_WD_i, Max_WD_i)
        problem_start_n.buildLinModel()
        problem_start_n.model.Params.MIPFocus = 1
        problem_start_n.model.Params.Heuristics = 1
        problem_start_n.model.Params.RINS = 10
        problem_start_n.model.Params.TimeLimit = time_cg_init
        problem_start_n.model.update()
        problem_start_n.model.optimize()

        # Schedules
        # Create
        start_values_perf_n = {(t, s): problem_start_n.perf[1, t, s].x for t in T for s in K}
        start_values_p_n = {(t): problem_start_n.p[1, t].x for t in T}
        start_values_x_n = {(t, s): problem_start_n.x[1, t, s].x for t in T for s in K}
        start_values_c_n = {(t): problem_start_n.sc[1, t].x for t in T}
        start_values_r_n = {(t): problem_start_n.r[1, t].x for t in T}
        start_values_eup_n = {(t): problem_start_n.e[1, t].x for t in T}
        start_values_elow_n = {(t): problem_start_n.b[1, t].x for t in T}

        # Initialize iterations
        itr = 0
        last_itr = 0

        # Create empty results lists
        histories_n = ["objValHistSP_n", "timeHist_n", "objValHistRMP_n", "avg_rc_hist_n", "lagrange_hist_n", "sum_rc_hist_n",
                     "avg_sp_time_n", "rmp_time_hist_n", "sp_time_hist_n"]
        histories_n_dict = {}
        for history in histories_n:
            histories_n_dict[history] = []
        objValHistSP_n, timeHist_n, objValHistRMP_n, avg_rc_hist_n, lagrange_hist_n, sum_rc_hist_n, avg_sp_time_n, rmp_time_hist_n, sp_time_hist_n = histories_n_dict.values()

        X_schedules_n = {}
        for index in I:
            X_schedules_n[f"Physician_{index}"] = []

        Perf_schedules_n = create_schedule_dict(start_values_perf_n, 1, T, K)
        Cons_schedules_n = create_schedule_dict(start_values_c_n, 1, T)
        Recovery_schedules_n = create_schedule_dict(start_values_r_n, 1, T)
        EUp_schedules_n = create_schedule_dict(start_values_eup_n, 1, T)
        ELow_schedules_n = create_schedule_dict(start_values_elow_n, 1, T)
        P_schedules_n = create_schedule_dict(start_values_p_n, 1, T)
        X1_schedules_n = create_schedule_dict(start_values_x_n, 1, T, K)

        master_n = MasterProblem(data, demand_dict, max_itr, itr, last_itr, output_len, start_values_perf)
        master_n.buildModel()

        # Initialize and solve relaxed model
        master_n.setStartSolution()
        master_n.updateModel()
        master_n.solveRelaxModel()

        # Retrieve dual values
        duals_i0_n = master_n.getDuals_i()
        duals_ts0_n = master_n.getDuals_ts()
        print(f"{duals_i0_n, duals_ts0_n}")

        # Start time count
        t0 = time.time()

        while modelImprovable and itr < max_itr:
            print("*{:^{output_len}}*".format(f"Begin Column Generation Iteration {itr}", output_len=output_len))

            # Start
            itr += 1

            # Solve RMP
            rmp_start_time = time.time()
            master_n.current_iteration = itr + 1
            master_n.solveRelaxModel()
            rmp_end_time = time.time()
            rmp_time_hist_n.append(rmp_end_time - rmp_start_time)

            objValHistRMP_n.append(master.model.objval)
            current_obj_n = master.model.objval
            current_bound_n = master.model.objval

            # Get and Print Duals
            duals_i_n = master_n.getDuals_i()
            duals_ts_n = master_n.getDuals_ts()

            # Solve SPs
            modelImprovable = False

            # Build SP
            subproblem_n = Subproblem(duals_i_n, duals_ts_n, data, 1, itr, 0, Min_WD_i, Max_WD_i, chi)
            subproblem_n.buildModel()

            # Save time to solve SP
            sub_start_time = time.time()
            subproblem_n.solveModel(time_cg)
            sub_end_time = time.time()
            sp_time_hist_n.append(sub_end_time - sub_start_time)

            sub_totaltime = sub_end_time - sub_start_time
            timeHist_n.append(sub_totaltime)
            index = 1

            keys = ["X", "Perf", "P", "C", "R", "EUp", "Elow", "X1"]
            methods = ["getOptX", "getOptPerf", "getOptP", "getOptC", "getOptR", "getOptEUp", "getOptElow", "getOptX"]
            schedules = [X_schedules_n, Perf_schedules_n, P_schedules_n, Cons_schedules_n, Recovery_schedules_n, EUp_schedules_n,
                         ELow_schedules_n, X1_schedules_n]

            for key, method, schedule in zip(keys, methods, schedules):
                value = getattr(subproblem_n, method)()
                schedule[f"Physician_{index}"].append(value)

            # Check if SP is solvable
            status = subproblem_n.getStatus()
            if status != 2:
                raise Exception(
                    "*{:^{output_len}}*".format("Pricing-Problem can not reach optimality!", output_len=output_len))

            # Save ObjVal History
            reducedCost_n = subproblem_n.model.objval
            objValHistSP_n.append(reducedCost_n)
            print("*{:^{output_len}}*".format(f"Reduced Costs in Iteration {itr}: {reducedCost}",
                                              output_len=output_len))

            # Increase latest used iteration
            last_itr = itr + 1

            # Generate and add columns with reduced cost
            if reducedCost_n < -threshold:
                Schedules_n = subproblem_n.getNewSchedule()
                master_n.addColumn(itr, Schedules_n)
                master_n.addLambda(itr)
                master_n.updateModel()
                modelImprovable = True

            # Update Model
            master_n.updateModel()

            if not modelImprovable:
                print("*" * (output_len + 2))
                break

        if modelImprovable and itr == max_itr:
            max_itr *= 2

        # Solve Master Problem with integrality restored
        master_n.finalSolve(time_cg)

        # Calc Stats
        ls_sc_n = plotPerformanceList(Cons_schedules, master_n.printLambdas())
        ls_r_n = plotPerformanceList(Recovery_schedules, master_n.printLambdas())
        ls_e_n = plotPerformanceList(EUp_schedules, master_n.printLambdas())
        ls_b_n = plotPerformanceList(ELow_schedules, master_n.printLambdas())
        ls_x_n = plotPerformanceList(X_schedules, master_n.printLambdas())
        understaffing1_n, u_results_n, sum_all_doctors_n, consistency2_n, consistency2_norm_n, understaffing1_norm_n, u_results_norm_n, sum_all_doctors_norm_n = master_n.calc_naive(plotPerformanceList(Perf_schedules, master.printLambdas()), ls_sc, ls_r, ls_e, ls_b, ls_x,epsilon)

        result = pd.DataFrame([{
            'epsilon': epsilon,
            'chi': chi,
            'obj': round(master.model.objval, 3),
            'undercover': round(understaffing1, 3),
            'undercover_norm': round(understaffing1_norm, 3),
            'cons': consistency2,
            'cons_norm': consistency2_norm,
            'perf': sum_all_doctors,
            'perf_norm': sum_all_doctors_norm,
            'undercover_n':round(understaffing1_n, 3),
            'undercover_norm_n':round(understaffing1_norm_n, 3),
            'cons_n': consistency2_n,
            'cons_norm_n': consistency2_norm_n,
            'perf_n': sum_all_doctors_n,
            'perf_norm_n': sum_all_doctors_norm_n
        }])

        master.process_LSR(ls_sc, len(master.nurses))


        results = pd.concat([results, result], ignore_index=True)

results.to_csv(file_name_csv, index=False)
results.to_excel(file_name_xlsx, index=False)
