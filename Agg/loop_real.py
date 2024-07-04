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
eps_ls = [0.1001]
chi_ls = [3]
T = list(range(1, 29))
I = list(range(1, 101))
K = [1, 2, 3]


# Ergebnisse DataFrame initialisieren
results = pd.DataFrame(columns=['epsilon', 'chi', 'undercover', 'undercover_norm', 'cons', 'cons_norm', 'perf', 'perf_norm', 'undercover_behav', 'undercover_norm_behav', 'cons_behav', 'cons_norm_behav', 'perf_behav', 'perf_norm_behav'])
results_comp = pd.DataFrame(columns=['epsilon', 'chi', 'obj', 'pos', 'undercover', 'undercover_norm', 'cons', 'cons_norm', 'perf', 'perf_norm'])

# Times and Parameter
time_Limit = 7200
time_cg = 7200
time_cg_init = 20

## Dataframe
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
file = f'study_results_{current_time}'
file_comp = f'study_results_comp_{current_time}'
file_name_csv = f'.{os.sep}results{os.sep}study{os.sep}{file}.csv'
file_name_csv2 = f'.{os.sep}results{os.sep}study{os.sep}{file_comp}.csv'
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
        print(len(demand_dict))
        max_itr = 200
        output_len = 98
        mue = 1e-4
        threshold = 4e-5

        data = pd.DataFrame({
            'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
            'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
            'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
        })

        # **** Column Generation - Behavior ****
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
        obj1 = round(master.model.objval, 3)
        ls_sc = plotPerformanceList(Cons_schedules, master.printLambdas())
        ls_r = plotPerformanceList(Recovery_schedules, master.printLambdas())
        ls_e = plotPerformanceList(EUp_schedules, master.printLambdas())
        ls_b = plotPerformanceList(ELow_schedules, master.printLambdas())
        ls_x = plotPerformanceList(X_schedules, master.printLambdas())
        understaffing1, u_results, sum_all_doctors, consistency2, consistency2_norm, understaffing1_norm, u_results_norm, sum_all_doctors_norm = master.calc_behavior(plotPerformanceList(Perf_schedules, master.printLambdas()), ls_sc)

        result_comp = pd.DataFrame([{
            'epsilon': epsilon,
            'chi': chi,
            'pos': 0,
            'obj': obj1,
            'undercover': round(understaffing1, 3),
            'undercover_norm': round(understaffing1_norm, 3),
            'cons': consistency2,
            'cons_norm': consistency2_norm,
            'perf': round(sum_all_doctors, 3),
            'perf_norm': round(sum_all_doctors_norm, 3)
        }])
        results_comp = pd.concat([results_comp, result_comp], ignore_index=True)



        # **** Column Generation - Behavior ****
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
            subproblem = Subproblem(duals_i, duals_ts, data, 1, itr, 0, Min_WD_i, Max_WD_i, chi)
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
        obj2 = round(master.model.objval, 3)
        ls_sc = plotPerformanceList(Cons_schedules, master.printLambdas())
        ls_r = plotPerformanceList(Recovery_schedules, master.printLambdas())
        ls_e = plotPerformanceList(EUp_schedules, master.printLambdas())
        ls_b = plotPerformanceList(ELow_schedules, master.printLambdas())
        ls_x = plotPerformanceList(X_schedules, master.printLambdas())
        understaffing1_behav, u_results_behav, sum_all_doctors_behav, consistency2_behav, consistency2_norm_behav, understaffing1_norm_behav, u_results_norm_behav, sum_all_doctors_norm_behav = master.calc_naive(plotPerformanceList(Perf_schedules, master.printLambdas()), ls_sc, ls_r, ls_e, ls_b, ls_x,eps)

        result = pd.DataFrame([{
            'epsilon': epsilon,
            'chi': chi,
            'undercover': round(understaffing1, 3),
            'undercover_norm': round(understaffing1_norm, 3),
            'cons': consistency2,
            'cons_norm': consistency2_norm,
            'perf': round(sum_all_doctors, 3),
            'perf_norm': round(sum_all_doctors_norm, 3),
            'undercover_behav': round(understaffing1_behav, 3),
            'undercover_norm_behav': round(understaffing1_norm_behav, 3),
            'cons_behav': consistency2_behav,
            'cons_norm_behav': consistency2_norm_behav,
            'perf_behav': round(sum_all_doctors_behav, 3),
            'perf_norm_behav': round(sum_all_doctors_norm_behav, 3)
        }])

        results = pd.concat([results, result], ignore_index=True)

        result_comp = pd.DataFrame([{
            'epsilon': epsilon,
            'chi': chi,
            'pos': 0,
            'obj': obj2,
            'undercover': round(understaffing1_behav, 3),
            'undercover_norm': round(understaffing1_norm_behav, 3),
            'cons': consistency2_behav,
            'cons_norm': consistency2_norm_behav,
            'perf': round(sum_all_doctors_behav, 3),
            'perf_norm': round(sum_all_doctors_norm_behav, 3)
        }])
        results_comp = pd.concat([results_comp, result_comp], ignore_index=True)

results.to_csv(file_name_csv, index=False)
results_comp.to_csv(file_name_csv2, index=False)
