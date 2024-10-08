import math
from masterproblem import *
import time
from setup import *
from gcutil import *
from subproblem import *
from compactsolver import Problem
from demand import *
from datetime import datetime
import os

# **** Prerequisites ****
# Create Dataframes
I_values = [150]
prob_values = [0.9, 1.0, 1.1]
pattern = 2
T = list(range(1, 29))
K = [1, 2, 3]

prob_mapping = {0.9: 'Low', 1.0: 'Medium', 1.1: 'High'}
pattern_mapping = {2: 'Noon'}

# Ergebnisse DataFrame initialisieren
results = pd.DataFrame(columns=['I', 'prob', 'lb_cg', 'ub_cg', 'gap_cg', 'time_cg', 'iter', 'time_rmp', 'time_sp', 'time_ip'])

# Times and Parameter
time_Limit = 7200
time_cg = 7200
time_cg_init = 60
time_compact = 20
eps = 0.06
chi = 3

## Dataframe
current_time = datetime.now().strftime('%Y-%m-%d_%H')
file = f'results_comp_anal_{current_time}'
file_name_csv = f'.{os.sep}results{os.sep}{file}.csv'

# Parameter
for I_len in I_values:
    I = list(range(1, I_len + 1))
    for prob in prob_values:

        print(f"")
        print(f"Iteration: {len(I)}-{prob}-{pattern}")
        print(f"")

        seed1 = 123 - math.floor(len(I)*len(T))
        print(seed1)
        random.seed(seed1)

        max_itr = 200
        output_len = 98
        mue = 1e-4
        threshold = 6e-5

        # Demand Dict
        demand_dict = demand_dict_fifty(len(T), prob, len(I), 2, 0.25)
        print('Demand dict', demand_dict)

        data = pd.DataFrame({
            'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
            'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
            'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
        })

        # **** Column Generation ****
        # Prerequisites
        modelImprovable = True

        # Get Starting Solutions
        problem_start = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i, chi)
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
            subproblem = Subproblem(duals_i, duals_ts, data, 1, itr, eps, Min_WD_i, Max_WD_i, chi)
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

            # Update previous_reduced_cost for the next iteration
            previous_reduced_cost = reducedCost
            print(
                "*{:^{output_len}}*".format(f"Reduced Costs in Iteration {itr}: {reducedCost}", output_len=output_len))

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
        time_ip_start = time.time()
        time_ip_end = time.time() - time_ip_start
        objValHistRMP.append(master.model.objval)
        final_obj = master.model.objval
        final_lb = objValHistRMP[-2]

        # Total Times
        time_rmp = round(sum(rmp_time_hist), 3)
        time_sp = round(sum(sp_time_hist), 3)

        # Capture total time and objval
        total_time_cg = time.time() - t0
        print(f"Total Time CG: {total_time_cg}")
        gap = round((((master.model.objval - objValHistRMP[-2]) / objValHistRMP[-2]) * 100), 3)
        print(f"GAP CG: {gap}")

        print("")
        print("")
        print("")
        print(master.model.objval)
        print("")
        print("")
        print("")


        result = pd.DataFrame([{
            'I': I_len,
            'prob': prob_mapping[prob],
            'lb_cg': round(objValHistRMP[-2], 3),
            'ub_cg': round(master.model.objval, 3),
            'gap_cg': gap,
            'time_cg': round(total_time_cg, 3),
            'iter': itr,
            'time_rmp': round(time_rmp, 3),
            'time_sp': round(time_sp, 3),
            'time_ip': round(time_ip_end, 3)
        }])

        results = pd.concat([results, result], ignore_index=True)

#results.to_csv(file_name_csv, index=False)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print(results)