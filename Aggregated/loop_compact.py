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
I_values = [50]
prob_values = [1.0]
patterns = [2]
T = list(range(1, 29))
K = [1, 2, 3]

prob_mapping = {1.0: 'Low', 1.1: 'Medium', 1.2: 'High'}
pattern_mapping = {2: 'Noon'}

# Ergebnisse DataFrame initialisieren
results = pd.DataFrame(columns=['I', 'prob', 'lb', 'ub', 'gap', 'time', 'lb_c', 'ub_c', 'gap_c', 'time_cg'])

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

            max_itr = 400
            output_len = 98
            mue = 1e-4
            threshold = 5e-5
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


            runtime = round(problem_t1 - problem_t0, 4)
            mip_gap = round(problem.model.MIPGap, 4)
            lower_bound = round(problem.model.ObjBound, 4)
            upper_bound = round(problem.model.ObjVal, 4)
            objective_value = round(problem.model.ObjVal, 4)

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
            problem_start.model.Params.TimeLimit = 60
            problem_start.model.update()
            problem_start.model.optimize()

            # Schedules
            # Create
            start_values_perf = {(i, t, s): problem_start.perf[i, t, s].x for i in I for t in T for s in K}
            start_values_p = {(i, t): problem_start.p[i, t].x for i in I for t in T}
            start_values_x = {(i, t, s): problem_start.x[i, t, s].x for i in I for t in T for s in K}
            start_values_c = {(i, t): problem_start.sc[i, t].x for i in I for t in T}
            start_values_r = {(i, t): problem_start.r[i, t].x for i in I for t in T}
            start_values_eup = {(i, t): problem_start.e[i, t].x for i in I for t in T}
            start_values_elow = {(i, t): problem_start.b[i, t].x for i in I for t in T}

            while True:
                # Initialize iterations
                itr = 0
                last_itr = 0

                # Create empty results lists
                histories = ["objValHistSP", "timeHist", "objValHistRMP", "avg_rc_hist", "lagrange_hist", "sum_rc_hist",
                             "avg_sp_time", "gap_rc_hist"]
                histories_dict = {}
                for history in histories:
                    histories_dict[history] = []
                objValHistSP, timeHist, objValHistRMP, avg_rc_hist, lagrange_hist, sum_rc_hist, avg_sp_time, gap_rc_hist = histories_dict.values()

                X_schedules = {}
                for index in I:
                    X_schedules[f"Physician_{index}"] = []

                Perf_schedules = create_schedule_dict(start_values_perf, I, T, K)
                Cons_schedules = create_schedule_dict(start_values_c, I, T)
                Recovery_schedules = create_schedule_dict(start_values_r, I, T)
                EUp_schedules = create_schedule_dict(start_values_eup, I, T)
                ELow_schedules = create_schedule_dict(start_values_elow, I, T)
                P_schedules = create_schedule_dict(start_values_p, I, T)
                X1_schedules = create_schedule_dict(start_values_x, I, T, K)

                master = MasterProblem(data, demand_dict, max_itr, itr, last_itr, output_len, start_values_perf)
                master.buildModel()

                # Initialize and solve relaxed model
                master.setStartSolution()
                master.updateModel()
                master.solveRelaxModel()

                # Retrieve dual values
                duals_i0 = master.getDuals_i()
                duals_ts0 = master.getDuals_ts()

                # Start time count
                t0 = time.time()

                while (modelImprovable) and itr < max_itr:
                    print(
                        "*{:^{output_len}}*".format(f"Begin Column Generation Iteration {itr}", output_len=output_len))

                    # Start
                    itr += 1

                    # Solve RMP
                    master.current_iteration = itr + 1
                    master.solveRelaxModel()
                    objValHistRMP.append(master.model.objval)
                    current_obj = master.model.objval
                    current_bound = master.model.objval

                    # Get and Print Duals
                    duals_i = master.getDuals_i()
                    duals_ts = master.getDuals_ts()

                    # Save current optimality gap
                    gap_rc = round(
                        ((round(master.model.objval, 3) - round(obj_val_problem, 3)) / round(master.model.objval, 3)),
                        3)
                    gap_rc_hist.append(gap_rc)

                    # Solve SPs
                    modelImprovable = False

                    # Build SP
                    subproblem = Subproblem(duals_i, duals_ts, data, 1, itr, eps, Min_WD_i, Max_WD_i)
                    subproblem.buildModel()

                    # Save time to solve SP
                    sub_t0 = time.time()
                    subproblem.solveModel(time_Limit)
                    sub_totaltime = time.time() - sub_t0
                    timeHist.append(sub_totaltime)

                    # Get optimal values
                    keys = ["X", "Perf", "P", "C", "R", "EUp", "Elow", "X1"]
                    methods = ["getOptX", "getOptPerf", "getOptP", "getOptC", "getOptR", "getOptEUp", "getOptElow",
                               "getOptX"]
                    schedules = [X_schedules, Perf_schedules, P_schedules, Cons_schedules, Recovery_schedules,
                                 EUp_schedules,
                                 ELow_schedules, X1_schedules]

                    for key, method, schedule in zip(keys, methods, schedules):
                        value = getattr(subproblem, method)()
                        schedule[f"Physician_{index}"].append(value)

                    # Check if SP is solvable
                    status = subproblem.getStatus()
                    if status != 2:
                        raise Exception("*{:^{output_len}}*".format("Pricing-Problem can not reach optimality!",
                                                                    output_len=output_len))

                    # Save ObjVal History
                    reducedCost = subproblem.model.objval
                    objValHistSP.append(reducedCost)
                    print(
                        "*{:^{output_len}}*".format(f"Reduced Costs in Iteration {itr}: {reducedCost}", output_len=output_len))

                    # Increase latest used iteration
                    last_itr = itr + 1

                    # Generate and add columns with reduced cost
                    if reducedCost < -threshold:
                        Schedules = subproblem.getNewSchedule()
                        for index in [1]:
                            master.addColumn(index, itr, Schedules)
                            master.addLambda(index, itr)
                            master.updateModel()

                        for index in range(2, len(I) + 1):
                            adjusted_Schedules = {(index,) + key[1:]: value for key, value in Schedules.items()}
                            master.addColumn(index, itr, adjusted_Schedules)
                            master.addLambda(index, itr)
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
                else:
                    break

            # Solve Master Problem with integrality restored
            master.finalSolve(time_Limit)
            objValHistRMP.append(master.model.objval)

            # Capture total time and objval
            total_time_cg = time.time() - t0
            final_obj_cg = master.model.objval
            gap = round((((final_obj_cg - objValHistRMP[-2]) / objValHistRMP[-2]) * 100), 3)


            result = pd.DataFrame([{
                'I': I_len,
                'prob': prob_mapping[prob],
                'lb': lower_bound,
                'ub': upper_bound,
                'gap': mip_gap,
                'time': runtime,
                'lb_c': round(objValHistRMP[-2], 4),
                'ub_c': round(final_obj_cg, 4),
                'gap_c': round(mip_gap, 4),
                'time_cg': round(total_time_cg, 4),
            }])

            results = pd.concat([results, result], ignore_index=True)





results.to_csv('compact.csv', index=False)
results.to_excel('compact.xlsx', index=False)
