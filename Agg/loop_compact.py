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
I_values = [25, 50, 100, 150]
prob_values = [1.0, 1.1, 1.2]
patterns = [2]
T = list(range(1, 29))
K = [1, 2, 3]

prob_mapping = {1.0: 'Low', 1.1: 'Medium', 1.2: 'High'}
pattern_mapping = {2: 'Noon'}

# Ergebnisse DataFrame initialisieren
results = pd.DataFrame(columns=['I', 'prob', 'lb', 'ub', 'gap', 'time', 'lb_c', 'ub_c', 'gap_c', 'time_cg', 'iter', 'time_rmp', 'time_sp'])

# Times and Parameter
time_Limit = 7200
time_cg = 7200
time_cg_init = 60
time_compact = 20
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
                demand_dict = demand_dict_fifty(len(T), prob, len(I), pattern, 0.25)

            seed = 123

            max_itr = 200
            output_len = 98
            mue = 1e-4
            threshold = 5e-5
            eps = 0.1

            # Demand Dict
            demand_dict = demand_dict_fifty(len(T), 1, len(I), 2, 0.1)

            data = pd.DataFrame({
                'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
                'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
                'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
            })

            # **** Compact Solver ****
            problem_t0 = time.time()
            problem = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i)
            problem.buildLinModel()
            problem.model.Params.TimeLimit = time_compact
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

            import time

            # Initialize iterations
            itr = 0
            last_itr = 0

            # Create empty results lists
            histories = ["objValHistSP", "timeHist", "objValHistRMP", "avg_rc_hist", "lagrange_hist", "sum_rc_hist",
                         "avg_sp_time", "gap_rc_hist", "rmp_time_hist", "sp_time_hist"]
            histories_dict = {}
            for history in histories:
                histories_dict[history] = []
            objValHistSP, timeHist, objValHistRMP, avg_rc_hist, lagrange_hist, sum_rc_hist, avg_sp_time, gap_rc_hist, rmp_time_hist, sp_time_hist = histories_dict.values()

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

                # Save current optimality gap
                gap_rc = round(
                    ((round(master.model.objval, 3) - round(obj_val_problem, 3)) / round(master.model.objval, 3)), 3)
                gap_rc_hist.append(gap_rc)

                # Solve SPs
                modelImprovable = False

                # Build SP
                subproblem = Subproblem(duals_i, duals_ts, data, 1, itr, eps, Min_WD_i, Max_WD_i)
                subproblem.buildModel()

                # Save time to solve SP
                sub_start_time = time.time()
                subproblem.solveModel(time_cg)
                sub_end_time = time.time()
                sp_time_hist.append(sub_end_time - sub_start_time)

                sub_totaltime = sub_end_time - sub_start_time
                timeHist.append(sub_totaltime)

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

            # Total Times
            time_rmp = sum(rmp_time_hist)
            time_sp = sum(sp_time_hist)

            # Capture total time and objval
            total_time_cg = time.time() - t0
            print(f"Total Time CG: {total_time_cg}")
            final_obj_cg = master.model.objval
            gap = round((((final_obj_cg - objValHistRMP[-2]) / objValHistRMP[-2]) * 100), 3)
            print(f"GAP CG: {gap}")

            result = pd.DataFrame([{
                'I': I_len,
                'prob': prob_mapping[prob],
                'lb': lower_bound,
                'ub': upper_bound,
                'gap': mip_gap,
                'time': runtime,
                'lb_cg': objValHistRMP[-2],
                'ub_cg': master.model.objval,
                'gap_cg': gap,
                'time_cg': total_time_cg,
                'iter': itr,
                'time_rmp': time_rmp,
                'time_sp': time_sp
            }])

            results = pd.concat([results, result], ignore_index=True)

results.to_csv('cg.csv', index=False)
results.to_excel('cg.xlsx', index=False)
