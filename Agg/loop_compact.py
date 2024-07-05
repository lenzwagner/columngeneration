import math

from masterproblem import *
import time
from setup import Min_WD_i, Max_WD_i
from gcutil import *
from subproblem import *
from compactsolver import Problem
from demand import *
from datetime import datetime
import os

# **** Prerequisites ****
# Create Dataframes
I_values = [100]
prob_values = [0.9,1.0,1.1]
pattern = 2
T = list(range(1, 29))
K = [1, 2, 3]

prob_mapping = {0.9: 'Low', 1.0: 'Medium', 1.1: 'High'}
pattern_mapping = {2: 'Noon'}

# Ergebnisse DataFrame initialisieren
results = pd.DataFrame(columns=['I', 'prob', 'lb', 'ub', 'gap', 'time', 'lb_cg', 'ub_cg', 'gap_cg', 'time_cg', 'iter', 'time_rmp', 'time_sp', 'time_ip'])

# Times and Parameter
time_Limit = 7200
time_cg = 7200
time_cg_init = 60
time_compact = 7200
eps = 0.1

## Dataframe
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
file = f'results_{current_time}'
file_name_csv = f'.{os.sep}results{os.sep}{file}.csv'
file_name_xlsx = f'.{os.sep}results{os.sep}{file}.xlsx'

# Parameter
for I_len in I_values:
    I = list(range(1, I_len + 1))
    for prob in prob_values:


        print(f"")
        print(f"Iteration: {len(I)}-{prob}-{pattern}")
        print(f"")

        seed1 = 123 - math.floor(len(I)*len(T)*prob)
        random.seed(seed1)
        demand_dict = demand_dict_fifty(len(T), prob, len(I), pattern, 0.25)

        max_itr = 200
        output_len = 98
        mue = 1e-4
        threshold = 4e-5
        eps = 0.1

        # Demand Dict
        demand_dict = demand_dict_fifty(len(T), 1, len(I), 2, 0.1)
        print('Demand dict', demand_dict)

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

        obj_val_problem = round(problem.model.objval, 2)
        time_problem = time.time() - problem_t0
        vals_prob = problem.get_final_values()


        runtime = round(problem_t1 - problem_t0, 2)
        mip_gap = round(problem.model.MIPGap, 3)
        lower_bound = round(problem.model.ObjBound, 2)
        print(f"lower_bound {lower_bound}")
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

            # Save current optimality gap
            gap_rc = round(
                ((round(master.model.objval, 3) - round(obj_val_problem, 3)) / round(master.model.objval, 3)), 3)
            gap_rc_hist.append(gap_rc)

            # Solve SPs
            modelImprovable = False

            # Build SP
            subproblem = Subproblem(duals_i, duals_ts, data, 1, itr, eps, Min_WD_i, Max_WD_i, 5)
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
        time_ip_start = time.time()
        master.finalSolve(time_cg)
        time_ip_end = time.time() - time_ip_start
        objValHistRMP.append(master.model.objval)

        # Total Times
        time_rmp = round(sum(rmp_time_hist), 3)
        time_sp = round(sum(sp_time_hist), 3)

        # Capture total time and objval
        total_time_cg = time.time() - t0
        print(f"Total Time CG: {total_time_cg}")
        gap = round((((master.model.objval - objValHistRMP[-2]) / objValHistRMP[-2]) * 100), 3)
        print(f"GAP CG: {gap}")

        result = pd.DataFrame([{
            'I': I_len,
            'prob': prob_mapping[prob],
            'lb': round(lower_bound, 3),
            'ub': round(upper_bound, 3),
            'gap': mip_gap,
            'time': round(runtime, 3),
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

results.to_csv(file_name_csv, index=False)
results.to_excel(file_name_xlsx, index=False)
