from masterproblem import *
import time
from setup import *
from gcutil import *
from subproblem import *
from compactsolver import Problem
from demand import *
import os

# **** Prerequisites ****
# Create Dataframes

I, T, K = list(range(1, 151)), list(range(1, 29)), list(range(1, 4))
data = pd.DataFrame({
    'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
    'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
    'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
})

# Remove unused files
for file in os.listdir():
    if file.endswith('.lp') or file.endswith('.sol') or file.endswith('.txt'):
        os.remove(file)

## Dataframe
results_cg = pd.DataFrame(columns=["seed","I", "D", "S", "objective_value_cp", "objective_value_cg", "time_cg", "time_cp", "gap", "mip_gap", "lowerbound", "optimal", "lagrange"])

# Parameter
seed1 = 123 - math.floor(len(I) * len(T))
print(seed1)
random.seed(seed1)

max_itr = 200
output_len = 98
mue = 1e-4
eps = 0.06
threshold = 6e-5

# Demand Dict
demand_dict = demand_dict_fifty(len(T), 1.1, len(I), 2, 0.25)
print('Demand dict', demand_dict)


# **** Compact Solver ****
problem_t0 = time.time()
problem = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i, 5)
problem.buildLinModel()
problem.updateModel()
problem.solveModel()

bound = problem.model.ObjBound
print(f"Bound {bound}")

obj_val_problem = round(problem.model.objval, 3)
time_problem = time.time() - problem_t0
vals_prob = problem.get_final_values()
print(obj_val_problem)


# **** Column Generation ****
# Prerequisites
modelImprovable = True
reached_max_itr = False

# Get Starting Solutions
problem_start = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i, 5)
problem_start.buildLinModel()
problem_start.model.Params.MIPFocus = 1
problem_start.model.Params.Heuristics = 1
problem_start.model.Params.RINS = 10
problem_start.model.Params.MIPGap = 0.8
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
    t0 = time.time()
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
        gap_rc = round(((round(master.model.objval, 3) - round(obj_val_problem, 3)) / round(master.model.objval, 3)), 3)
        gap_rc_hist.append(gap_rc)

        # Solve SPs
        modelImprovable = False
        print("*{:^{output_len}}*".format(f"Begin Column Generation Iteration {itr}", output_len=output_len))

        # Build SP
        subproblem = Subproblem(duals_i, duals_ts, data, 1, itr, eps, Min_WD_i, Max_WD_i)
        subproblem.buildModel()

        # Save time to solve SP
        sub_t0 = time.time()
        subproblem.solveModel()
        sub_totaltime = time.time() - sub_t0
        timeHist.append(sub_totaltime)

        # Get optimal values
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
            raise Exception("*{:^{output_len}}*".format("Pricing-Problem can not reach optimality!", output_len=output_len))

        # Save ObjVal History
        reducedCost = subproblem.model.objval
        objValHistSP.append(reducedCost)

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

        avg_time = sum(timeHist)/len(timeHist)
        avg_sp_time.append(avg_time)
        timeHist.clear()

        print("*{:^{output_len}}*".format(f"End Column Generation Iteration {itr}", output_len=output_len))

        if not modelImprovable:
            print("*{:^{output_len}}*".format("", output_len=output_len))
            print("*{:^{output_len}}*".format("No more improvable columns found.", output_len=output_len))
            print("*{:^{output_len}}*".format("", output_len=output_len))
            print("*" * (output_len + 2))

            break

    if modelImprovable and itr == max_itr:
        print("*{:^{output_len}}*".format("More iterations needed. Increase max_itr and restart the process.",
                                          output_len=output_len))
        max_itr *= 2
    else:
        break

# Solve Master Problem with integrality restored
master.finalSolve()
objValHistRMP.append(master.model.objval)

# Capture total time and objval
total_time_cg = time.time() - t0
final_obj_cg = master.model.objval

# Calculate Gap
# Relative to the lower bound (best possible achievable solution)
gap = ((objValHistRMP[-1]-objValHistRMP[-2])/objValHistRMP[-2])*100

# Lagragian Bound
# Only yields feasible results if the SPs are solved to optimality
lagranigan_bound = round((objValHistRMP[-2] + sum_rc_hist[-1]), 3)

### Store results
mip_gap = round((((final_obj_cg-obj_val_problem) / obj_val_problem) * 100),3)

printResults(itr, total_time_cg, time_problem, output_len, final_obj_cg, objValHistRMP[-2], lagranigan_bound,
             obj_val_problem, eps)

if mip_gap < 0.001:
    opt = 1
else:
    opt = 0

print(f"Is opt? {opt}: with Gap {mip_gap}")

new_row = pd.DataFrame({
    "seed": [seed],
    "I": [len(master.nurses)],
    "D": [len(master.days)],
    "S": [len(master.shifts)],
    "objective_value_cp": [round(obj_val_problem,2)],
    "objective_value_cg": [round(final_obj_cg, 2)],
    "time_cp": [round(time_problem,2)],
    "time_cg": [round(total_time_cg, 2)],
    "gap": [gap],
    "mip_gap": [mip_gap],
    "lowerbound": [bound],
    "optimal": [opt],
    "lagrange": [lagranigan_bound]

})
results_cg = pd.concat([results_cg, new_row], ignore_index=True)



results_cg.to_csv('cg.csv', index=False)
results_cg.to_excel('cg.xlsx', index=False)
