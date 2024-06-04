from masterproblem import *
import time
from plots import *
from setup import *
from gcutil import *
from subproblem import *
from compactsolver import Problem

# **** Prerequisites ****
# Create Dataframes
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
results = pd.DataFrame(columns=["I", "D", "S", "objective_value", "time", "gap", "mip_gap", "chi", "epsilon", "consistency"])



# Ergebnisse ausgeben
print(results)
# Parameter
random.seed(13338)
time_Limit = 3600
max_itr = 20
output_len = 98
mue = 1e-4
threshold = 5e-7
eps = 0

# Demand Dict
demand_dict = generate_cost(len(T), len(I), len(K))


# **** Compact Solver ****
problem_t0 = time.time()
problem = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i)
problem.buildLinModel()
problem.updateModel()
problem.solveModel()

obj_val_problem = round(problem.model.objval, 3)
time_problem = time.time() - problem_t0
vals_prob = problem.get_final_values()
print(obj_val_problem)


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
problem_start.model.Params.MIPGap = 0.8
problem_start.model.update()
problem_start.model.optimize()
start_values_perf = {}
for i in I:
    for t in T:
        for s in K:
            start_values_perf[(i, t, s)] = problem_start.perf[i, t, s].x

start_values_p = {}
for i in I:
    for t in T:
        start_values_p[(i, t)] = problem_start.p[i, t].x

start_values_x = {}
for i in I:
    for t in T:
        for s in K:
            start_values_x[(i, t, s)] = problem_start.x[i, t, s].x

start_values_c = {}
for i in I:
    for t in T:
        start_values_c[(i, t)] = problem_start.sc[i, t].x

start_values_r = {}
for i in I:
    for t in T:
        start_values_r[(i, t)] = problem_start.r[i, t].x

start_values_eup = {}
for i in I:
    for t in T:
        start_values_eup[(i, t)] = problem_start.e[i, t].x

start_values_elow = {}
for i in I:
    for t in T:
        start_values_elow[(i, t)] = problem_start.b[i, t].x

while True:
    # Initialize iterations
    itr = 0
    t0 = time.time()
    last_itr = 0

    # Create empty results lists
    objValHistSP = []
    timeHist = []
    objValHistRMP = []
    avg_rc_hist = []
    lagrange_hist = []
    sum_rc_hist = []
    avg_sp_time = []
    gap_rc_hist = []

    X_schedules = {}
    for index in I:
        X_schedules[f"Physician_{index}"] = []

    start_values_perf_dict = {}
    for i in I:
        start_values_perf_dict[f"Physician_{i}"] = {(i, t, s): start_values_perf[(i, t, s)] for t in T for s in K}

    Perf_schedules = {}
    for index in I:
        Perf_schedules[f"Physician_{index}"] = [start_values_perf_dict[f"Physician_{index}"]]

    start_values_c_dict = {}
    for i in I:
        start_values_c_dict[f"Physician_{i}"] = {(i, t): start_values_c[(i, t)] for t in T}

    Cons_schedules = {}
    for index in I:
        Cons_schedules[f"Physician_{index}"] = [start_values_c_dict[f"Physician_{index}"]]

    start_values_r_dict = {}
    for i in I:
        start_values_r_dict[f"Physician_{i}"] = {(i, t): start_values_r[(i, t)] for t in T}

    Recovery_schedules = {}
    for index in I:
        Recovery_schedules[f"Physician_{index}"] = [start_values_r_dict[f"Physician_{index}"]]

    start_values_eup_dict = {}
    for i in I:
        start_values_eup_dict[f"Physician_{i}"] = {(i, t): start_values_eup[(i, t)] for t in T}

    EUp_schedules = {}
    for index in I:
        EUp_schedules[f"Physician_{index}"] = [start_values_eup_dict[f"Physician_{index}"]]

    start_values_elow_dict = {}
    for i in I:
        start_values_elow_dict[f"Physician_{i}"] = {(i, t): start_values_elow[(i, t)] for t in T}

    ELow_schedules = {}
    for index in I:
        ELow_schedules[f"Physician_{index}"] = [start_values_elow_dict[f"Physician_{index}"]]

    start_values_p_dict = {}
    for i in I:
        start_values_p_dict[f"Physician_{i}"] = {(i, t): start_values_p[(i, t)] for t in T}

    P_schedules = {}
    for index in I:
        P_schedules[f"Physician_{index}"] = [start_values_p_dict[f"Physician_{index}"]]

    start_values_x_dict = {}
    for i in I:
        start_values_x_dict[f"Physician_{i}"] = {(i, t, s): start_values_x[(i, t, s)] for t in T for s in K}

    X1_schedules = {}
    for index in I:
        X1_schedules[f"Physician_{index}"] = [start_values_x_dict[f"Physician_{index}"]]


    master = MasterProblem(data, demand_dict, max_itr, itr, last_itr, output_len, start_values_perf)
    master.buildModel()
    print("*" * (output_len + 2))
    print("*{:^{output_len}}*".format("Restricted Master Problem successfully built!", output_len=output_len))
    print("*" * (output_len + 2))

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

        # Get and Print Duals
        duals_i = master.getDuals_i()
        duals_ts = master.getDuals_ts()
        #print(f"DualsI: {duals_i}")
        #print(f"DualsTs: {duals_ts}")

        # Save current optimality gap
        gap_rc = round(((round(master.model.objval, 3) - round(obj_val_problem, 3)) / round(master.model.objval, 3)), 3)
        gap_rc_hist.append(gap_rc)

        # Solve SPs
        modelImprovable = False
        for index in I:
            print("*{:^{output_len}}*".format(f"Begin Column Generation Iteration {itr}", output_len=output_len))

            # Build SP
            subproblem = Subproblem(duals_i, duals_ts, data, index, itr, eps, Min_WD_i, Max_WD_i)
            subproblem.buildModel()

            # Save time to solve SP
            sub_t0 = time.time()
            subproblem.solveModel(time_Limit)
            sub_totaltime = time.time() - sub_t0
            timeHist.append(sub_totaltime)

            # Get optimal values
            optx_values = subproblem.getOptX()
            X_schedules[f"Physician_{index}"].append(optx_values)
            optPerf_values = subproblem.getOptPerf()
            Perf_schedules[f"Physician_{index}"].append(optPerf_values)
            optP_values = subproblem.getOptP()
            P_schedules[f"Physician_{index}"].append(optP_values)
            optc_values = subproblem.getOptC()
            Cons_schedules[f"Physician_{index}"].append(optc_values)
            optr_values = subproblem.getOptR()
            Recovery_schedules[f"Physician_{index}"].append(optr_values)

            opteup_values = subproblem.getOptEUp()
            EUp_schedules[f"Physician_{index}"].append(opteup_values)

            optelow_values = subproblem.getOptElow()

            ELow_schedules[f"Physician_{index}"].append(optelow_values)


            optf_values = subproblem.getOptF()
            optn_values = subproblem.getOptN()

            print(f"SC-sched {optc_values}")
            print(f"R-sched {optr_values}")
            print(f"F-sched {optf_values}")
            print(f"N-sched {optn_values}")
            print(f"B-sched {optelow_values}")


            optx1_values = subproblem.getOptX()
            X1_schedules[f"Physician_{index}"].append(optx1_values)

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
                master.addColumn(index, itr, Schedules)
                master.addLambda(index, itr)
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

        avg_time = sum(timeHist)/len(timeHist)
        avg_sp_time.append(avg_time)
        timeHist.clear()

        print("*{:^{output_len}}*".format(f"End Column Generation Iteration {itr}", output_len=output_len))

        if not modelImprovable:
            master.model.write("Final_LP.sol")
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

print(f" List of objvals relaxed {objValHistRMP}")
# Solve Master Problem with integrality restored
master.finalSolve(time_Limit)
master.model.write("Final_IP.sol")
objValHistRMP.append(master.model.objval)

# Capture total time and objval
total_time_cg = time.time() - t0
final_obj_cg = master.model.objval

#### Calculate Gap
# Relative to the lower bound (best possible achievable solution)
gap = ((objValHistRMP[-1]-objValHistRMP[-2])/objValHistRMP[-2])*100
mip_gap = round((((final_obj_cg-objValHistRMP[-2]) / objValHistRMP[-2]) * 100),3)

# Lagragian Bound
# Only yields feasible results if the SPs are solved to optimality
lagranigan_bound = round((objValHistRMP[-2] + sum_rc_hist[-1]), 3)
print(f"Lagrangian Bound {sum_rc_hist}")


### Store results
# Calculate Consistency
result, _ = total_consistency(master.printLambdas(), Cons_schedules)


new_row = pd.DataFrame({
    "I": [len(master.nurses)],
    "D": [len(master.days)],
    "S": [len(master.shifts)],
    "objective_value": [round(final_obj_cg,2)],
    "time": [round(total_time_cg,2)],
    "gap": [gap],
    "mip_gap": [mip_gap],
    "chi": [subproblem.chi],
    "epsilon": [subproblem.epsilon],
    "consistency": [result]
})
results_df = pd.concat([results, new_row], ignore_index=True)
results_df.to_excel('results.xlsx', index=False)
print(results_df)


# Print Results
printResults(itr, total_time_cg, time_problem, output_len, final_obj_cg, objValHistRMP[-2], lagranigan_bound, obj_val_problem, eps)

print(f"SP: {sum_rc_hist}")
print(f"MP: {objValHistRMP}")

# Plots
lagrangeprimal(sum_rc_hist, objValHistRMP, 'primal_dual_plot')
plot_obj_val(objValHistRMP, 'obj_val_plot')
plot_avg_rc(avg_rc_hist, 'rc_vals_plot')
performancePlot(plotPerformanceList(master.printLambdas(), P_schedules, I ,max_itr), len(T), len(I), 'perf_over_time')


### Calculate Metrics
# Before
ls_sc = plotPerformanceList(master.printLambdas(), Cons_schedules, I ,max_itr)
ls_r = plotPerformanceList(master.printLambdas(), Recovery_schedules, I ,max_itr)
ls_e = plotPerformanceList(master.printLambdas(), EUp_schedules, I ,max_itr)
ls_b = plotPerformanceList(master.printLambdas(), ELow_schedules, I ,max_itr)
ls_x = plotPerformanceList(master.printLambdas(), X_schedules, I ,max_itr)

# Behvaior
master.calc_behavior(plotPerformanceList(master.printLambdas(), Perf_schedules, I ,max_itr), ls_sc)

# Naive
master.calc_naive(plotPerformanceList(master.printLambdas(), Perf_schedules, I ,max_itr), ls_sc, ls_r, ls_e, ls_b, ls_x, 0.1)

print(f"Lambdas {master.printLambdas()}")