from masterproblem import *
from subproblem import *
from gcutil import *
from compactsolver import *

def column_generation_behavior(data, demand_dict, eps, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi, threshold, time_cg, I, T, K):
    # **** Column Generation ****
    # Prerequisites
    modelImprovable = True

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
        schedules = [X_schedules, Perf_schedules, P_schedules, Cons_schedules, Recovery_schedules, EUp_schedules, ELow_schedules, X1_schedules]

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
    master.finalSolve(time_cg)
    objValHistRMP.append(master.model.objval)

    # Calc Stats
    ls_sc = plotPerformanceList(Cons_schedules, master.printLambdas())
    ls_p = plotPerformanceList(Perf_schedules, master.printLambdas())
    understaffing1, u_results, sum_all_doctors, consistency2, consistency2_norm, understaffing1_norm, u_results_norm, sum_all_doctors_norm = master.calc_behavior(ls_p, ls_sc)

    res = {
        round(understaffing1, 3),
        u_results,
        sum_all_doctors,
        consistency2,
        consistency2_norm,
        round(understaffing1_norm, 2),
        u_results_norm,
        sum_all_doctors_norm
    }

    return round(understaffing1, 3), round(u_results, 3), round(sum_all_doctors, 3), round(consistency2, 3), round(consistency2_norm, 3), round(understaffing1_norm, 3), round(u_results_norm, 3), round(sum_all_doctors_norm, 3)