import gurobipy as gu

class MasterProblem:
    def __init__(self, df, Demand, max_iteration, current_iteration, last, nr, start):
        self.iteration = current_iteration
        self.max_iteration = max_iteration
        self.nurses = df['I'].dropna().astype(int).unique().tolist()
        self.days = df['T'].dropna().astype(int).unique().tolist()
        self.shifts = df['K'].dropna().astype(int).unique().tolist()
        self._current_iteration = current_iteration
        self.roster = [i for i in range(1, self.max_iteration + 2)]
        self.rosterinitial = [i for i in range(1, 2)]
        self.demand = Demand
        self.model = gu.Model("MasterProblem")
        self.cons_demand = {}
        self.newvar = {}
        self.last_itr = last
        self.max_itr = max_iteration
        self.cons_lmbda = {}
        self.output_len = nr
        self.start_values = start
        self.demand_values = [self.demand[key] for key in self.demand.keys()]


    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.model.update()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.u = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.CONTINUOUS, lb=0, name='u')
        self.performance_i = self.model.addVars(self.nurses, self.days, self.shifts, self.roster,
                                               vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name='performance_i')
        self.lmbda = self.model.addVars(self.nurses, self.roster, vtype=gu.GRB.BINARY, lb=0, name='lmbda')

    def generateConstraints(self):
        for i in self.nurses:
            self.cons_lmbda[i] = self.model.addLConstr(1 == gu.quicksum(self.lmbda[i, r] for r in self.rosterinitial), name = "lmb("+str(i)+")")
        for t in self.days:
            for s in self.shifts:
                self.cons_demand[t, s] = self.model.addConstr(
                    gu.quicksum(self.performance_i[i, t, s, r]*self.lmbda[i, r] for i in self.nurses for r in self.rosterinitial) +
                    self.u[t, s] >= self.demand[t, s], "demand("+str(t)+","+str(s)+")")
        return self.cons_lmbda, self.cons_demand

    def generateObjective(self):
        self.model.setObjective(gu.quicksum(self.u[t, s] for t in self.days for s in self.shifts),
                                sense=gu.GRB.MINIMIZE)

    def getDuals_i(self):
        Pi_cons_lmbda = self.model.getAttr("Pi", self.cons_lmbda)
        return Pi_cons_lmbda

    def getDuals_ts(self):
        Pi_cons_demand = self.model.getAttr("QCPi", self.cons_demand)
        return Pi_cons_demand

    def updateModel(self):
        self.model.update()

    def setStartSolution(self):
        for i in self.nurses:
            for t in self.days:
                for s in self.shifts:
                    if (i, t, s) in self.start_values:
                        self.model.addLConstr(self.performance_i[i ,t, s, 1] == self.start_values[i, t, s])

    def addColumn(self, index, itr, schedule):
        self.nurseIndex = index
        self.rosterIndex = itr + 1
        for t in self.days:
            for s in self.shifts:
                qexpr = self.model.getQCRow(self.cons_demand[t, s])
                qexpr.add(schedule[self.nurseIndex, t, s, self.rosterIndex] * self.lmbda[self.nurseIndex, self.rosterIndex], 1)
                rhs = self.cons_demand[t, s].getAttr('QCRHS')
                sense = self.cons_demand[t, s].getAttr('QCSense')
                name = self.cons_demand[t, s].getAttr('QCName')
                newcon = self.model.addQConstr(qexpr, sense, rhs, name)
                self.model.remove(self.cons_demand[t, s])
                self.cons_demand[t, s] = newcon
        self.model.update()

    def printLambdas(self):
        return self.model.getAttr("X", self.lmbda)

    def addLambda(self, index, itr):
        self.nurseIndex = index
        self.rosterIndex = itr + 1
        self.newlmbcoef = 1.0
        current_lmb_cons = self.cons_lmbda[self.nurseIndex]
        expr = self.model.getRow(current_lmb_cons)
        new_lmbcoef = self.newlmbcoef
        expr.add(self.lmbda[self.nurseIndex, self.rosterIndex], new_lmbcoef)
        rhs_lmb = current_lmb_cons.getAttr('RHS')
        sense_lmb = current_lmb_cons.getAttr('Sense')
        name_lmb = current_lmb_cons.getAttr('ConstrName')
        newconlmb = self.model.addLConstr(expr, sense_lmb, rhs_lmb, name_lmb)
        self.model.remove(current_lmb_cons)
        self.cons_lmbda[self.nurseIndex] = newconlmb

    def finalSolve(self, timeLimit):
        try:
            self.model.setParam('TimeLimit', timeLimit)
            self.model.Params.IntegralityFocus = 1
            self.model.Params.FeasibilityTol = 1e-9
            self.model.Params.BarConvTol = 0.0
            self.model.Params.MIPGap = 1e-4
            self.model.Params.OutputFlag = 0
            self.model.setAttr("vType", self.lmbda, gu.GRB.BINARY)
            self.model.update()
            self.model.optimize()
            if self.model.status == gu.GRB.OPTIMAL:
                print("*" * (self.output_len + 2))
                print("*{:^{output_len}}*".format("***** Optimal solution found *****", output_len=self.output_len))
                print("*" * (self.output_len + 2))
            else:
                print("*" * (self.output_len + 2))
                print("*{:^{output_len}}*".format("***** No optimal solution found *****", output_len=self.output_len))
                print("*" * (self.output_len + 2))
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

    def solveModel(self, timeLimit):
        try:
            self.model.setParam('TimeLimit', timeLimit)
            self.model.Params.QCPDual = 1
            self.model.Params.OutputFlag = 0
            self.model.Params.IntegralityFocus = 1
            self.model.Params.FeasibilityTol = 1e-9
            self.model.Params.BarConvTol = 0.0
            self.model.Params.MIPGap = 1e-4
            self.model.setParam('ConcurrentMIP', 2)
            self.model.optimize()
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

    def solveRelaxModel(self):
        try:
            self.model.Params.OutputFlag = 0
            self.model.Params.MIPGap = 1e-6
            self.model.Params.Method = 2
            self.model.Params.Crossover = 0
            self.model.Params.QCPDual = 1
            for v in self.model.getVars():
                v.setAttr('vtype', 'C')
                v.setAttr('lb', 0.0)
            self.model.optimize()
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

    def calc_behavior(self, lst, ls_sc):
        consistency = sum(ls_sc)
        consistency_norm = sum(ls_sc) / (len(self.nurses)*len(self.days))
        sublist_length = len(lst) // len(self.nurses)
        p_values = [lst[i * sublist_length:(i + 1) * sublist_length] for i in range(len(self.nurses))]

        x_values = [[1.0 if value > 0 else 0.0 for value in sublist] for sublist in p_values]
        u_results = round(sum(self.u[t, k].x for t in self.days for k in self.shifts), 2)
        sum_xWerte = [sum(row[i] for row in x_values) for i in range(len(x_values[0]))]

        comparison_result = [
            max(0, self.demand_values[i] - sum_xWerte[i])
            for i in range(len(self.demand))
        ]

        understaffing = round(sum(comparison_result), 3)
        perf_loss = round(u_results - understaffing, 3)

        # Ausgabe
        print("\nUndercoverage: {:.2f}\nUnderstaffing: {:.2f}\nPerformance Loss: {:.2f}\nConsistency: {:.2f}\n".format(u_results,
                                                                                                  understaffing,
                                                                                                  perf_loss, consistency))

        return u_results, understaffing, perf_loss, consistency, consistency_norm

    def calc_naive(self, lst, ls_sc, ls_r, ls_e, ls_b, ls_x, mue):
        consistency = sum(ls_sc)
        u_results = round(sum(self.u[t, k].x for t in self.days for k in self.shifts), 2)

        consistency_norm = sum(ls_sc) / (len(self.nurses) * len(self.days))

        chunk_size = len(ls_sc) // len(self.nurses)
        chunk_size2 = len(ls_x) // len(self.nurses)
        self.sum_all_doctors = 0

        sublist_length = len(lst) // len(self.nurses)

        p_values = [lst[i * sublist_length:(i + 1) * sublist_length] for i in range(len(self.nurses))]



        x_values = [[1.0 if value > 0 else 0.0 for value in sublist] for sublist in p_values]

        u_results = round(sum(self.u[t, k].x for t in self.days for k in self.shifts), 2)

        sum_xWerte = [sum(row[i] for row in x_values) for i in range(len(x_values[0]))]

        self.sum_xWerte = sum_xWerte
        self.sum_all_doctors = 0

        print(f"x-vals:{self.sum_xWerte}")
        self.sum_values = sum(self.demand_values)
        self.cumulative_sum = [0]
        self.doctors_cumulative_multiplied = []
        self.vals = self.demand_values
        print(f"D-vals:{self.vals}")


        self.comp_result = []
        for i in range(len(self.vals)):
            if self.vals[i] < self.sum_xWerte[i]:
                self.comp_result.append(0)
            else:
                self.comp_result.append(1)
        print(f"Comps:{self.comp_result}")

        self.doctors_cumulative_multiplied = []
        for i in self.nurses:
            start_index = (i - 1) * chunk_size
            end_index = i * chunk_size

            start_index2 = (i - 1) * chunk_size2
            end_index2 = i * chunk_size2

            sublist_sc = ls_sc[start_index:end_index]
            sublist_r = ls_r[start_index:end_index]
            sublist_e = ls_e[start_index:end_index]
            sublist_b = ls_b[start_index:end_index]
            sublist_x = ls_x[start_index2:end_index2]

            doctor_values = sublist_sc
            r_values = sublist_r
            e_values = sublist_e
            b_values = sublist_b
            x_i_values = sublist_x

            print(f"B_{i} : {b_values}")
            print(f"E_{i} : {e_values}")
            print(f"R_{i} : {r_values}")
            print(f"C_{i} : {doctor_values}")

            self.cumulative_sum = [0]
            for i in range(1, len(doctor_values)):
                if r_values[i] == 1 and doctor_values[i] == 0 and self.cumulative_sum[-1] > 0:
                    self.cumulative_sum.append(self.cumulative_sum[-1] - 1)
                elif r_values[i] == 1 and doctor_values[i] == 1 and self.cumulative_sum[-1] > 0:
                    self.cumulative_sum.append(self.cumulative_sum[-1])
                elif r_values[i] == 1 and doctor_values[i] == 0 and self.cumulative_sum[-1] == 0:
                    self.cumulative_sum.append(self.cumulative_sum[-1])
                elif r_values[i] == 1 and doctor_values[i] == 1 and self.cumulative_sum[-1] == 0:
                    self.cumulative_sum.append(self.cumulative_sum[-1])
                else:
                    self.cumulative_sum.append(self.cumulative_sum[-1] + doctor_values[i])

            self.cumulative_sum2 = self.cumulative_sum.copy()
            modified_values = self.cumulative_sum.copy()
            reduction = 0

            for i in range(len(e_values)):
                if e_values[i] == 1.0:
                    if i == 0 or modified_values[i - 1] > 0:
                        reduction += 1

                modified_values[i] = self.cumulative_sum[i] - reduction


            self.cumulative_sum1 = []
            for element in modified_values:
                for _ in range(len(self.shifts)):
                    self.cumulative_sum1.append(element)

            print(f"Cums1  :{self.cumulative_sum}")
            print(f"Cums2  :{modified_values}")

            print(f"CumsFinal:{self.cumulative_sum1}")

            self.cumulative_values = [x * mue for x in self.cumulative_sum2]
            print(f"CumsVals:{self.cumulative_values}")

            self.multiplied_values = [self.cumulative_values[j] * x_i_values[j] for j in
                                      range(len(self.cumulative_values))]
            print(f"MulitVals:{self.multiplied_values}")

            self.multiplied_values1 = [self.multiplied_values[j] * self.comp_result[j] for j in
                                       range(len(self.multiplied_values))]
            print(f"FinalVals:{self.multiplied_values1}")

            self.total_sum = sum(self.multiplied_values1)
            self.doctors_cumulative_multiplied.append(self.total_sum)
            self.sum_all_doctors += self.total_sum

        self.understaffing1 = u_results + self.sum_all_doctors
        print("\nUndercoverage: {:.2f}\nUnderstaffing: {:.2f}\nPerformance Loss: {:.2f}\nConsistency: {:.2f}\n".format(
            self.understaffing1,
            u_results, self.sum_all_doctors, consistency))

        return self.understaffing1, u_results, self.sum_all_doctors, consistency, consistency_norm