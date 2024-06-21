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
        self.demand_values = [self.demand[key] for key in self.demand.keys()]
        self.start = start


    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.model.update()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.u = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.CONTINUOUS, lb=0, name='u')
        self.performance_i = self.model.addVars(self.days, self.shifts, self.roster,
                                               vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name='performance_i')
        self.lmbda = self.model.addVars(self.roster, vtype=gu.GRB.INTEGER, lb=0, name='lmbda')

    def generateConstraints(self):
        self.cons_lmbda = self.model.addLConstr(len(self.nurses) == gu.quicksum(self.lmbda[r] for r in self.rosterinitial), name = "lmb")
        for t in self.days:
            for s in self.shifts:
                self.cons_demand[t, s] = self.model.addConstr(
                    gu.quicksum(self.performance_i[t, s, r] * self.lmbda[r] for r in self.rosterinitial) +
                    self.u[t, s] >= self.demand[t, s], "demand("+str(t)+","+str(s)+")")
        return self.cons_lmbda, self.cons_demand

    def generateObjective(self):
        self.model.setObjective(gu.quicksum(self.u[t, s] for t in self.days for s in self.shifts),
                                sense=gu.GRB.MINIMIZE)

    def getDuals_i(self):
        Pi_cons_lmbda = self.cons_lmbda.Pi
        return Pi_cons_lmbda

    def getDuals_ts(self):
        Pi_cons_demand = self.model.getAttr("QCPi", self.cons_demand)
        return Pi_cons_demand

    def updateModel(self):
        self.model.update()

    def setStartSolution(self):
        for t in self.days:
            for s in self.shifts:
                if (t, s) in self.start:
                    self.model.addLConstr(self.performance_i[t, s, 1] == self.start[t, s])

    def addColumn(self, itr, schedule):
        self.rosterIndex = itr + 1
        for t in self.days:
            for s in self.shifts:
                qexpr = self.model.getQCRow(self.cons_demand[t, s])
                qexpr.add(schedule[t, s, self.rosterIndex] * self.lmbda[self.rosterIndex], 1)
                rhs = self.cons_demand[t, s].getAttr('QCRHS')
                sense = self.cons_demand[t, s].getAttr('QCSense')
                name = self.cons_demand[t, s].getAttr('QCName')
                newcon = self.model.addQConstr(qexpr, sense, rhs, name)
                self.model.remove(self.cons_demand[t, s])
                self.cons_demand[t, s] = newcon
        self.model.update()

    def printLambdas(self):
        return self.model.getAttr("X", self.lmbda)

    def addLambda(self, itr):
        self.rosterIndex = itr + 1
        self.newlmbcoef = 1.0
        current_lmb_cons = self.cons_lmbda
        expr = self.model.getRow(current_lmb_cons)
        new_lmbcoef = self.newlmbcoef
        expr.add(self.lmbda[self.rosterIndex], new_lmbcoef)
        rhs_lmb = current_lmb_cons.getAttr('RHS')
        sense_lmb = current_lmb_cons.getAttr('Sense')
        name_lmb = current_lmb_cons.getAttr('ConstrName')
        newconlmb = self.model.addLConstr(expr, sense_lmb, rhs_lmb, name_lmb)
        self.model.remove(current_lmb_cons)
        self.cons_lmbda = newconlmb

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
        consistency_norm = sum(ls_sc) / (len(self.nurses) * len(self.days))

        self.sum_all_doctors = 0
        sublist_length = len(lst) // len(self.nurses)
        sublist_length_short = len(ls_sc) // len(self.nurses)
        p_values = [lst[i * sublist_length:(i + 1) * sublist_length] for i in range(len(self.nurses))]
        sc_values2 = [ls_sc[i * sublist_length_short:(i + 1) * sublist_length_short] for i in range(len(self.nurses))]
        r_values2 = [ls_r[i * sublist_length_short:(i + 1) * sublist_length_short] for i in range(len(self.nurses))]
        e_values2 = [ls_e[i * sublist_length_short:(i + 1) * sublist_length_short] for i in range(len(self.nurses))]
        b_values2 = [ls_b[i * sublist_length_short:(i + 1) * sublist_length_short] for i in range(len(self.nurses))]
        x_values = [[1.0 if value > 0 else 0.0 for value in sublist] for sublist in p_values]
        u_results = round(sum(self.u[t, k].x for t in self.days for k in self.shifts), 2)
        sum_xWerte = [sum(row[i] for row in x_values) for i in range(len(x_values[0]))]

        self.sum_xWerte = sum_xWerte
        self.sum_all_doctors = 0

        self.sum_values = sum(self.demand_values)
        self.cumulative_sum = [0]
        self.doctors_cumulative_multiplied = []
        self.vals = self.demand_values

        self.comp_result = []
        for i in range(len(self.vals)):
            if self.vals[i] < self.sum_xWerte[i]:
                self.comp_result.append(0)
            else:
                self.comp_result.append(1)

        index = 0
        self.doctors_cumulative_multiplied = []
        for i in self.nurses:
            doctor_values = sc_values2[index]
            r_values = r_values2[index]
            e_values = e_values2[index]
            b_values = b_values2[index]
            x_i_values = x_values[index]
            index += 1

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

            reduction2 = 0
            modified_values2 = modified_values.copy()
            for i in range(len(b_values)):
                if b_values[i] == 1.0:
                    if i == 0 or modified_values2[i - 1] > 0:
                        reduction2 += 1

                modified_values2[i] = modified_values[i] + reduction2

            self.cumulative_sum1 = []
            for element in modified_values2:
                for _ in range(len(self.shifts)):
                    self.cumulative_sum1.append(element)

            self.cumulative_values = [x * mue for x in self.cumulative_sum1]
            self.multiplied_values = [self.cumulative_values[j] * x_i_values[j] for j in
                                      range(len(self.cumulative_values))]
            self.multiplied_values1 = [self.multiplied_values[j] * self.comp_result[j] for j in
                                       range(len(self.multiplied_values))]
            self.total_sum = sum(self.multiplied_values1)
            self.doctors_cumulative_multiplied.append(self.total_sum)
            self.sum_all_doctors += self.total_sum

            print(f"self.cumulative_values:{self.cumulative_values}")
            print(f"self.multiplied_values:{self.multiplied_values}")
            print(f"self.multiplied_values1:{self.multiplied_values1}")
            print(f"self.total_sum:{self.total_sum}")

        self.understaffing1 = u_results + self.sum_all_doctors
        print("\nUndercoverage: {:.2f}\nUnderstaffing: {:.2f}\nPerformance Loss: {:.2f}\nConsistency: {:.2f}\n".format(
            self.understaffing1,
            u_results, self.sum_all_doctors, consistency))

        return self.understaffing1, u_results, self.sum_all_doctors, consistency, consistency_norm