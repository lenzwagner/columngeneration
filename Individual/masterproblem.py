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

    def calc_behavior(self, lst):
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
        print("\nUndercoverage: {:.2f}\nUnderstaffing: {:.2f}\nPerformance Loss: {:.2f}\n".format(u_results,
                                                                                                  understaffing,
                                                                                                  perf_loss))

        return u_results, understaffing, perf_loss


    def calc_pr(self, x_values):

        sum_pWerte = []
        for i in range(len(x_values[0])):
            sum_value = sum(row[i] for row in x_values)
            sum_pWerte.append(sum_value)
        return sum_pWerte