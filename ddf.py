from gurobipy import *
import gurobipy as gu
import pandas as pd
import time
import random

# Create Dataframes
import pandas as pd

I_list, T_list, K_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3]
I_list1 = pd.DataFrame(I_list, columns=['I'])
T_list1 = pd.DataFrame(T_list, columns=['T'])
K_list1 = pd.DataFrame(K_list, columns=['K'])
DataDF = pd.concat([I_list1, T_list1, K_list1], axis=1)
Demand_Dict = {(1, 1): 2, (1, 2): 1, (1, 3): 0, (2, 1): 1, (2, 2): 2, (2, 3): 0, (3, 1): 1, (3, 2): 1, (3, 3): 1,
(4, 1): 1, (4, 2): 2, (4, 3): 0, (5, 1): 2, (5, 2): 0, (5, 3): 1, (6, 1): 1, (6, 2): 1, (6, 3): 1,
(7, 1): 0, (7, 2): 3, (7, 3): 0}

# Generate Alpha
def gen_alpha(seed):
    random.seed(seed)
    return {(t): round(random.random(), 3) for t in range(1, 8)}

# General Parameter
max_itr, seed = 10, 123

class MasterProblem:
    def __init__(self, dfData, DemandDF, max_iteration, current_iteration):
        self.iteration = current_iteration
        self.max_iteration = max_iteration
        self.nurses = dfData['I'].dropna().astype(int).unique().tolist()
        self.days = dfData['T'].dropna().astype(int).unique().tolist()
        self.shifts = dfData['K'].dropna().astype(int).unique().tolist()
        self._current_iteration = current_iteration
        self.roster = [i for i in range(1, self.max_iteration + 2)]
        self.rosterinitial = [i for i in range(1, 2)]
        self.demand = DemandDF
        self.model = gu.Model("MasterProblem")
        self.cons_demand = {}
        self.cons_demand_2 = {}
        self.newvar = {}
        self.cons_lmbda = {}

    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.model.update()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.slack = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.CONTINUOUS, lb=0, name='slack')
        self.motivation_i = self.model.addVars(self.days, self.shifts, self.roster, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name='motivation_i')
        self.x_i = self.model.addVars(self.days, self.shifts, self.roster, vtype=gu.GRB.BINARY, name='x_i')
        self.lmbda = self.model.addVars(self.roster, vtype=gu.GRB.INTEGER, lb=0, name='lmbda')

    def generateConstraints(self):
        self.cons_lmbda = self.model.addConstr(len(self.nurses) == gu.quicksum(self.lmbda[r] for r in self.rosterinitial), name = "lmb")
        for t in self.days:
            for s in self.shifts:
                self.cons_demand[t, s] = self.model.addQConstr(gu.quicksum(self.motivation_i[t, s, r]*self.lmbda[r] for r in self.rosterinitial) + self.slack[t, s] >= self.demand[t, s], "demand("+str(t)+","+str(s)+")")
        return self.cons_lmbda, self.cons_demand

    def generateObjective(self):
        self.model.setObjective(gu.quicksum(self.slack[t, s] for t in self.days for s in self.shifts), sense=gu.GRB.MINIMIZE)

    def solveRelaxModel(self):
        self.model.Params.QCPDual = 1
        self.model.Params.NonConvex = 2
        for v in self.model.getVars():
            v.setAttr('vtype', 'C')
        self.model.optimize()

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
                self.model.addLConstr(0 == self.motivation_i[t, s, 1])
        self.model.update()

    def solveModel(self):
        self.model.Params.QCPDual = 1
        self.model.Params.OutputFlag = 0
        self.model.optimize()

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

    def finalSolve(self):
        self.model.setAttr("vType", self.lmbda, gu.GRB.INTEGER)
        self.model.update()
        self.model.optimize()


class Subproblem:
    def __init__(self, duals_i, duals_ts, dfData, M, iteration, alpha):
        itr = iteration + 1
        self.days = dfData['T'].dropna().astype(int).unique().tolist()
        self.shifts = dfData['K'].dropna().astype(int).unique().tolist()
        self.duals_i = duals_i
        self.duals_ts = duals_ts
        self.Max = 5
        self.Min = 2
        self.M = M
        self.alpha = alpha
        self.model = gu.Model("Subproblem")
        self.itr = itr

    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.x = self.model.addVars(self.days, self.shifts, [self.itr], vtype=gu.GRB.BINARY, name='x')
        self.y = self.model.addVars(self.days, vtype=gu.GRB.BINARY, name='y')
        self.mood = self.model.addVars(self.days, vtype=gu.GRB.CONTINUOUS, lb=0, name='mood')
        self.motivation = self.model.addVars(self.days, self.shifts, [self.itr], vtype=gu.GRB.CONTINUOUS, lb=0, name='motivation')

    def generateConstraints(self):
        for t in self.days:
            self.model.addLConstr(self.mood[t] == 1- self.alpha[t])
            self.model.addLConstr(quicksum(self.x[t, s, self.itr] for s in self.shifts) == self.y[t])
            self.model.addLConstr(gu.quicksum(self.x[t, s, self.itr] for s in self.shifts) <= 1)
            for s in self.shifts:
                self.model.addLConstr(self.motivation[t, s, self.itr] >= self.mood[t] - self.M * (1 - self.x[t, s, self.itr]))
                self.model.addLConstr(self.motivation[t, s, self.itr] <= self.mood[t] + self.M * (1 - self.x[t, s, self.itr]))
                self.model.addLConstr(self.motivation[t, s, self.itr] <= self.x[t, s, self.itr])
        for t in range(1, len(self.days) - self.Max + 1):
            self.model.addLConstr(gu.quicksum(self.y[u] for u in range(t, t + 1 + self.Max)) <= self.Max)
        self.model.addLConstr(self.Min <= quicksum(self.y[t] for t in self.days))

    def generateObjective(self):
        self.model.setObjective(0 - gu.quicksum(self.motivation[t, s, self.itr] * self.duals_ts[t, s] for t in self.days for s in self.shifts) - self.duals_i, sense=gu.GRB.MINIMIZE)

    def getNewSchedule(self):
        return self.model.getAttr("X", self.motivation)


#### Column Generation
modelImprovable = True
t0 = time.time()
itr = 0

# Build & Solve MP
master = MasterProblem(DataDF, Demand_Dict, max_itr, itr)
master.buildModel()
master.updateModel()
master.setStartSolution()
master.solveRelaxModel()

# Get Duals from MP
duals_i, duals_ts = master.getDuals_i(), master.getDuals_ts()

t0 = time.time()
while (modelImprovable) and itr < max_itr:
    # Start
    itr += 1
    print('* Current CG iteration: ', itr)

    # Solve RMP
    master.current_iteration = itr + 1
    master.solveRelaxModel()

    # Get Duals
    duals_i = master.getDuals_i()
    duals_ts = master.getDuals_ts()

    # Solve SPs
    modelImprovable = False
    subproblem = Subproblem(duals_i, duals_ts, DataDF,1e6, itr, gen_alpha(seed))
    subproblem.buildModel()
    subproblem.model.optimize()
    reducedCost = subproblem.model.objval

    if reducedCost < -1e-6:
        Schedules = subproblem.getNewSchedule()
        master.addColumn(itr, Schedules)
        master.addLambda(itr)
        master.updateModel()
        modelImprovable = True
    master.updateModel()

    if not modelImprovable:
        print("*{:^88}*".format("No more improvable columns found."))


# Solve MP
master.finalSolve()
print(master.model.objval)

for k in range(master.model.SolCount):
    master.model.setParam(gu.GRB.Param.SolutionNumber, k)
    vals = master.model.getAttr("Xn", master.lmbda)
    solution = {key: round(value) for key, value in vals.items()}

    print(f"Sol {k}: {solution}")
    # Überprüfen Sie die Zulässigkeit der Lösung
    sum_lambda = sum(solution.values())
    if abs(sum_lambda - 3) > 1e-6:
        print(f"Skipping infeasible solution {k}: sum of lambda = {sum_lambda}")
        continue  # Überspringen Sie diese Lösung und fahren Sie mit der nächsten fort

    print(f"Processing feasible solution {k}")
    print(f"Sol {k}: {solution}")