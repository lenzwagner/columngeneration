import gurobipy as gp
from gurobipy import GRB

# Beispieldaten
N = range(5)  # 5 Krankenschwestern
D = range(7)  # 7 Tage
S = range(3)  # 3 Schichten

# Zufällige Kosten und Anforderungen generieren
import random

c = {(n, d, s): random.randint(1, 10) for n in N for d in D for s in S}
R = {(d, s): random.randint(1, 3) for d in D for s in S}

# Initiale Lagrange-Multiplikatoren
lambda_vals = {(d, s): 1.0 for d in D for s in S}


# Funktion zum Lösen des relaxierten Problems
def solve_relaxed(lambda_vals):
    m = gp.Model("Nurse_Rostering_Relaxed")

    # Entscheidungsvariablen
    x = m.addVars(N, D, S, vtype=GRB.BINARY, name="x")

    # Zielfunktion
    obj = gp.quicksum(c[n, d, s] * x[n, d, s] for n in N for d in D for s in S)
    obj += gp.quicksum(lambda_vals[d, s] * (R[d, s] - gp.quicksum(x[n, d, s] for n in N)) for d in D for s in S)
    m.setObjective(obj, GRB.MINIMIZE)

    # Nebenbedingung: Höchstens eine Schicht pro Tag pro Krankenschwester
    m.addConstrs((gp.quicksum(x[n, d, s] for s in S) <= 1 for n in N for d in D), "one_shift_per_day")

    m.optimize()

    return m.objVal, {(n, d, s): x[n, d, s].X for n in N for d in D for s in S}


# Subgradient-Methode
max_iter = 100
step_size = 2.0
best_sol = float('inf')

for i in range(max_iter):
    obj_val, x_vals = solve_relaxed(lambda_vals)

    if obj_val < best_sol:
        best_sol = obj_val

    # Subgradienten berechnen
    subgradients = {(d, s): R[d, s] - sum(x_vals[n, d, s] for n in N) for d in D for s in S}
    print("")
    print("")
    print("")
    print("")
    print(f"Subgradient: in Iteration: {i} mit {best_sol}")
    print("")
    print("")
    print("")
    print("")
    # Lagrange-Multiplikatoren aktualisieren
    for d in D:
        for s in S:
            lambda_vals[d, s] = max(0, lambda_vals[d, s] + step_size * subgradients[d, s])

    step_size *= 0.95  # Schrittweite reduzieren

print(f"Beste gefundene Lösung: {best_sol}")