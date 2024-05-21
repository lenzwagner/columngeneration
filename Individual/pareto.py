import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Values
chi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]

consistency = {(chi, epsilon): np.random.uniform(0, 1) for chi in chi_values for epsilon in epsilon_values}
flexibility = {(chi, epsilon): np.random.uniform(0, 1) for chi in chi_values for epsilon in epsilon_values}
undercoverage = {(chi, epsilon): np.random.uniform(0, 1) for chi in chi_values for epsilon in epsilon_values}

# List
solutions = []
for chi, epsilon in consistency.keys():
    solutions.append((flexibility[(chi, epsilon)], consistency[(chi, epsilon)], undercoverage[(chi, epsilon)]))

pareto_front = []
for soln in solutions:
    not_dominated = True
    for other in solutions:
        if (other[0] >= soln[0] and other[1] >= soln[1] and other[2] < soln[2]):
            not_dominated = False
            break
    if not_dominated:
        pareto_front.append(soln)

# Plot
pareto_front = np.array(pareto_front)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pareto_front[:,0], pareto_front[:,1], pareto_front[:,2], c='b', marker='o', label='Pareto Front')
ax.set_xlabel('Flexibility')
ax.set_ylabel('Consistency')
ax.set_zlabel('Undercoverage')
plt.title('Pareto Frontier')
plt.legend()
plt.show()