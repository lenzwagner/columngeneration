import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.plotting import parallel_coordinates

import numpy as np

# Ihre Daten
data = {
    'understaff1': [[0.1, 0.2], [0.2, 0.1], [0.15, 0.01], [0.3, 0.2], [0.33, 0.25]],
    'viol1': [[0.22, 0.32], [0.12, 0.40], [0.31, 0.15], [0.23, 0.12], [0.34, 0.55]],
    'omega1': [[1, 2], [2, 2], [3, 2], [1, 3], [3, 1]],
    'gamma1': [[0.22, 0.22], [0.02, 0.04], [0.4, 0.25], [0.31, 0.32], [0.34, 0.25]],
    'understaff2': [[0.21, 0.12], [0.32, 0.13], [0.13, 0.4], [0.12, 0.23], [0.04, 0.35]],
    'viol2': [[0.32, 0.14], [0.12, 0.12], [0.18, 0.15], [0.25, 0.04], [0.32, 0.15]],
    'omega2': [[2, 2], [1, 0], [3, 4], [4, 0], [5, 3]],
    'gamma2': [[0.2, 0.12], [0.22, 0.14], [0.1, 0.05], [0.35, 0.02], [0.3, 0.05]]
}

# Mittelwerte für jede Parameterkombination berechnen
mean_understaff = []
mean_viol = []
mean_omega = []
mean_gamma = []

for i in range(5):  # 5 Parameterkombinationen
    understaff_values = data['understaff1'][i] + data['understaff2'][i]
    viol_values = data['viol1'][i] + data['viol2'][i]
    omega_values = data['omega1'][i] + data['omega2'][i]
    gamma_values = data['gamma1'][i] + data['gamma2'][i]
    
    mean_understaff.append(np.mean(understaff_values))
    mean_viol.append(np.mean(viol_values))
    mean_omega.append(np.mean(omega_values))
    mean_gamma.append(np.mean(gamma_values))

# Kombinieren der Mittelwerte
points = list(zip(mean_understaff, mean_viol))
params = list(zip(mean_omega, mean_gamma))

# Funktion zur Identifizierung Pareto-optimaler Punkte
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)
            is_efficient[i] = True
    return is_efficient

# Pareto-optimale Punkte identifizieren
pareto_points = np.array(points)[is_pareto_efficient(np.array(points))]

# Sortieren der Pareto-Punkte für die Frontier
pareto_frontier = sorted(pareto_points, key=lambda x: x[0])

print("Pareto Frontier:")
for point in pareto_frontier:
    print(f"Understaff: {point[0]:.3f}, Violations: {point[1]:.3f}")

# Berechnen der Pareto-Effizienz für jede Parameterkombination
pareto_efficiency = []
for point in points:
    efficiency = sum(1 for p in pareto_points if np.allclose(p, point))
    pareto_efficiency.append(efficiency)

print("\nPareto-Effizienz für jede Parameterkombination:")
for i, (eff, param) in enumerate(zip(pareto_efficiency, params)):
    print(f"Kombination {i+1}: Effizienz: {eff}, Omega: {param[0]:.2f}, Gamma: {param[1]:.2f}")


def create_heatmap(omega, gamma, efficiency):
    # Erstellen Sie ein DataFrame für die Heatmap
    df = pd.DataFrame({
        'Omega': omega,
        'Gamma': gamma,
        'Efficiency': efficiency
    })
    
    # Pivot-Tabelle erstellen
    pivot_table = df.pivot_table(values='Efficiency', index='Omega', columns='Gamma', aggfunc='first')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Heatmap der Pareto-Effizienz')
    plt.xlabel('Gamma')
    plt.ylabel('Omega')
    plt.show()

def create_parallel_coordinates(understaff, viol, omega, gamma, efficiency):
    data = {
        'Understaff': understaff,
        'Violations': viol,
        'Omega': omega,
        'Gamma': gamma,
        'Efficiency': efficiency
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    parallel_coordinates(df, 'Efficiency', colormap=plt.get_cmap("viridis"))
    plt.title('Parallel Coordinates Plot')
    plt.xlabel('Metrics and Parameters')
    plt.ylabel('Scaled Values')
    plt.legend(title='Efficiency', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

# Angenommen, Sie haben diese Werte aus Ihrem vorherigen Code:
# mean_understaff, mean_viol, mean_omega, mean_gamma, pareto_efficiency

# Aufruf der Funktionen
create_heatmap(mean_omega, mean_gamma, pareto_efficiency)
create_parallel_coordinates(mean_understaff, mean_viol, mean_omega, mean_gamma, pareto_efficiency)
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_pareto_frontier(understaff, viol, omega, gamma, pareto_frontier):
    plt.figure(figsize=(12, 8))
    
    # Alle Punkte plotten mit Parameterkombinationen in der Legende
    scatter_plots = []
    for i, (u, v, o, g) in enumerate(zip(understaff, viol, omega, gamma)):
        scatter = plt.scatter(u, v, label=f'Kombination {i+1} (ω={o:.2f}, γ={g:.2f})')
        scatter_plots.append(scatter)
    
    # Pareto-Frontier plotten
    pareto_understaff, pareto_viol = zip(*pareto_frontier)
    plt.plot(pareto_understaff, pareto_viol, 'r--', label='Pareto-Frontier')
    
    # Rote "X" für Pareto-Frontier-Punkte
    for u, v in pareto_frontier:
        plt.plot(u, v, 'rx', markersize=12, markeredgewidth=2)
    
    plt.xlabel('Understaff')
    plt.ylabel('Violations')
    plt.title('Pareto-Frontier mit Parameterkombinationen')
    
    # Legende erstellen
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Verwendung der Funktion (nach der Berechnung der Pareto-Frontier):
plot_pareto_frontier(mean_understaff, mean_viol, mean_omega, mean_gamma, pareto_frontier)