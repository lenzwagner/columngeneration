import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Beispielsdaten erstellen
data = {
    'undercoverage': [10, 20, 30, 25, 15, 35, 40, 50, 45, 55, 60, 70, 65, 75, 80, 90, 85, 1, 100, 110,
                      95, 100, 105, 110, 115, 120, 125, 130, 135, 140],
    'consistency': [40, 35, 30, 32, 37, 25, 20, 15, 18, 12, 10, 7, 9, 6, 5, 3, 4, 2, 1, 0.5,
                    0.1, 42, 40, 38, 35, 33, 30, 28, 25, 22],
    'variant': [f'Var{i}' for i in range(1, 31)],
    'chi': [3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4],
    'epsilon': [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0, 0.02,
                0.04, 0.06, 0.08, 0.1, 0, 0.02, 0.04, 0.06, 0.08, 0.1]
}

df = pd.DataFrame(data)

# Pareto-Frontier berechnen
def pareto_frontier(df):
    pareto_front = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if (other['undercoverage'] <= row['undercoverage'] and other['consistency'] <= row['consistency']) and (other['undercoverage'] < row['undercoverage'] or other['consistency'] < row['consistency']):
                dominated = True
                break
        if not dominated:
            pareto_front.append(row)
    pareto_front_df = pd.DataFrame(pareto_front)
    pareto_front_df = pareto_front_df.sort_values(by=['undercoverage'])
    return pareto_front_df

pareto_df = pareto_frontier(df)

# Plot erstellen
plt.figure(figsize=(12, 8))
colors = plt.cm.tab20.colors

for i, row in df.iterrows():
    plt.scatter(row['undercoverage'], row['consistency'], color=colors[i % len(colors)], label=f"$\chi={row['chi']}, \epsilon={row['epsilon']}$")

for i, row in pareto_df.iterrows():
    plt.scatter(row['undercoverage'], row['consistency'], color=colors[i % len(colors)], edgecolors='black', linewidths=2, alpha=0.6, s=100)

# Verbinde die Pareto-optimalen Punkte
plt.plot(pareto_df['undercoverage'], pareto_df['consistency'], linestyle='-', marker='x', color='red', alpha=0.7)

# Legende außerhalb des Plots positionieren
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)

plt.xlabel('Undercoverage')
plt.ylabel('Consistency (ø Shift Changes)')
plt.title('Pareto Frontier')
plt.grid(True)
plt.tight_layout()
plt.show()