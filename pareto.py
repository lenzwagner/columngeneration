import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Beispielsdaten generieren (mit festen Werten, einschließlich nicht-Pareto-optimaler Punkte)
data = {
    'undercoverage': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.05],
    'consistency': [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.95],
    'variant': ['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10', 'var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10'],
    'chi': np.random.choice([3, 4, 5, 6], 20),
    'epsilon': np.random.choice([0, 0.02, 0.04, 0.06, 0.08, 0.1], 20)
}

df = pd.DataFrame(data)

# Funktion zur Berechnung der Pareto-Frontier
def pareto_frontier(Xs, Ys):
    sorted_list = sorted([[Xs.iloc[i], Ys.iloc[i]] for i in range(len(Xs))], key=lambda x: (x[0], -x[1]))
    p_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if pair[1] >= p_front[-1][1]:
            p_front.append(pair)
    return np.array(p_front)

# Farben für die Varianten
colors = plt.colormaps['tab20']

plt.figure(figsize=(12, 8))

# Plotten der Punkte und Pareto-Frontier
pareto_points = []
for idx, variant in enumerate(df['variant'].unique()):
    subset = df[df['variant'] == variant]
    Xs = subset['undercoverage']
    Ys = subset['consistency']

    p_front = pareto_frontier(Xs, Ys)
    pareto_points.append(p_front)

    plt.scatter(Xs, Ys, label=rf'$\epsilon={subset["epsilon"].values[0]}$ $\chi={subset["chi"].values[0]}$',
                color=colors(idx))
    plt.plot(p_front[:, 0], p_front[:, 1], linestyle='-', color=colors(idx), alpha=0.5)
    plt.scatter(p_front[:, 0], p_front[:, 1], color=colors(idx), edgecolor='black', s=100, marker='x')  # Pareto-Punkte kennzeichnen

plt.xlabel('Undercoverage')
plt.ylabel('Consistency')
plt.title('Pareto Frontier based on Undercoverage and Consistency')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Legende rechts platzieren
plt.show()

# Pareto-optimale Punkte ausgeben
for i, points in enumerate(pareto_points):
    print(f'Pareto-optimale Punkte für Variante {df["variant"].unique()[i]}:')
    print(points)
