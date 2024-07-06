import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Beispielhafte Datenstruktur für zwei Parameterkombinationen mit neun Modellinstanzen pro Kombination
data = {
    'undercoverage1': [10, 11, 9, 10, 12, 8, 10, 11, 9],
    'consistency1': [1, 1, 2, 2, 2, 3, 3, 3, 3],
    'chi1': [3] * 9,
    'epsilon1': [0.0] * 9,
    'undercoverage2': [20, 19, 21, 20, 18, 20, 19, 21, 20],
    'consistency2': [2, 2, 3, 3, 3, 4, 4, 4, 4],
    'chi2': [4] * 9,
    'epsilon2': [0.02] * 9,
}

# Erstelle DataFrames
df1 = pd.DataFrame({
    'undercoverage': data['undercoverage1'],
    'consistency': data['consistency1'],
    'chi': data['chi1'],
    'epsilon': data['epsilon1']
})

df2 = pd.DataFrame({
    'undercoverage': data['undercoverage2'],
    'consistency': data['consistency2'],
    'chi': data['chi2'],
    'epsilon': data['epsilon2']
})

df = pd.concat([df1, df2], ignore_index=True)

# Mittelwerte und Standardabweichungen berechnen
agg_data = df.groupby(['chi', 'epsilon']).agg({
    'undercoverage': ['mean', 'std'],
    'consistency': ['mean', 'std']
}).reset_index()
agg_data.columns = ['chi', 'epsilon', 'mean_undercoverage', 'std_undercoverage', 'mean_consistency', 'std_consistency']

# Pareto-Frontier berechnen
def pareto_frontier(df):
    pareto_front = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if (other['mean_undercoverage'] <= row['mean_undercoverage'] and other['mean_consistency'] <= row['mean_consistency']) and (
                    other['mean_undercoverage'] < row['mean_undercoverage'] or other['mean_consistency'] < row['mean_consistency']):
                dominated = True
                break
        if not dominated:
            pareto_front.append(row)
    pareto_front_df = pd.DataFrame(pareto_front)
    pareto_front_df = pareto_front_df.sort_values(by=['mean_undercoverage'])
    return pareto_front_df

pareto_df = pareto_frontier(agg_data)

# Plot erstellen
plt.figure(figsize=(14, 10))
colors = plt.cm.tab20.colors

# Double Boxplots hinzufügen
unique_params = df[['chi', 'epsilon']].drop_duplicates().sort_values(by=['chi', 'epsilon']).reset_index(drop=True)
positions = np.arange(len(unique_params)) * 2

for i, (index, params) in enumerate(unique_params.iterrows()):
    chi = params['chi']
    epsilon = params['epsilon']
    
    subset = df[(df['chi'] == chi) & (df['epsilon'] == epsilon)]
    
    # Boxplot für Undercoverage
    plt.boxplot(subset['undercoverage'], positions=[positions[i] - 0.2], widths=0.3, patch_artist=True,
                boxprops=dict(facecolor=colors[i % len(colors)], color='black'),
                medianprops=dict(color='black'))
    
    # Boxplot für Consistency
    plt.boxplot(subset['consistency'], positions=[positions[i] + 0.2], widths=0.3, patch_artist=True,
                boxprops=dict(facecolor=colors[i % len(colors)], color='black'),
                medianprops=dict(color='black'))

# Pareto-optimale Punkte hervorheben
for i, row in pareto_df.iterrows():
    plt.scatter(positions[i], row['mean_undercoverage'], color='red', edgecolors='black',
                linewidths=2, alpha=0.6, s=150, marker='o')
    plt.scatter(positions[i], row['mean_consistency'], color='red', edgecolors='black',
                linewidths=2, alpha=0.6, s=150, marker='o')

# Verbinde die Pareto-optimalen Punkte
plt.plot(positions[:len(pareto_df)], pareto_df['mean_undercoverage'], linestyle='-', marker='x', color='red', alpha=0.7)
plt.plot(positions[:len(pareto_df)], pareto_df['mean_consistency'], linestyle='-', marker='x', color='red', alpha=0.7)

# Legende außerhalb des Plots positionieren
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)

plt.xlabel('Parameterkombinationen')
plt.ylabel('Skalierte Werte')
plt.xticks(ticks=positions, labels=[f"($\chi={row['chi']}, \epsilon={row['epsilon']}$)" for _, row in unique_params.iterrows()], rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
