import pandas as pd
import matplotlib.pyplot as plt
import csv

data = {
    'undercoverage1': [], 'consistency1': [], 'chi1': [], 'epsilon1': [],
    'undercoverage2': [], 'consistency2': [], 'chi2': [], 'epsilon2': []
}

with open('ihre_datei.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        data['undercoverage1'].append(float(row['under1']))
        data['undercoverage2'].append(float(row['under2']))
        data['consistency1'].append(float(row['cons1']))
        data['consistency2'].append(float(row['cons2']))
        data['chi1'].append(float(row['chi']))
        data['chi2'].append(float(row['chi']))
        data['epsilon1'].append(float(row['epsilon']))
        data['epsilon2'].append(float(row['epsilon']))

print(data)

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


# Pareto-Frontier berechnen
def pareto_frontier(df):
    pareto_front = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if (other['undercoverage'] <= row['undercoverage'] and other['consistency'] <= row['consistency']) and (
                    other['undercoverage'] < row['undercoverage'] or other['consistency'] < row['consistency']):
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

# Dictionary to store the labels to avoid duplication in the legend
labels_dict = {}

# Punkte aus der ersten Liste (Kreise)
for i, row in df1.iterrows():
    label = f"$\chi={row['chi']}, \epsilon={row['epsilon']}$"
    if label not in labels_dict:
        labels_dict[label] = plt.scatter(row['undercoverage'], row['consistency'], color=colors[i % len(colors)], marker='o', s=100, label=label)
    else:
        plt.scatter(row['undercoverage'], row['consistency'], color=colors[i % len(colors)], marker='o', s=100)

# Punkte aus der zweiten Liste (Quadrate)
for i, row in df2.iterrows():
    label = f"$\chi={row['chi']}, \epsilon={row['epsilon']}$"
    if label not in labels_dict:
        labels_dict[label] = plt.scatter(row['undercoverage'], row['consistency'], color=colors[i % len(colors)], marker='s', s=100, alpha=0.8, label=label)
    else:
        plt.scatter(row['undercoverage'], row['consistency'], color=colors[i % len(colors)], marker='s', s=100, alpha=0.8)

# Pareto-optimale Punkte hervorheben
for i, row in pareto_df.iterrows():
    index_in_df1 = df1[(df1['undercoverage'] == row['undercoverage']) &
                       (df1['consistency'] == row['consistency']) &
                       (df1['chi'] == row['chi']) &
                       (df1['epsilon'] == row['epsilon'])].index
    index_in_df2 = df2[(df2['undercoverage'] == row['undercoverage']) &
                       (df2['consistency'] == row['consistency']) &
                       (df2['chi'] == row['chi']) &
                       (df2['epsilon'] == row['epsilon'])].index

    if not index_in_df1.empty:
        color_index = index_in_df1[0] % len(colors)
        plt.scatter(row['undercoverage'], row['consistency'], color=colors[color_index], edgecolors='black',
                    linewidths=2, alpha=0.6, s=150, marker='o')
    if not index_in_df2.empty:
        color_index = index_in_df2[0] % len(colors)
        plt.scatter(row['undercoverage'], row['consistency'], color=colors[color_index], edgecolors='black',
                    linewidths=2, alpha=0.6, s=150, marker='s')

# Verbinde die Pareto-optimalen Punkte
plt.plot(pareto_df['undercoverage'], pareto_df['consistency'], linestyle='-', marker='x', color='red', alpha=0.7)

# Legende außerhalb des Plots positionieren
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)

plt.xlabel('Scaled Undercoverage')
plt.ylabel('Scaled Consistency (ø Shift Changes)')
plt.grid(True)
plt.tight_layout()
plt.show()