import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np

# Data initialization
data = {
    'undercoverage1': [], 'consistency1': [], 'chi1': [], 'epsilon1': [],
    'undercoverage2': [], 'consistency2': [], 'chi2': [], 'epsilon2': []
}

# Reading data from CSV file
with open('data/data_new.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        data['undercoverage1'].append(float(row['undercover_norm']))
        data['undercoverage2'].append(float(row['undercover_norm_n']))
        data['consistency1'].append(float(row['cons_norm']))
        data['consistency2'].append(float(row['cons_norm_n']))
        data['chi1'].append(int(float(row['chi'])))  # Convert chi to int
        data['chi2'].append(int(float(row['chi'])))  # Convert chi to int
        data['epsilon1'].append(float(row['epsilon']))
        data['epsilon2'].append(float(row['epsilon']))

# Creating DataFrames
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

# Calculate Pareto Frontier
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

# Output Pareto-Frontier points to console
print("Punkte auf der Pareto-Frontier:")
print(pareto_df)

# Create plot
plt.figure(figsize=(12, 8))

# Adjusting the color range to focus on the brighter part of the magma palette
colors = plt.cm.magma(np.linspace(0.2, 0.8, max(len(df1), len(df2))))  # Using a subset of magma color palette

# Dictionary to store the labels to avoid duplication in the legend
labels_dict = {}

# Points from the first list (Circles)
for i, row in df1.iterrows():
    label = f"BAP $\\epsilon={row['epsilon']} / \\chi={int(row['chi'])}$"
    color_index = i % len(colors)
    if label not in labels_dict:
        labels_dict[label] = plt.scatter(row['undercoverage'], row['consistency'], color=colors[color_index], marker='o', s=100, label=label)
    else:
        plt.scatter(row['undercoverage'], row['consistency'], color=colors[color_index], marker='o', s=100)

# Points from the second list (Squares)
for i, row in df2.iterrows():
    label = f"NPP $\\epsilon={row['epsilon']} / \\chi={int(row['chi'])}$"
    color_index = i % len(colors)
    if label not in labels_dict:
        labels_dict[label] = plt.scatter(row['undercoverage'], row['consistency'], color=colors[color_index], marker='s', s=100, alpha=0.8, label=label)
    else:
        plt.scatter(row['undercoverage'], row['consistency'], color=colors[color_index], marker='s', s=100, alpha=0.8)

# Highlight Pareto-optimal points
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

# Connect Pareto-optimal points with dashed line
pareto_line, = plt.plot(pareto_df['undercoverage'], pareto_df['consistency'], linestyle='--', color='red', linewidth=2, alpha=0.7)

# Position the legend outside the plot
plt.legend(title='Combinations:', loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)

# Increase the font size of axis labels marginally
plt.xlabel('Scaled Undercoverage', fontsize=14)
plt.ylabel('Scaled Consistency (ø Shift Changes)', fontsize=14)
plt.title('Pareto-Frontier', fontsize=18)
plt.grid(True)

# Add Pareto frontier to legend
plt.legend(handles=list(labels_dict.values()) + [pareto_line], labels=list(labels_dict.keys()) + ['Pareto-Frontier Line'], title='Combinations:', loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)

# Set axis limits
plt.xlim(df['undercoverage'].min() - 1, df['undercoverage'].max() + 1)
plt.ylim(df['consistency'].min() - 1, df['consistency'].max() + 1)

plt.tight_layout()
plt.show()