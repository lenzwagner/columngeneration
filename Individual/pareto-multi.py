import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from collections import defaultdict
import csv

# Read the CSV file containing mean values
data_mean = pd.read_csv('data_mean.csv')

# Extract unique values for parameters
param1_values = sorted(data_mean['epsilon'].unique().tolist())
param2_values = sorted(data_mean['chi'].unique().tolist())

# Helper function to format float values
def to_python_float(value):
    return float(format(value, '.4f'))

# Initialize dictionaries to store mean values
means_model1 = {}
means_model2 = {}

# Populate dictionaries with mean values from the CSV
for _, row in data_mean.iterrows():
    epsilon = to_python_float(row['epsilon'])
    chi = int(row['chi'])
    undercover = to_python_float(row['undercover'])
    consistency = to_python_float(row['consistency'])
    undercover_n = to_python_float(row['undercover_n'])
    consistency_n = to_python_float(row['consistency_n'])

    means_model1[(epsilon, chi)] = (undercover, consistency)
    means_model2[(epsilon, chi)] = (undercover_n, consistency_n)

# Read the CSV file containing standard deviation values
data_std = []
with open('data_std.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data_std.append(row)

# Initialize a dictionary to store standard deviation values
std_dev_matrix = defaultdict(dict)
for row in data_std:
    model = row['model']
    epsilon = float(row['epsilon'])
    chi = int(row['chi'])
    undercover_std = float(row['undercover'])
    consistency_std = float(row['consistency'])

    std_dev_matrix[(model, epsilon, chi)] = (undercover_std, consistency_std)

# Output the standard deviation matrix
for key, std_devs in std_dev_matrix.items():
    model, epsilon, chi = key
    undercover_std, consistency_std = std_devs
    print(f"({model}, {epsilon}, {chi}): ({undercover_std:.4f}, {consistency_std:.4f})")

# Generate data for plotting
data = {
    'Model': [],
    'Param1': np.repeat(param1_values, len(param2_values) * 2),
    'Param2': np.tile(np.repeat(param2_values, 1), len(param1_values) * 2),
    'Metrik1': [],
    'Metrik2': []
}

for param1 in param1_values:
    for param2 in param2_values:
        mean1, mean2 = means_model1[(param1, param2)]
        data['Model'].append('BAM')
        data['Metrik1'].append(mean1)
        data['Metrik2'].append(mean2)

        mean1, mean2 = means_model2[(param1, param2)]
        data['Model'].append('NPM')
        data['Metrik1'].append(mean1)
        data['Metrik2'].append(mean2)

df = pd.DataFrame(data)

# Pareto frontier function
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True
    return is_efficient

# Calculate mean
mean_df = df.groupby(['Model', 'Param1', 'Param2']).mean().reset_index()

# Create a DataFrame for the standard deviations
std_df = pd.DataFrame(columns=['Model', 'Param1', 'Param2', 'Metrik1_std', 'Metrik2_std'])
for key, (std1, std2) in std_dev_matrix.items():
    new_row = pd.DataFrame({
        'Model': [key[0]],
        'Param1': [key[1]],
        'Param2': [key[2]],
        'Metrik1_std': [std1],
        'Metrik2_std': [std2]
    })
    std_df = pd.concat([std_df, new_row], ignore_index=True)

# Create combination column for unique colors
mean_df['Combination'] = mean_df.apply(lambda row: f'$\\varepsilon={row.Param1:.1f}$ / $\\chi={int(row.Param2)}$', axis=1)
std_df['Combination'] = std_df.apply(lambda row: f'$\\varepsilon={row.Param1:.1f}$ / $\\chi={int(row.Param2)}$', axis=1)

# Calculate Pareto frontier for the means
costs = mean_df[['Metrik1', 'Metrik2']].values
pareto_efficient = is_pareto_efficient(costs)
pareto_front = mean_df[pareto_efficient]

# Sort Pareto frontier by Metrik1
pareto_front = pareto_front.sort_values('Metrik1')

# Color palette for different combinations
palette = sns.color_palette("magma", len(mean_df['Combination'].unique()))
colors = dict(zip(mean_df['Combination'].unique(), palette))

plt.figure(figsize=(16, 8))  # Increased width for legend outside

# Plot all points and store handles and labels
handles = []
labels = []

for key, grp in mean_df.groupby(['Model', 'Combination']):
    model, combination = key
    std_values = std_df[(std_df['Model'] == model) & (std_df['Combination'] == combination)]

    base_color = colors[combination]

    if model == 'BAM':
        marker = 'o'
        alpha = 1.0
        linestyle = '-'
        edgecolor = 'none'  # No edge color for non-Pareto points
    else:  # NPM
        marker = 's'
        alpha = 0.7
        linestyle = ':'
        edgecolor = 'none'  # No edge color for non-Pareto points

    color = (base_color[0], base_color[1], base_color[2], alpha)

    h = plt.errorbar(grp['Metrik1'], grp['Metrik2'],
                     xerr=std_values['Metrik1_std'],
                     yerr=std_values['Metrik2_std'],
                     fmt=marker, alpha=alpha, color=color,
                     ecolor=color, elinewidth=1, capsize=3,
                     linestyle=linestyle, markersize=8,
                     markeredgewidth=1.5, markeredgecolor=edgecolor)

    handles.append(h[0])
    labels.append(f'{model} {combination}')

# Plot Pareto frontier line
pareto_line, = plt.plot(pareto_front['Metrik1'], pareto_front['Metrik2'], 'r--', linewidth=2, label='Pareto-Frontier Line')

# Outline Pareto frontier points
for _, row in pareto_front.iterrows():
    model = row['Model']
    combination = row['Combination']
    std_values = std_df[(std_df['Model'] == model) & (std_df['Combination'] == combination)]

    if model == 'BAM':
        marker = 'o'
        alpha = 1.0
        edgecolor = 'black'
    else:  # NPM
        marker = 's'
        alpha = 0.7
        edgecolor = 'gray'

    base_color = colors[combination]
    color = (base_color[0], base_color[1], base_color[2], alpha)

    plt.errorbar(row['Metrik1'], row['Metrik2'],
                 xerr=std_values['Metrik1_std'],
                 yerr=std_values['Metrik2_std'],
                 fmt=marker, alpha=alpha, color=color,
                 ecolor=color, elinewidth=1, capsize=3,
                 linestyle='None', markersize=8,
                 markeredgewidth=1.5, markeredgecolor=edgecolor)

plt.xlabel('Scaled Undercoverage', fontsize=12)
plt.ylabel('Scaled Consistency (ø Shift Changes)', fontsize=12)

# Dynamically set x and y axis limits
x_min = mean_df['Metrik1'].min() - 0.05
x_max = mean_df['Metrik1'].max() + 0.05
y_min = mean_df['Metrik2'].min() - 0.05
y_max = mean_df['Metrik2'].max() + 0.05

plt.xlim(max(0, x_min), min(1, x_max))
plt.ylim(max(0, y_min), min(1, y_max))

# Add grid to the plot
plt.grid(True)

# Add Pareto frontier to legend
handles.append(pareto_line)
labels.append('Pareto-Frontier Line')

# Create legend outside the plot
plt.legend(handles=handles, labels=labels, title='Combinations:', loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()
