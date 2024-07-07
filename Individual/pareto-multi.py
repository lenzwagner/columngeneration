import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Updated parameter combinations
param1_values = [0.1, 0.2, 0.3]
param2_values = [10, 20, 30]

# Adjusted means for parameter combinations for both models (scaled to 0-1)
means_model1 = {
    (0.1, 10): (0.2, 0.75),
    (0.1, 20): (0.35, 0.6),
    (0.1, 30): (0.5, 0.55),
    (0.2, 10): (0.45, 0.5),
    (0.2, 20): (0.6, 0.45),
    (0.2, 30): (0.75, 0.4),
    (0.3, 10): (0.7, 0.35),
    (0.3, 20): (0.85, 0.3),
    (0.3, 30): (0.40, 0.25)
}

means_model2 = {
    (0.1, 10): (0.22, 0.77),
    (0.1, 20): (0.37, 0.62),
    (0.1, 30): (0.52, 0.57),
    (0.2, 10): (0.47, 0.52),
    (0.2, 20): (0.62, 0.47),
    (0.2, 30): (0.77, 0.42),
    (0.3, 10): (0.72, 0.37),
    (0.3, 20): (0.87, 0.32),
    (0.3, 30): (0.62, 0.17)
}

# Vordefinierte Standardabweichungen (scaled to 0-1)
std_dev_matrix = {
    ('BAM', 0.1, 10): (0.02, 0.002),
    ('BAM', 0.1, 20): (0.04, 0.05),
    ('BAM', 0.1, 30): (0.06, 0.04),
    ('BAM', 0.2, 10): (0.05, 0.05),
    ('BAM', 0.2, 20): (0.002, 0.04),
    ('BAM', 0.2, 30): (0.06, 0.03),
    ('BAM', 0.3, 10): (0.05, 0.04),
    ('BAM', 0.3, 20): (0.04, 0.03),
    ('BAM', 0.3, 30): (0.06, 0.02),
    ('NPM', 0.1, 10): (0.05, 0.06),
    ('NPM', 0.1, 20): (0.04, 0.05),
    ('NPM', 0.1, 30): (0.06, 0.04),
    ('NPM', 0.2, 10): (0.05, 0.05),
    ('NPM', 0.2, 20): (0.04, 0.04),
    ('NPM', 0.2, 30): (0.06, 0.03),
    ('NPM', 0.3, 10): (0.05, 0.04),
    ('NPM', 0.3, 20): (0.04, 0.03),
    ('NPM', 0.3, 30): (0.06, 0.02),
}

# Generate data
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

# Create std_df from std_dev_matrix
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

# Calculate Pareto frontier for the means (globally across all means)
costs = mean_df[['Metrik1', 'Metrik2']].values
pareto_efficient = is_pareto_efficient(costs)
pareto_front = mean_df[pareto_efficient]

# Sort Pareto frontier by Metrik1
pareto_front = pareto_front.sort_values('Metrik1')

# Nice color palette for different combinations
palette = sns.color_palette("magma", len(mean_df['Combination'].unique()))
colors = dict(zip(mean_df['Combination'].unique(), palette))

plt.figure(figsize=(16, 8))  # Vergrößerte Breite für die Legende außerhalb

# Plot all points and store handles and labels
handles = []
labels = []

for key, grp in mean_df.groupby(['Model', 'Combination']):
    model, combination = key
    std_values = std_df[(std_df['Model'] == model) & (std_df['Combination'] == combination)]

    if model == 'BAM':
        marker = 'o'
        alpha = 1.0
        linestyle = '-'
        edgecolor = 'black'
    else:  # NPM
        marker = 's'
        alpha = 0.7
        linestyle = ':'
        edgecolor = 'gray'

    base_color = colors[combination]
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
pareto_line, = plt.plot(pareto_front['Metrik1'], pareto_front['Metrik2'], 'r--', linewidth=2,
                        label='Pareto-Frontier Line')

plt.xlabel('Scaled Undercoverage', fontsize=12)
plt.ylabel('Scaled Consistency (ø Shift Changes)', fontsize=12)

# Dynamically set x and y axis limits
x_min = mean_df['Metrik1'].min() - 0.05
x_max = mean_df['Metrik1'].max() + 0.05
y_min = mean_df['Metrik2'].min() - 0.05
y_max = mean_df['Metrik2'].max() + 0.05

plt.xlim(max(0, x_min), min(1, x_max))
plt.ylim(max(0, y_min), min(1, y_max))

# Add Pareto frontier to legend
handles.append(pareto_line)
labels.append('Pareto-Frontier Line')

# Create legend outside the plot
plt.legend(handles=handles, labels=labels, title='Models and Combinations:',
           loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()

