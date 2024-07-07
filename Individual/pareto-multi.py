import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example parameter combinations and instances
param1_values = [1, 2, 3]
param2_values = [10, 20, 30]
instances = 12

np.random.seed(123)

# Adjusted means for parameter combinations for both models
means_model1 = {
    (1, 10): (20, 75),
    (1, 20): (35, 60),
    (1, 30): (50, 55),
    (2, 10): (45, 50),
    (2, 20): (60, 45),
    (2, 30): (75, 40),
    (3, 10): (70, 35),
    (3, 20): (85, 30),
    (3, 30): (100, 25)
}

means_model2 = {
    (1, 10): (22, 77),
    (1, 20): (37, 62),
    (1, 30): (52, 57),
    (2, 10): (47, 52),
    (2, 20): (62, 47),
    (2, 30): (77, 42),
    (3, 10): (72, 37),
    (3, 20): (87, 32),
    (3, 30): (102, 27)
}

# Slightly increased standard deviation
std_dev = 8

# Generate data
data = {
    'Model': [],
    'Param1': np.repeat(param1_values, len(param2_values) * instances * 2),
    'Param2': np.tile(np.repeat(param2_values, instances), len(param1_values) * 2),
    'Metrik1': [],
    'Metrik2': [],
    'Instanz': np.tile(range(1, instances + 1), len(param1_values) * len(param2_values) * 2)
}

for param1 in param1_values:
    for param2 in param2_values:
        mean1, mean2 = means_model1[(param1, param2)]
        data['Model'].extend(['Model 1'] * instances)
        data['Metrik1'].extend(mean1 + np.random.randn(instances) * std_dev)
        data['Metrik2'].extend(mean2 + np.random.randn(instances) * std_dev)
        
        mean1, mean2 = means_model2[(param1, param2)]
        data['Model'].extend(['Model 2'] * instances)
        data['Metrik1'].extend(mean1 + np.random.randn(instances) * std_dev)
        data['Metrik2'].extend(mean2 + np.random.randn(instances) * std_dev)

df = pd.DataFrame(data)

# Pareto frontier function
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True
    return is_efficient

# Calculate mean and standard deviation
mean_df = df.groupby(['Model', 'Param1', 'Param2']).mean().reset_index()
std_df = df.groupby(['Model', 'Param1', 'Param2']).std().reset_index()

# Create combination column for unique colors
mean_df['Combination'] = mean_df.apply(lambda row: f'{int(row.Param1)}, {int(row.Param2)}', axis=1)
std_df['Combination'] = std_df.apply(lambda row: f'{int(row.Param1)}, {int(row.Param2)}', axis=1)

# Calculate Pareto frontier for the means (globally across all means)
costs = mean_df[['Metrik1', 'Metrik2']].values
pareto_efficient = is_pareto_efficient(costs)
pareto_front = mean_df[pareto_efficient]

# Sort Pareto frontier by Metrik1
pareto_front = pareto_front.sort_values('Metrik1')

# Nice color palette for different combinations
palette = sns.color_palette("magma", len(mean_df['Combination'].unique()))
colors = dict(zip(mean_df['Combination'].unique(), palette))

plt.figure(figsize=(12, 6))  # Enlarged width for the legend

# Plot all points
for key, grp in mean_df.groupby(['Model', 'Combination']):
    key_str = key if isinstance(key, str) else key[0]
    alpha = 1.0 if key[0] == 'Model 1' else 0.5  # Set alpha value based on the model
    plt.errorbar(grp['Metrik1'], grp['Metrik2'], 
                 xerr=std_df[(std_df['Model'] == key[0]) & (std_df['Combination'] == key[1])]['Metrik1'], 
                 yerr=std_df[(std_df['Model'] == key[0]) & (std_df['Combination'] == key[1])]['Metrik2'], 
                 fmt='o', alpha=alpha, label=f'{key[0]}: {key[1]}', color=colors[key[1]])

# Plot Pareto frontier line
plt.plot(pareto_front['Metrik1'], pareto_front['Metrik2'], 'r--', linewidth=2, label='Pareto-Frontier Line')

plt.xlabel('Scaled Undercoverage')
plt.ylabel('Scaled Consistency (ø Shift Changes)')

# Place legend outside the box
plt.legend(title='Combinations', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to make space for the legend
plt.tight_layout()
plt.show()