import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data preparation
epsilon = [0, 0.02, 0.04, 0.06, 0.08, 0.10]
chi = [3, 4, 5, 6, 7]

# Undercoverage data
data_BAP_undercoverage = np.array([
    [8.960, 8.960, 8.960, 8.960, 8.960],
    [9.002, 9.038, 9.064, 9.091, 9.126],
    [9.043, 9.120, 9.168, 9.227, 9.290],
    [9.084, 9.190, 9.273, 9.345, 9.433],
    [9.120, 9.261, 9.356, 9.447, 9.541],
    [9.157, 9.320, 9.423, 9.524, 9.631]
])

data_NPP_undercoverage = np.array([
    [8.960, 8.960, 8.960, 8.960, 8.960],
    [9.562, 9.856, 9.928, 9.978, 10.005],
    [10.165, 10.752, 10.897, 10.997, 11.050],
    [10.767, 11.648, 11.865, 12.015, 12.095],
    [11.369, 12.544, 12.834, 13.033, 13.140],
    [11.971, 13.440, 13.802, 14.051, 14.185]
])

# Consistency data
data_BAP_consistency = np.array([
    [7.090, 7.202, 7.601, 7.482, 7.119],
    [4.166, 3.980, 3.643, 3.613, 3.600],
    [4.310, 3.860, 3.640, 3.576, 3.545],
    [4.139, 3.891, 3.533, 3.447, 3.419],
    [4.207, 3.721, 3.399, 3.357, 3.250],
    [4.119, 3.657, 3.231, 3.090, 2.918]
])

data_NPP_consistency = np.full((6, 5), 7.326)

# 1. Heatmaps for Undercoverage
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(data_BAP_undercoverage, annot=True, fmt='.3f', cmap='Blues', ax=ax1)
ax1.set_title('BAP Undercoverage')
ax1.set_xlabel('χ')
ax1.set_ylabel('ε')
ax1.set_xticklabels(chi)
ax1.set_yticklabels(epsilon)

sns.heatmap(data_NPP_undercoverage, annot=True, fmt='.3f', cmap='Reds', ax=ax2)
ax2.set_title('NPP Undercoverage')
ax2.set_xlabel('χ')
ax2.set_ylabel('ε')
ax2.set_xticklabels(chi)
ax2.set_yticklabels(epsilon)

plt.tight_layout()
plt.savefig('heatmap_undercoverage.png')
plt.close()

# 2. Heatmaps for Consistency
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(data_BAP_consistency, annot=True, fmt='.3f', cmap='Greens', ax=ax1)
ax1.set_title('BAP Consistency')
ax1.set_xlabel('χ')
ax1.set_ylabel('ε')
ax1.set_xticklabels(chi)
ax1.set_yticklabels(epsilon)

sns.heatmap(data_NPP_consistency, annot=True, fmt='.3f', cmap='Oranges', ax=ax2)
ax2.set_title('NPP Consistency')
ax2.set_xlabel('χ')
ax2.set_ylabel('ε')
ax2.set_xticklabels(chi)
ax2.set_yticklabels(epsilon)

plt.tight_layout()
plt.savefig('heatmap_consistency.png')
plt.close()

# 3. Line Charts for Undercoverage
plt.figure(figsize=(12, 6))

for i, c in enumerate(chi):
    plt.plot(epsilon, data_BAP_undercoverage[:, i], label=f'BAP (χ={c})', marker='o')
    plt.plot(epsilon, data_NPP_undercoverage[:, i], label=f'NPP (χ={c})', marker='s', linestyle='--')

plt.xlabel('ε')
plt.ylabel('Undercoverage')
plt.title('Undercoverage for BAP and NPP')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('line_chart_undercoverage.png')
plt.close()

# 4. Line Charts for Consistency
plt.figure(figsize=(12, 6))

for i, c in enumerate(chi):
    plt.plot(epsilon, data_BAP_consistency[:, i], label=f'BAP (χ={c})', marker='o')
    plt.plot(epsilon, data_NPP_consistency[:, i], label=f'NPP (χ={c})', marker='s', linestyle='--')

plt.xlabel('ε')
plt.ylabel('Consistency')
plt.title('Consistency for BAP and NPP')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('line_chart_consistency.png')
plt.close()

print("All visualizations have been generated and saved as PNG files.")