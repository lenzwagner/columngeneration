import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_data(file, x_axis='omega'):
    data = pd.read_csv(file)

    if x_axis not in ['omega', 'gamma']:
        print("Invalid x_axis. Please choose 'omega' or 'gamma'.")
        return

    y_col = 'price'
    y_col_n = 'price_n'

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 8))

    colors = plt.cm.magma(np.linspace(0, 0.9, 6))

    grouped = data.groupby('gamma')

    for i, (gamma, group) in enumerate(grouped):
        group = group.sort_values('omega')

        # Calculate mean, min, and max for M1 and M2
        m1_mean = group[y_col].groupby(group['omega']).mean()
        m1_min = group[y_col].groupby(group['omega']).min()
        m1_max = group[y_col].groupby(group['omega']).max()
        m2_mean = group[y_col_n].groupby(group['omega']).mean()
        m2_min = group[y_col_n].groupby(group['omega']).min()
        m2_max = group[y_col_n].groupby(group['omega']).max()

        # Plot mean points and connect them
        plt.plot(m1_mean.index, m1_mean.values, color=colors[i % len(colors)], marker='o',
                 label=f'M1 ({int(gamma)})')
        plt.plot(m2_mean.index, m2_mean.values, color=colors[i % len(colors)], marker='s',
                 linestyle='--', label=f'M2 ({int(gamma)})')

        # Add error bars
        plt.errorbar(m1_mean.index, m1_mean.values,
                     yerr=[m1_mean.values - m1_min.values, m1_max.values - m1_mean.values],
                     fmt='none', capsize=5, color=colors[i % len(colors)])
        plt.errorbar(m2_mean.index, m2_mean.values,
                     yerr=[m2_mean.values - m2_min.values, m2_max.values - m2_mean.values],
                     fmt='none', capsize=5, color=colors[i % len(colors)], linestyle='--')

    plt.ylabel('Price', fontsize=13)
    plt.xlabel(r'Omega $\omega$' if x_axis == 'omega' else r'$\gamma$', fontsize=13)

    x_min = data[x_axis].min()
    x_max = data[x_axis].max()
    plt.xlim(-0.002, x_max * 1.02)

    y_min = min(data[y_col].min(), data[y_col_n].min())
    plt.ylim(y_min * 0.95, None)

    if x_axis == 'gamma':
        plt.xticks(np.arange(0, x_max + 1, 1))
    else:
        ticks = [0] + list(data[x_axis].unique())
        plt.xticks(ticks)

    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: ('M2' in x[1], x[1]))
    handles, labels = zip(*sorted_handles_labels)

    plt.legend(handles=handles, labels=labels, bbox_to_anchor=(0.02, 0.98), loc='upper left')
    plt.tight_layout()

    plt.show()


# Example data
np.random.seed(42)

omega_values = [0, 0.15, 0.3, 0.4, 0.5, 0.1]
gamma_values = [1,2,3,4,5]

data = []

for gamma in gamma_values:
    for omega in omega_values:
        # M1 values
        m1_mean = 9 + omega * 10 + (gamma - 3) * 0.2
        m1_variation = 0.2 + omega * 0.5
        m1_values = np.random.normal(m1_mean, m1_variation, 10)

        # M2 values
        m2_mean = m1_mean + 1 + omega * 15 + (gamma - 3) * 0.3
        m2_variation = 0.3 + omega * 0.7
        m2_values = np.random.normal(m2_mean, m2_variation, 10)

        for m1, m2 in zip(m1_values, m2_values):
            data.append({
                'omega': omega,
                'gamma': gamma,
                'price': max(m1, 8),
                'price_n': max(m2, m1 + 0.1)
            })

df = pd.DataFrame(data)
df.to_csv('example_data_with_errors.csv', index=False)

plot_data('example_data_with_errors.csv')