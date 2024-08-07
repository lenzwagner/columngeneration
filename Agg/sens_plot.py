import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np
import pandas as pd

def plot_data(option, file, metric, x_axis='epsilon', grid=True, legend_option=1):
    data = pd.read_csv(file)

    mpl.rcParams['font.family'] = 'Latin Modern Roman'
    mpl.rcParams['mathtext.fontset'] = 'cm'


    if metric not in ['cons', 'undercover']:
        print("Invalid metric. Please choose 'cons' or 'undercover'.")
        return

    if x_axis not in ['epsilon', 'chi']:
        print("Invalid x_axis. Please choose 'epsilon' or 'chi'.")
        return

    # Set column names based on the chosen metric
    y_col = f'{metric}_norm'
    y_col_n = f'{metric}_norm_n'

    # Set Seaborn style
    sns.set_theme(style="darkgrid" if grid else "whitegrid")

    # Create plot
    plt.figure(figsize=(14, 6))

    # Use the specified color palette
    colors = plt.cm.magma(np.linspace(0, 0.9, 5))

    if option == 2:
        # Group data by chi
        grouped = data.groupby('chi')

        for i, (chi, group) in enumerate(grouped):
            # Sort the group by epsilon
            group = group.sort_values('epsilon')

            # Calculate mean, min, and max for HSA and MSA
            HSA_mean = group[y_col].groupby(group['epsilon']).mean()
            HSA_min = group[y_col].groupby(group['epsilon']).min()
            HSA_max = group[y_col].groupby(group['epsilon']).max()
            MSA_mean = group[y_col_n].groupby(group['epsilon']).mean()
            MSA_min = group[y_col_n].groupby(group['epsilon']).min()
            MSA_max = group[y_col_n].groupby(group['epsilon']).max()

            # Plot mean points and connect them
            plt.plot(HSA_mean.index, HSA_mean.values, color=colors[i % len(colors)], marker='o',
                     label=f'HSA (χ={int(chi)})')
            plt.plot(MSA_mean.index, MSA_mean.values, color=colors[i % len(colors)], marker='s',
                     linestyle='--', label=f'MSA (χ={int(chi)})')

            # Add error bars
            HSA_error = plt.errorbar(HSA_mean.index, HSA_mean.values,
                                     yerr=[HSA_mean.values - HSA_min.values, HSA_max.values - HSA_mean.values],
                                     fmt='none', capsize=5, color=colors[i % len(colors)])
            MSA_error = plt.errorbar(MSA_mean.index, MSA_mean.values,
                                     yerr=[MSA_mean.values - MSA_min.values, MSA_max.values - MSA_mean.values],
                                     fmt='none', capsize=5, color=colors[i % len(colors)], linestyle='--')
            # Set linestyle for MSA error bars
            MSA_error[-1][0].set_linestyle('--')

        plt.ylabel(f'{"Total Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"}', fontsize=14)
        plt.xlabel(r'Epsilon $\varepsilon$' if x_axis == 'epsilon' else r'$\chi$', fontsize=14)

    else:
        print("Only option 2 is supported for this plot type.")
        return

    # Adjust x-axis to start from a negative value and end at max(x_axis)*1.02
    x_min = data[x_axis].min()
    x_max = data[x_axis].max()
    plt.xlim(-0.002, x_max * 1.02)  # Start from -0.02 to give space on the left

    # Adjust y-axis to start from min*0.95
    y_min = min(data[y_col].min(), data[y_col_n].min())
    plt.ylim(y_min * 0.95, None)

    # Ensure chi values on the x-axis are displayed as integers if x_axis is 'chi'
    if x_axis == 'chi':
        plt.xticks(np.arange(0, x_max + 1, 1))  # Start from 0 for chi values
    else:
        # For epsilon, include 0 in the ticks
        ticks = [0] + list(data[x_axis].unique())
        plt.xticks(ticks)

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: (int(x[1].split('χ=')[1][:-1]), 'MSA' in x[1]))
    handles, labels = zip(*sorted_handles_labels)

    # Different legend placement options
    if legend_option == 1:
        # Option 1: Legend outside the plot on the right
        plt.legend(handles=handles, labels=labels, bbox_to_anchor=(0.02, 0.98), loc='upper left')
        plt.tight_layout()
    elif legend_option == 2:
        # Option 2: Legend inside the plot in the upper right corner
        plt.legend(handles=handles, labels=labels, bbox_to_anchor=(0.98, 0.75), loc='upper right')
        plt.tight_layout()
    elif legend_option == 3:
        # Option 3: Legend below the plot
        plt.legend(handles=handles, labels=labels, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4)
        plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust the bottom margin to accommodate the legend
    else:
        print("Invalid legend_option. Please choose 1, 2, or 3.")
        return

    # Display the plot
    plt.show()

# Example function calls with different legend options
#plot_data(2, 'data/data_sens.csv', 'undercover', x_axis='epsilon', grid=False, legend_option=1)
#plot_data(2, 'data/data_sens.csv', 'cons', x_axis='epsilon', grid=False, legend_option=2)


def plot_data_both(file, x_axis='epsilon', grid=True, legend_option_left=1, legend_position_right=(1.02, 1)):
    data = pd.read_csv(file)

    # Set Seaborn style
    sns.set_theme(style="darkgrid" if grid else "whitegrid")

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Use the specified color palette
    colors = plt.cm.magma(np.linspace(0.15, 0.85, len(data['chi'].unique())))

    metrics = ['undercover', 'cons']

    for ax, metric in zip(axs, metrics):
        y_col = f'{metric}_norm'
        y_col_n = f'{metric}_norm_n'

        # Group data by chi
        grouped = data.groupby('chi')

        for i, (chi, group) in enumerate(grouped):
            # Sort the group by epsilon
            group = group.sort_values('epsilon')

            # Calculate mean, min, and max for HSA and MSA
            HSA_mean = group[y_col].groupby(group['epsilon']).mean()
            HSA_min = group[y_col].groupby(group['epsilon']).min()
            HSA_max = group[y_col].groupby(group['epsilon']).max()
            MSA_mean = group[y_col_n].groupby(group['epsilon']).mean()
            MSA_min = group[y_col_n].groupby(group['epsilon']).min()
            MSA_max = group[y_col_n].groupby(group['epsilon']).max()

            # Plot HSA on the subplot
            ax.plot(HSA_mean.index, HSA_mean.values, color=colors[i % len(colors)], marker='o',
                    label=f'HSA (χ={int(chi)})')
            HSA_error = ax.errorbar(HSA_mean.index, HSA_mean.values,
                                    yerr=[HSA_mean.values - HSA_min.values, HSA_max.values - HSA_mean.values],
                                    fmt='none', capsize=5, color=colors[i % len(colors)])

            # Plot MSA on the subplot
            ax.plot(MSA_mean.index, MSA_mean.values, color=colors[i % len(colors)], marker='s',
                    linestyle='--', label=f'MSA (χ={int(chi)})', alpha=0.8)
            MSA_error = ax.errorbar(MSA_mean.index, MSA_mean.values,
                                    yerr=[MSA_mean.values - MSA_min.values, MSA_max.values - MSA_mean.values],
                                    fmt='none', capsize=5, color=colors[i % len(colors)], linestyle='--')
            # Set linestyle for MSA error bars
            MSA_error[-1][0].set_linestyle('--')

        # Set y-axis label
        ax.set_ylabel(f'{"Total Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"}',
                      fontsize=13)

        # Legend settings
        handles, labels = ax.get_legend_handles_labels()
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: (int(x[1].split('χ=')[1][:-1]), 'MSA' in x[1]))
        handles, labels = zip(*sorted_handles_labels)

        if ax == axs[0]:
            # Left plot legend
            if legend_option_left == 1:
                ax.legend(handles=handles, labels=labels, bbox_to_anchor=(0.02, 0.98), loc='upper left', fontsize=9)
            elif legend_option_left == 2:
                ax.legend(handles=handles, labels=labels, loc='upper right')
            elif legend_option_left == 3:
                ax.legend(handles=handles, labels=labels, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4)
            else:
                print("Invalid legend_option_left. Please choose 1, 2, or 3.")
                return
        else:
            # Right plot legend with custom position
            ax.legend(handles=handles, labels=labels, bbox_to_anchor=legend_position_right, fontsize=9, loc='upper left')

    # Adjust x-axis to start from a negative value and end at max(x_axis)*1.02
    x_min = data[x_axis].min()
    x_max = data[x_axis].max()
    for ax in axs:
        ax.set_xlim(-0.002, x_max * 1.02)  # Start from -0.002 to give space on the left

    # Adjust y-axes individually
    y_min_undercover = data['undercover_norm'].min()
    y_min_cons = data['cons_norm'].min()

    axs[0].set_ylim(y_min_undercover * 0.95, None)  # Left plot
    axs[1].set_ylim(y_min_cons * 0.95, None)  # Right plot

    # Ensure epsilon values include 0 in the ticks
    ticks = [0] + list(data[x_axis].unique())
    for ax in axs:
        ax.set_xticks(ticks)

    # Set a common xlabel for the figure
    fig.supxlabel(r'Epsilon $\varepsilon$', fontsize=14)

    plt.tight_layout()
    plt.savefig('images/sens_nr.svg', bbox_inches='tight')
    # Display the plot
    plt.show()


def plot_data_both_pattern(file, x_axis='epsilon', grid=True, legend_option_left=1, legend_position_right=(1.02, 1)):
    data = pd.read_csv(file)

    mpl.rcParams['font.family'] = 'Latin Modern Roman'
    mpl.rcParams['mathtext.fontset'] = 'cm'

    # Set Seaborn style
    sns.set_theme(style="darkgrid" if grid else "whitegrid")

    # Create subplots

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Use the specified color palette
    colors = plt.cm.magma(np.linspace(0.15, 0.85, len(data['chi'].unique())))

    metrics = ['undercover', 'cons']

    for ax, metric in zip(axs, metrics):
        y_col = f'{metric}_norm'
        y_col_n = f'{metric}_norm_n'

        # Group data by chi
        grouped = data.groupby('chi')

        for i, (chi, group) in enumerate(grouped):
            # Sort the group by epsilon
            group = group.sort_values('epsilon')

            # Calculate mean, min, and max for HSA and MSA, grouped by epsilon and pattern
            HSA_stats = group.groupby(['epsilon', 'pattern'])[y_col].agg(['mean', 'min', 'max'])
            MSA_stats = group.groupby(['epsilon', 'pattern'])[y_col_n].agg(['mean', 'min', 'max'])

            # Calculate overall mean for each epsilon
            HSA_mean = HSA_stats['mean'].groupby(level=0).mean()
            MSA_mean = MSA_stats['mean'].groupby(level=0).mean()

            # Calculate error bars
            HSA_yerr = [HSA_mean - HSA_stats['min'].groupby(level=0).min(),
                        HSA_stats['max'].groupby(level=0).max() - HSA_mean]
            MSA_yerr = [MSA_mean - MSA_stats['min'].groupby(level=0).min(),
                        MSA_stats['max'].groupby(level=0).max() - MSA_mean]

            # Plot HSA on the subplot
            ax.plot(HSA_mean.index, HSA_mean.values, color=colors[i % len(colors)], marker='o',
                    label=f'HSA (χ={int(chi)})')
            HSA_error = ax.errorbar(HSA_mean.index, HSA_mean.values, yerr=HSA_yerr,
                                    fmt='none', capsize=5, color=colors[i % len(colors)])

            # Plot MSA on the subplot
            ax.plot(MSA_mean.index, MSA_mean.values, color=colors[i % len(colors)], marker='s',
                    linestyle='--', label=f'MSA (χ={int(chi)})', alpha=0.8)
            MSA_error = ax.errorbar(MSA_mean.index, MSA_mean.values, yerr=MSA_yerr,
                                    fmt='none', capsize=5, color=colors[i % len(colors)], linestyle='--')
            # Set linestyle for MSA error bars
            MSA_error[-1][0].set_linestyle('--')

        # Set y-axis label
        ax.set_ylabel(f'{"Total Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"}',
                      fontsize=13)

        # Legend settings
        handles, labels = ax.get_legend_handles_labels()
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: (int(x[1].split('χ=')[1][:-1]), 'MSA' in x[1]))
        handles, labels = zip(*sorted_handles_labels)

        if ax == axs[0]:
            # Left plot legend
            if legend_option_left == 1:
                ax.legend(handles=handles, labels=labels, bbox_to_anchor=(0.02, 0.98), loc='upper left', fontsize=9)
            elif legend_option_left == 2:
                ax.legend(handles=handles, labels=labels, loc='upper right')
            elif legend_option_left == 3:
                ax.legend(handles=handles, labels=labels, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4)
            else:
                print("Invalid legend_option_left. Please choose 1, 2, or 3.")
                return
        else:
            # Right plot legend with custom position
            ax.legend(handles=handles, labels=labels, bbox_to_anchor=legend_position_right, fontsize=9, loc='upper left')

    # Adjust x-axis to start from a negative value and end at max(x_axis)*1.02
    x_min = data[x_axis].min()
    x_max = data[x_axis].max()
    for ax in axs:
        ax.set_xlim(-0.002, x_max * 1.02)  # Start from -0.002 to give space on the left

    # Adjust y-axes individually
    y_min_undercover = data['undercover_norm'].min()
    y_min_cons = data['cons_norm'].min()

    axs[0].set_ylim(y_min_undercover * 0.95, None)  # Left plot
    axs[1].set_ylim(y_min_cons * 0.95, None)  # Right plot

    # Ensure epsilon values include 0 in the ticks
    ticks = [0] + list(data[x_axis].unique())
    for ax in axs:
        ax.set_xticks(ticks)

    # Set a common xlabel for the figure
    fig.supxlabel(r'Epsilon $\varepsilon$', fontsize=14)

    plt.tight_layout()
    plt.savefig('images/sens_nr_pattern.svg', bbox_inches='tight')
    # Display the plot
    plt.show()

# Example function call
plot_data_both('data/data_sens_all.csv', x_axis='epsilon', grid=False, legend_option_left=1, legend_position_right=(0.8, 0.76))
plot_data_both_pattern('data/Relevant/sens_pat_all.csv', x_axis='epsilon', grid=False, legend_option_left=1, legend_position_right=(0.79, 0.68))

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np
import pandas as pd

def plot_combined(file1, file2, x_axis='epsilon', grid=True):
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)

    mpl.rcParams['font.family'] = 'Latin Modern Roman'
    mpl.rcParams['mathtext.fontset'] = 'cm'

    # Set Seaborn style
    sns.set_theme(style="darkgrid" if grid else "whitegrid")

    # Create 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), sharex=True)

    # Use the specified color palette
    colors = plt.cm.magma(np.linspace(0.15, 0.85, len(data1['chi'].unique())))

    metrics = ['undercover', 'cons']
    data_pairs = [(data1, 'HSA (χ={int(chi)})'), (data2, 'MSA (χ={int(chi)})')]

    for i, (ax_row, (data, plot_title)) in enumerate(zip(axs, data_pairs)):
        for ax, metric in zip(ax_row, metrics):
            y_col = f'{metric}_norm'
            y_col_n = f'{metric}_norm_n'

            # Group data by chi
            grouped = data.groupby('chi')

            for j, (chi, group) in enumerate(grouped):
                # Sort the group by epsilon
                group = group.sort_values('epsilon')

                # Calculate mean, min, and max for HSA and MSA, grouped by epsilon and pattern
                HSA_stats = group.groupby(['epsilon', 'pattern'])[y_col].agg(['mean', 'min', 'max'])
                MSA_stats = group.groupby(['epsilon', 'pattern'])[y_col_n].agg(['mean', 'min', 'max'])

                # Calculate overall mean for each epsilon
                HSA_mean = HSA_stats['mean'].groupby(level=0).mean()
                MSA_mean = MSA_stats['mean'].groupby(level=0).mean()

                # Calculate error bars
                HSA_yerr = [HSA_mean - HSA_stats['min'].groupby(level=0).min(),
                            HSA_stats['max'].groupby(level=0).max() - HSA_mean]
                MSA_yerr = [MSA_mean - MSA_stats['min'].groupby(level=0).min(),
                            MSA_stats['max'].groupby(level=0).max() - MSA_mean]

                # Plot HSA on the subplot
                ax.plot(HSA_mean.index, HSA_mean.values, color=colors[j % len(colors)], marker='o',
                        label=f'HSA (χ={int(chi)})')
                HSA_error = ax.errorbar(HSA_mean.index, HSA_mean.values, yerr=HSA_yerr,
                                        fmt='none', capsize=5, color=colors[j % len(colors)])

                # Plot MSA on the subplot
                ax.plot(MSA_mean.index, MSA_mean.values, color=colors[j % len(colors)], marker='s',
                        linestyle='--', label=f'MSA (χ={int(chi)})', alpha=0.8)
                MSA_error = ax.errorbar(MSA_mean.index, MSA_mean.values, yerr=MSA_yerr,
                                        fmt='none', capsize=5, color=colors[j % len(colors)], linestyle='--')
                # Set linestyle for MSA error bars
                MSA_error[-1][0].set_linestyle('--')

            # Set y-axis label
            ax.set_ylabel(f'{"Total Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"}',
                          fontsize=13)

            # Set plot title
            ax.set_title(f'{plot_title} - {metric.capitalize()}', fontsize=14)

            # Legend settings
            handles, labels = ax.get_legend_handles_labels()
            sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: (int(x[1].split('χ=')[1][:-1]), 'MSA' in x[1]))
            handles, labels = zip(*sorted_handles_labels)

            ax.legend(handles=handles, labels=labels, fontsize=9)

    # Adjust x-axis to start from a negative value and end at max(x_axis)*1.02
    x_min = min(data1[x_axis].min(), data2[x_axis].min())
    x_max = max(data1[x_axis].max(), data2[x_axis].max())
    for ax_row in axs:
        for ax in ax_row:
            ax.set_xlim(-0.002, x_max * 1.02)  # Start from -0.002 to give space on the left

    # Adjust y-axes individually
    y_min_undercover = min(data1['undercover_norm'].min(), data2['undercover_norm'].min())
    y_min_cons = min(data1['cons_norm'].min(), data2['cons_norm'].min())

    axs[0, 0].set_ylim(y_min_undercover * 0.95, None)  # Left plot, undercover
    axs[0, 1].set_ylim(y_min_cons * 0.95, None)  # Right plot, cons

    axs[1, 0].set_ylim(y_min_undercover * 0.95, None)  # Left plot, undercover
    axs[1, 1].set_ylim(y_min_cons * 0.95, None)  # Right plot, cons

    # Ensure epsilon values include 0 in the ticks
    ticks = [0] + list(data1[x_axis].unique()) + list(data2[x_axis].unique())
    ticks = sorted(set(ticks))
    for ax_row in axs:
        for ax in ax_row:
            ax.set_xticks(ticks)

    # Set a common xlabel for the figure
    fig.supxlabel(r'Epsilon $\varepsilon$', fontsize=14)

    # Set a common legend below the x-axis label
    handles, labels = fig.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Adjust the bottom to accommodate the legend
    plt.savefig('images/combined_plots.svg', bbox_inches='tight')
    plt.show()

# Example function call
plot_combined('data/data_sens.csv', 'data/Relevant/sens_pat.csv', x_axis='epsilon', grid=False)
