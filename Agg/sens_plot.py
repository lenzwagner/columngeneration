import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_data(option, file, metric, x_axis='epsilon', grid=True, legend_option=1):
    data = pd.read_csv(file)

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
    plt.figure(figsize=(12, 8))

    # Use the specified color palette
    colors = plt.cm.magma(np.linspace(0, 0.9, 6))

    if option == 2:
        # Group data by chi
        grouped = data.groupby('chi')

        for i, (chi, group) in enumerate(grouped):
            # Sort the group by epsilon
            group = group.sort_values('epsilon')

            # Calculate mean, min, and max for BAP and NPP
            bap_mean = group[y_col].groupby(group['epsilon']).mean()
            bap_min = group[y_col].groupby(group['epsilon']).min()
            bap_max = group[y_col].groupby(group['epsilon']).max()
            npp_mean = group[y_col_n].groupby(group['epsilon']).mean()
            npp_min = group[y_col_n].groupby(group['epsilon']).min()
            npp_max = group[y_col_n].groupby(group['epsilon']).max()

            # Plot mean points and connect them
            plt.plot(bap_mean.index, bap_mean.values, color=colors[i % len(colors)], marker='o',
                     label=f'BAP (χ={int(chi)})')
            plt.plot(npp_mean.index, npp_mean.values, color=colors[i % len(colors)], marker='s',
                     linestyle='--', label=f'NPP (χ={int(chi)})')

            # Add error bars
            plt.errorbar(bap_mean.index, bap_mean.values,
                         yerr=[bap_mean.values - bap_min.values, bap_max.values - bap_mean.values],
                         fmt='none', capsize=5, color=colors[i % len(colors)])
            plt.errorbar(npp_mean.index, npp_mean.values,
                         yerr=[npp_mean.values - npp_min.values, npp_max.values - npp_mean.values],
                         fmt='none', capsize=5, color=colors[i % len(colors)], linestyle='--')

        plt.ylabel(f'{"Total Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"}', fontsize=13)
        plt.xlabel(r'Epsilon $\varepsilon$' if x_axis == 'epsilon' else r'$\chi$', fontsize=13)

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
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: ('NPP' in x[1], x[1]))
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
plot_data(2, 'data/data_sens.csv', 'undercover', x_axis='epsilon', grid=False, legend_option=1)
plot_data(2, 'data/data_sens.csv', 'cons', x_axis='epsilon', grid=False, legend_option=2)


def plot_data_both(file, x_axis='epsilon', grid=True, legend_option_left=1, legend_position_right=(1.02, 1)):
    data = pd.read_csv(file)

    # Set Seaborn style
    sns.set_theme(style="darkgrid" if grid else "whitegrid")

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)

    # Use the specified color palette
    colors = plt.cm.magma(np.linspace(0, 0.9, 6))

    metrics = ['undercover', 'cons']

    for ax, metric in zip(axs, metrics):
        y_col = f'{metric}_norm'
        y_col_n = f'{metric}_norm_n'

        # Group data by chi
        grouped = data.groupby('chi')

        for i, (chi, group) in enumerate(grouped):
            # Sort the group by epsilon
            group = group.sort_values('epsilon')

            # Calculate mean, min, and max for BAP and NPP
            bap_mean = group[y_col].groupby(group['epsilon']).mean()
            bap_min = group[y_col].groupby(group['epsilon']).min()
            bap_max = group[y_col].groupby(group['epsilon']).max()
            npp_mean = group[y_col_n].groupby(group['epsilon']).mean()
            npp_min = group[y_col_n].groupby(group['epsilon']).min()
            npp_max = group[y_col_n].groupby(group['epsilon']).max()

            # Plot BAP on the subplot
            ax.plot(bap_mean.index, bap_mean.values, color=colors[i % len(colors)], marker='o',
                    label=f'BAP (χ={int(chi)})')
            ax.errorbar(bap_mean.index, bap_mean.values,
                        yerr=[bap_mean.values - bap_min.values, bap_max.values - bap_mean.values],
                        fmt='none', capsize=5, color=colors[i % len(colors)])

            # Plot NPP on the subplot
            ax.plot(npp_mean.index, npp_mean.values, color=colors[i % len(colors)], marker='s',
                    linestyle='--', label=f'NPP (χ={int(chi)})')
            ax.errorbar(npp_mean.index, npp_mean.values,
                        yerr=[npp_mean.values - npp_min.values, npp_max.values - npp_mean.values],
                        fmt='none', capsize=5, color=colors[i % len(colors)], linestyle='--')

        # Set y-axis label
        ax.set_ylabel(f'{"Total Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"}',
                      fontsize=11)

        # Legend settings
        handles, labels = ax.get_legend_handles_labels()
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: ('NPP' in x[1], x[1]))
        handles, labels = zip(*sorted_handles_labels)

        if ax == axs[0]:
            # Left plot legend
            if legend_option_left == 1:
                ax.legend(handles=handles, labels=labels, bbox_to_anchor=(0.02, 0.98), loc='upper left', fontsize = 8)
            elif legend_option_left == 2:
                ax.legend(handles=handles, labels=labels, loc='upper right')
            elif legend_option_left == 3:
                ax.legend(handles=handles, labels=labels, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4)
            else:
                print("Invalid legend_option_left. Please choose 1, 2, or 3.")
                return
        else:
            # Right plot legend with custom position
            ax.legend(handles=handles, labels=labels, bbox_to_anchor=legend_position_right, fontsize = 8, loc='upper left')

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
    fig.supxlabel(r'Epsilon $\varepsilon$', fontsize=11)

    plt.tight_layout()
    plt.savefig('sens_nr.svg', bbox_inches='tight')
    # Display the plot
    plt.show()


# Example function call
plot_data_both('data/data_sens.csv', x_axis='epsilon', grid=False, legend_option_left=1, legend_position_right=(0.8, 0.76))