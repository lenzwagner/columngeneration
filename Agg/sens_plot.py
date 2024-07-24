import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_data(option, file, metric, x_axis='epsilon', grid=True):
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

    # Use a Seaborn color palette
    palette = sns.color_palette("deep")

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
            plt.plot(bap_mean.index, bap_mean.values, color=palette[i % len(palette)], marker='o',
                     label=f'BAP (χ={int(chi)})')
            plt.plot(npp_mean.index, npp_mean.values, color=palette[i % len(palette)], marker='s',
                     linestyle='--', label=f'NPP (χ={int(chi)})')

            # Add error bars
            plt.errorbar(bap_mean.index, bap_mean.values,
                         yerr=[bap_mean.values - bap_min.values, bap_max.values - bap_mean.values],
                         fmt='none', capsize=5, color=palette[i % len(palette)])
            plt.errorbar(npp_mean.index, npp_mean.values,
                         yerr=[npp_mean.values - npp_min.values, npp_max.values - npp_mean.values],
                         fmt='none', capsize=5, color=palette[i % len(palette)], linestyle='--')

        plt.ylabel(f'{"Total Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"}', fontsize=13)
        plt.xlabel(r'Epsilon $\varepsilon$' if x_axis == 'epsilon' else r'$\chi$', fontsize=13)
        plt.title(f'{"Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"} vs {r"$\varepsilon$" if x_axis == "epsilon" else "χ"} for different {r"$\chi$" if x_axis == "epsilon" else r"$\varepsilon$"} values', fontsize=15)

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
    plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.005, 1), loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()

# Example function calls
plot_data(2, 'data/data_sens.csv', 'undercover', x_axis='epsilon', grid=False)
plot_data(2, 'data/data_sens.csv', 'cons', x_axis='epsilon', grid=False)