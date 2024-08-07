import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

palett = plt.cm.magma(np.linspace(0.15, 0.85, 3))


def plot_data(option, file, name, metric, pt, x_axis='epsilon', grid=True):
    file1 = str(name)
    file_name = f'.{os.sep}images{os.sep}{file1}.svg'
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
    pt_in = pt / 72
    width_plt = round(pt_in)
    height_plt = round((width_plt / 16) * 9)
    plt.figure(figsize=(12, 6))

    # Use a Seaborn color palette
    palette = palett

    if option == 1:
        # Sort data by the x_axis
        data_HSA = data.sort_values(x_axis)
        data_MSA = data.sort_values(x_axis)

        # Plot lines
        plt.plot(data_HSA[x_axis], data_HSA[y_col], color=palette[0], label='HSA', linestyle='-')
        plt.plot(data_MSA[x_axis], data_MSA[y_col_n], color=palette[1], label='MSA', linestyle='--', alpha=0.8)

        # Add scatter plots
        sns.scatterplot(data=data, x=x_axis, y=y_col, color=palette[0], marker='o')
        sns.scatterplot(data=data, x=x_axis, y=y_col_n, color=palette[1], marker='s')

        plt.ylabel(f'{"Total Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"}', fontsize=13)
        plt.xlabel(r'Epsilon $\varepsilon$' if x_axis == 'epsilon' else r'$\chi$', fontsize=13)
        plt.title(
            f'{"Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"} vs {r"$\varepsilon$" if x_axis == "epsilon" else "χ"}',
            fontsize=15)

    elif option == 2:
        # Additional points and lines for each value
        other_axis = 'chi' if x_axis == 'epsilon' else 'epsilon'
        for i, val in enumerate(sorted(data[other_axis].unique())):
            val_data = data[data[other_axis] == val].sort_values(x_axis)
            if other_axis == 'chi':
                HSA_label = f'HSA ({r"$\chi$"}={int(val)})'
                MSA_label = f'MSA ({r"$\chi$"}={int(val)})'
            else:
                HSA_label = f'HSA ({r"$\varepsilon$"}={val:.2f})'
                MSA_label = f'MSA ({r"$\varepsilon$"}={val:.2f})'

            # Use the same color for HSA and MSA with the same value
            color = palette[(i + 2) % len(palette)]  # Start from the third color in the palette

            sns.scatterplot(data=val_data, x=x_axis, y=y_col, color=color, marker='o', label=HSA_label)
            sns.scatterplot(data=val_data, x=x_axis, y=y_col_n, color=color, marker='s', label=MSA_label)

            # Add connecting lines with different styles
            plt.plot(val_data[x_axis], val_data[y_col], c=color, linestyle='-', alpha=1)
            plt.plot(val_data[x_axis], val_data[y_col_n], c=color, linestyle='--', alpha=0.8)

        plt.ylabel(f'{"Total Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"}', fontsize=13)
        plt.xlabel(r'Epsilon $\varepsilon$' if x_axis == 'epsilon' else r'$\chi$', fontsize=13)

    else:
        print("Invalid option. Please choose 1 or 2.")
        return

    # Adjust x-axis from 0 to max(x_axis)*1.2
    plt.xlim(-0.0015, data[x_axis].max() * 1.02)

    # Adjust y-axis to start from min*0.9
    y_min = min(data[y_col].min(), data[y_col_n].min())
    plt.ylim(y_min * 0.95, None)

    # Ensure chi values on the x-axis are displayed as integers if x_axis is 'chi'
    if x_axis == 'chi':
        plt.xticks(np.arange(data[x_axis].min(), data[x_axis].max() + 1, 1))

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()

    # Sort labels for alternation
    unique_vals = sorted(set(label.split('=')[-1].strip(')') for label in labels))
    sorted_labels = []
    for val in unique_vals:
        for handle, label in zip(handles, labels):
            if val in label:
                sorted_labels.append((handle, label))

    # Unpack the sorted labels
    new_handles, new_labels = zip(*sorted_labels)

    # Place the legend inside the plot for the specific case
    if option == 2 and metric == 'cons' and x_axis == 'epsilon':
        plt.legend(new_handles, new_labels, loc='center right', bbox_to_anchor=(0.98, 0.55))
    else:
        # Use the previous logic for other cases
        if metric == 'undercover' and x_axis == 'epsilon':
            plt.legend(new_handles, new_labels, loc='upper left')
        else:
            plt.legend(new_handles, new_labels, bbox_to_anchor=(1.005, 1), loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()

def plot_two_plots(option1, option2, file1, file2, metric1, metric2, x_axis1='epsilon', x_axis2='epsilon', grid=True):
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)

    # Validity checks for metric and x_axis
    for metric in [metric1, metric2]:
        if metric not in ['cons', 'undercover']:
            print("Invalid metric. Please choose 'cons' or 'undercover'.")
            return

    for x_axis in [x_axis1, x_axis2]:
        if x_axis not in ['epsilon', 'chi']:
            print("Invalid x_axis. Please choose 'epsilon' or 'chi'.")
            return

    # Set column names based on the chosen metric
    y_col1 = f'{metric1}_norm'
    y_col_n1 = f'{metric1}_norm_n'
    y_col2 = f'{metric2}_norm'
    y_col_n2 = f'{metric2}_norm_n'

    # Set Seaborn style
    sns.set_theme(style="darkgrid" if grid else "whitegrid")

    # Create a figure with 1 row and 2 columns, and extra space on the right for the legend
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Use a Seaborn color palette
    palette = sns.color_palette("deep")

    def plot_single(ax, option, data, x_axis, y_col, y_col_n, metric):
        if option == 1:
            # Sort data by the x_axis
            data_HSA = data.sort_values(x_axis)
            data_MSA = data.sort_values(x_axis)

            # Plot lines
            ax.plot(data_HSA[x_axis], data_HSA[y_col], color=palette[0], label='HSA', linestyle='-')
            ax.plot(data_MSA[x_axis], data_MSA[y_col_n], color=palette[1], label='MSA', linestyle='-', alpha = 0.8)

            # Add scatter plots
            sns.scatterplot(data=data, x=x_axis, y=y_col, color=palette[0], marker='o', ax=ax, legend=False)
            sns.scatterplot(data=data, x=x_axis, y=y_col_n, color=palette[1], marker='s', ax=ax, legend=False)

            ax.set_title(
                f'{"Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"} vs {r"$\varepsilon$" if x_axis == "epsilon" else "χ"}',
                fontsize=15)

        elif option == 2:
            # Plot scatter plots for overall trend
            sns.scatterplot(data=data, x=x_axis, y=y_col, color=palette[0], marker='o', label='Trend-HSA', ax=ax,
                            legend=False)
            sns.scatterplot(data=data, x=x_axis, y=y_col_n, color=palette[1], marker='s', label='Trend-MSA', ax=ax,
                            legend=False)

            # Add trend lines
            sns.regplot(data=data, x=x_axis, y=y_col, scatter=False, color=palette[0], ax=ax, ci=None)
            sns.regplot(data=data, x=x_axis, y=y_col_n, scatter=False, color=palette[1], ax=ax, ci=None)

            ax.set_title(
                f'{"Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"} vs {r"$\varepsilon$" if x_axis == "epsilon" else "χ"} for different {r"$\chi$" if x_axis == "epsilon" else r"$\varepsilon$"} values',
                fontsize=15)

            # Additional points and lines for each value
            other_axis = 'chi' if x_axis == 'epsilon' else 'epsilon'
            for j, val in enumerate(sorted(data[other_axis].unique())):
                val_data = data[data[other_axis] == val].sort_values(x_axis)
                if other_axis == 'chi':
                    HSA_label = f'HSA ({r"$\chi$"}={int(val)})'
                    MSA_label = f'MSA ({r"$\chi$"}={int(val)})'
                else:
                    HSA_label = f'HSA ({r"$\varepsilon$"}={val:.2f})'
                    MSA_label = f'MSA ({r"$\varepsilon$"}={val:.2f})'

                # Use the same color for HSA and MSA with the same value
                color = palette[(j + 2) % len(palette)]

                sns.scatterplot(data=val_data, x=x_axis, y=y_col, color=color, marker='o', label=HSA_label, ax=ax,
                                legend=False)
                sns.scatterplot(data=val_data, x=x_axis, y=y_col_n, color=color, marker='s', label=MSA_label, ax=ax,
                                legend=False)

                # Add connecting lines
                ax.plot(val_data[x_axis], val_data[y_col], c=color, linestyle='-', alpha=0.7)
                ax.plot(val_data[x_axis], val_data[y_col_n], c=color, linestyle='--', alpha=0.7)

        else:
            print("Invalid option. Please choose 1 or 2.")
            return

        # Adjust x-axis from 0 to max(x_axis)*1.2
        ax.set_xlim(data[x_axis].min() * 0.95, data[x_axis].max() * 1.02)

        # Adjust y-axis to start from min*0.9
        y_min = min(data[y_col].min(), data[y_col_n].min())
        ax.set_ylim(y_min * 0.95, None)

        # Ensure chi values on the x-axis are displayed as integers if x_axis is 'chi'
        if x_axis == 'chi':
            ax.set_xticks(np.arange(data[x_axis].min(), data[x_axis].max() + 1, 1))

        # Remove x-axis label for individual plots
        ax.set_xlabel('')

    # Plot the first plot
    plot_single(ax1, option1, data1, x_axis1, y_col1, y_col_n1, metric1)
    ax1.set_ylabel(f'{"Total Undercoverage" if metric1 == "undercover" else "Ø Number of Shift Changes"}', fontsize=13)

    # Plot the second plot
    plot_single(ax2, option2, data2, x_axis2, y_col2, y_col_n2, metric2)
    ax2.set_ylabel(f'{"Total Undercoverage" if metric2 == "undercover" else "Ø Number of Shift Changes"}', fontsize=13)

    # Add x-axis label
    fig.text(0.5, 0.02, r'Epsilon $\varepsilon$' if x_axis1 == 'epsilon' else r'$\chi$', ha='center', fontsize=13)

    # Collect legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Function to format labels
    def format_label(label):
        if 'χ=' in label:
            parts = label.split('χ=')
            value = float(parts[1])
            return f"{parts[0]}χ={int(value)}"
        return label

    # Format the labels
    labels1 = [format_label(label) for label in labels1]
    labels2 = [format_label(label) for label in labels2]

    # Merge handles and labels from both plots
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    # Remove duplicates
    unique_labels = []
    unique_handles = []
    for handle, label in zip(all_handles, all_labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    # Add a single legend for both plots to the right of the second plot
    fig.legend(unique_handles, unique_labels, bbox_to_anchor=(0.91, 0.5), loc='center left', fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Adjust the subplot layout to make room for the legend
    plt.subplots_adjust(right=0.9, bottom=0.1)

    # Display the plot
    plt.show()



# Example function call with grid option
# Example function calls
#plot_data(1, 'data/data3.csv', 'undercover', x_axis='epsilon', grid=False) # Epsilon on x-axis
plot_data(2, 'data/data_all.csv',  'varunder', 'undercover', 468,  x_axis='epsilon', grid=False) # Epsilon on x-axis
#plot_data(1, 'data/data3.csv', 'cons', x_axis='epsilon', grid=False) # Epsilon on x-axis
plot_data(2, 'data/data_all.csv', 'varcons', 'cons', 468, x_axis='epsilon', grid=False) # Epsilon on x-axis
#plot_data(1, 'data/data2.csv', 'undercover', x_axis='chi') # Chi on x-axis
#plot_data(2, 'data/data.csv', 'undercover', x_axis='chi') # Chi on x-axis
#plot_data(1, 'data/data2.csv', 'cons', x_axis='chi') # Chi on x-axis
#plot_data(2, 'data/data.csv', 'cons', x_axis='chi') # Chi on x-axis
#plot_two_plots(2, 2, 'data/data.csv', 'data/data.csv', 'undercover', 'cons', x_axis1='epsilon', x_axis2='epsilon', grid=True)
#plot_two_plots(2, 2, 'data/data.csv', 'data/data.csv', 'undercover', 'cons', x_axis1='chi', x_axis2='chi', grid=True)



