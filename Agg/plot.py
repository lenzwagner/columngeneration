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

    y_col = f'{metric}_norm'
    y_col_n = f'{metric}_norm_n'

    sns.set_theme(style="darkgrid" if grid else "whitegrid")

    pt_in = pt / 72
    width_plt = round(pt_in)
    height_plt = round((width_plt / 16) * 9)
    plt.figure(figsize=(11, 5))

    palette = palett

    # Define symbols
    epsilon_symbol = r'$\varepsilon$'
    chi_symbol = r'$\chi$'

    if option == 1:
        data_HSA = data.sort_values(x_axis)
        data_MSA = data.sort_values(x_axis)

        hsa_line, = plt.plot(data_HSA[x_axis], data_HSA[y_col], color=palette[0], label='HSA', linestyle='-')
        msa_line, = plt.plot(data_MSA[x_axis], data_MSA[y_col_n], color=palette[1], label='MSA', linestyle='--', alpha=0.8)

        sns.scatterplot(data=data, x=x_axis, y=y_col, color=palette[0], marker='o')
        sns.scatterplot(data=data, x=x_axis, y=y_col_n, color=palette[1], marker='s')

        plt.ylabel("Total Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes", fontsize=13)
        plt.xlabel(f"Epsilon {epsilon_symbol}" if x_axis == 'epsilon' else chi_symbol, fontsize=13)
        plt.title(
            "{} vs {}".format(
                "Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes",
                epsilon_symbol if x_axis == "epsilon" else chi_symbol
            ),
            fontsize=15
        )

    elif option == 2:
        other_axis = 'chi' if x_axis == 'epsilon' else 'epsilon'
        lines = []
        for i, val in enumerate(sorted(data[other_axis].unique())):
            val_data = data[data[other_axis] == val].sort_values(x_axis)
            if other_axis == 'chi':
                HSA_label = f'HSA ({chi_symbol}={int(val)})'
                MSA_label = f'MSA ({chi_symbol}={int(val)})'
            else:
                HSA_label = f'HSA ({epsilon_symbol}={val:.2f})'
                MSA_label = f'MSA ({epsilon_symbol}={val:.2f})'

            color = palette[(i + 2) % len(palette)]

            sns.scatterplot(data=val_data, x=x_axis, y=y_col, color=palette[0], marker='o', label=HSA_label)
            sns.scatterplot(data=val_data, x=x_axis, y=y_col_n, color=palette[1], marker='s', label=MSA_label)

            hsa_line, = plt.plot(val_data[x_axis], val_data[y_col], c=color, linestyle='-', alpha=1)
            msa_line, = plt.plot(val_data[x_axis], val_data[y_col_n], c=color, linestyle='--', alpha=0.8)

            lines.extend([hsa_line, msa_line])

        plt.ylabel("Total Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes", fontsize=13)
        plt.xlabel(f"Epsilon {epsilon_symbol}" if x_axis == 'epsilon' else chi_symbol, fontsize=13)

    else:
        print("Invalid option. Please choose 1 or 2.")
        return

    plt.xlim(-0.0015, data[x_axis].max() * 1.02)

    y_min = min(data[y_col].min(), data[y_col_n].min())
    plt.ylim(y_min * 0.95, None)

    if x_axis == 'chi':
        plt.xticks(np.arange(data[x_axis].min(), data[x_axis].max() + 1, 1))

    handles, labels = plt.gca().get_legend_handles_labels()

    unique_vals = sorted(set(label.split('=')[-1].strip(')') for label in labels))
    sorted_labels = []
    for val in unique_vals:
        for handle, label in zip(handles, labels):
            if val in label:
                sorted_labels.append((handle, label))

    new_handles, new_labels = zip(*sorted_labels)

    if option == 1:
        new_handles = [hsa_line, msa_line]
    else:
        new_handles = lines

    if option == 2 and metric == 'cons' and x_axis == 'epsilon':
        plt.legend(new_handles, new_labels, loc='center right', bbox_to_anchor=(0.98, 0.55))
    else:
        if metric == 'undercover' and x_axis == 'epsilon':
            plt.legend(new_handles, new_labels, loc='upper left')
        else:
            plt.legend(new_handles, new_labels, bbox_to_anchor=(1.005, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()

def plot_two_plots(option1, option2, file1, file2, metric1, metric2, x_axis1='epsilon', x_axis2='epsilon', grid=True):
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)

    for metric in [metric1, metric2]:
        if metric not in ['cons', 'undercover']:
            print("Invalid metric. Please choose 'cons' or 'undercover'.")
            return

    for x_axis in [x_axis1, x_axis2]:
        if x_axis not in ['epsilon', 'chi']:
            print("Invalid x_axis. Please choose 'epsilon' or 'chi'.")
            return

    y_col1 = f'{metric1}_norm'
    y_col_n1 = f'{metric1}_norm_n'
    y_col2 = f'{metric2}_norm'
    y_col_n2 = f'{metric2}_norm_n'

    sns.set_theme(style="darkgrid" if grid else "whitegrid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    palette = sns.color_palette("deep")

    # Define symbols
    epsilon_symbol = r'$\varepsilon$'
    chi_symbol = r'$\chi$'

    def plot_single(ax, option, data, x_axis, y_col, y_col_n, metric):
        if option == 1:
            data_HSA = data.sort_values(x_axis)
            data_MSA = data.sort_values(x_axis)

            ax.plot(data_HSA[x_axis], data_HSA[y_col], color=palette[0], label='HSA', linestyle='-')
            ax.plot(data_MSA[x_axis], data_MSA[y_col_n], color=palette[1], label='MSA', linestyle='-', alpha=0.8)

            sns.scatterplot(data=data, x=x_axis, y=y_col, color=palette[0], marker='o', ax=ax, legend=False)
            sns.scatterplot(data=data, x=x_axis, y=y_col_n, color=palette[1], marker='s', ax=ax, legend=False)

            ax.set_title(
                "{} vs {}".format(
                    "Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes",
                    epsilon_symbol if x_axis == "epsilon" else chi_symbol
                ),
                fontsize=15
            )

        elif option == 2:
            sns.scatterplot(data=data, x=x_axis, y=y_col, color=palette[0], marker='o', label='Trend-HSA', ax=ax, legend=False)
            sns.scatterplot(data=data, x=x_axis, y=y_col_n, color=palette[1], marker='s', label='Trend-MSA', ax=ax, legend=False)

            sns.regplot(data=data, x=x_axis, y=y_col, scatter=False, color=palette[0], ax=ax, ci=None)
            sns.regplot(data=data, x=x_axis, y=y_col_n, scatter=False, color=palette[1], ax=ax, ci=None)

            ax.set_title(
                "{} vs {} for different {} values".format(
                    "Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes",
                    epsilon_symbol if x_axis == "epsilon" else chi_symbol,
                    chi_symbol if x_axis == "epsilon" else epsilon_symbol
                ),
                fontsize=15
            )

            other_axis = 'chi' if x_axis == 'epsilon' else 'epsilon'
            for j, val in enumerate(sorted(data[other_axis].unique())):
                val_data = data[data[other_axis] == val].sort_values(x_axis)
                if other_axis == 'chi':
                    HSA_label = f'HSA ({chi_symbol}={int(val)})'
                    MSA_label = f'MSA ({chi_symbol}={int(val)})'
                else:
                    HSA_label = f'HSA ({epsilon_symbol}={val:.2f})'
                    MSA_label = f'MSA ({epsilon_symbol}={val:.2f})'

                color = palette[(j + 2) % len(palette)]

                sns.scatterplot(data=val_data, x=x_axis, y=y_col, color=color, marker='o', label=HSA_label, ax=ax, legend=False)
                sns.scatterplot(data=val_data, x=x_axis, y=y_col_n, color=color, marker='s', label=MSA_label, ax=ax, legend=False)

                ax.plot(val_data[x_axis], val_data[y_col], c=color, linestyle='-', alpha=0.7)
                ax.plot(val_data[x_axis], val_data[y_col_n], c=color, linestyle='--', alpha=0.7)

        else:
            print("Invalid option. Please choose 1 or 2.")
            return

        ax.set_xlim(data[x_axis].min() * 0.95, data[x_axis].max() * 1.02)

        y_min = min(data[y_col].min(), data[y_col_n].min())
        ax.set_ylim(y_min * 0.95, None)

        if x_axis == 'chi':
            ax.set_xticks(np.arange(data[x_axis].min(), data[x_axis].max() + 1, 1))

        ax.set_xlabel('')

    plot_single(ax1, option1, data1, x_axis1, y_col1, y_col_n1, metric1)
    ax1.set_ylabel("Total Undercoverage" if metric1 == "undercover" else "Ø Number of Shift Changes", fontsize=13)

    plot_single(ax2, option2, data2, x_axis2, y_col2, y_col_n2, metric2)
    ax2.set_ylabel("Total Undercoverage" if metric2 == "undercover" else "Ø Number of Shift Changes", fontsize=13)

    fig.text(0.5, 0.02, f"Epsilon {epsilon_symbol}" if x_axis1 == 'epsilon' else chi_symbol, ha='center', fontsize=13)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    def format_label(label):
        if 'χ=' in label:
            parts = label.split('χ=')
            value = float(parts[1])
            return f"{parts[0]}χ={int(value)}"
        return label

    labels1 = [format_label(label) for label in labels1]
    labels2 = [format_label(label) for label in labels2]

    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    unique_labels = []
    unique_handles = []
    for handle, label in zip(all_handles, all_labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    fig.legend(unique_handles, unique_labels, bbox_to_anchor=(0.91, 0.5), loc='center left', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(right=0.9, bottom=0.1)
    plt.show()

# Example function calls
plot_data(2, 'data/data_all.csv', 'varunder', 'undercover', 468, x_axis='epsilon', grid=False)
plot_data(2, 'data/data_all.csv', 'varcons', 'cons', 468, x_axis='epsilon', grid=False)
# Uncomment the following lines if you want to run these plots
# plot_data(1, 'data/data3.csv', 'undercover', x_axis='epsilon', grid=False)
# plot_data(1, 'data/data3.csv', 'cons', x_axis='epsilon', grid=False)
# plot_data(1, 'data/data2.csv', 'undercover', x_axis='chi')
# plot_data(2, 'data/data.csv', 'undercover', x_axis='chi')
# plot_data(1, 'data/data2.csv', 'cons', x_axis='chi')
# plot_data(2, 'data/data.csv', 'cons', x_axis='chi')
# plot_two_plots(2, 2, 'data/data.csv', 'data/data.csv', 'undercover', 'cons', x_axis1='epsilon', x_axis2='epsilon', grid=True)
# plot_two_plots(2, 2, 'data/data.csv', 'data/