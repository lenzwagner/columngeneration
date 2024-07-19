import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_data(option, file, metric, x_axis='epsilon'):
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
    sns.set_theme(style="darkgrid")

    # Create plot
    plt.figure(figsize=(12, 8))

    # Use a Seaborn color palette
    palette = sns.color_palette("deep")

    if option == 1:
        # Sort data by the x_axis
        data_bap = data.sort_values(x_axis)
        data_npp = data.sort_values(x_axis)

        # Plot lines
        plt.plot(data_bap[x_axis], data_bap[y_col], color=palette[0], label='BAP', linestyle='-')
        plt.plot(data_npp[x_axis], data_npp[y_col_n], color=palette[1], label='NPP', linestyle='--')

        # Add scatter plots
        sns.scatterplot(data=data, x=x_axis, y=y_col, color=palette[0], marker='o')
        sns.scatterplot(data=data, x=x_axis, y=y_col_n, color=palette[1], marker='s')

        plt.ylabel(f'{"Total Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"}', fontsize=13)
        plt.xlabel(r'Epsilon $\varepsilon$' if x_axis == 'epsilon' else r'$\chi$', fontsize=13)
        plt.title(
            f'{"Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"} vs {"$\\varepsilon$" if x_axis == "epsilon" else "χ"}',
            fontsize=15)

    elif option == 2:
        # Plot scatter plots for overall trend
        sns.scatterplot(data=data, x=x_axis, y=y_col, color=palette[0], marker='o', label='Trend-BAP')
        sns.scatterplot(data=data, x=x_axis, y=y_col_n, color=palette[1], marker='s', label='Trend-NPP')

        # Add trend lines
        sns.regplot(data=data, x=x_axis, y=y_col, scatter=False, color=palette[0])
        sns.regplot(data=data, x=x_axis, y=y_col_n, scatter=False, color=palette[1])

        plt.ylabel(f'{"Total Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"}', fontsize=13)
        plt.xlabel(r'Epsilon $\varepsilon$' if x_axis == 'epsilon' else r'$\chi$', fontsize=13)
        plt.title(
            f'{"Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"} vs {"Epsilon" if x_axis == "epsilon" else "χ"} for different {r"$\chi$" if x_axis == "epsilon" else r"$\varepsilon$"} values',
            fontsize=15)

        # Additional points and lines for each value
        other_axis = 'chi' if x_axis == 'epsilon' else 'epsilon'
        for i, val in enumerate(sorted(data[other_axis].unique())):
            val_data = data[data[other_axis] == val].sort_values(x_axis)
            if other_axis == 'chi':
                bap_label = f'BAP ({r"$\chi$"}={int(val)})'
                npp_label = f'NPP ({r"$\chi$"}={int(val)})'
            else:
                bap_label = f'BAP ({r"$\varepsilon$"}={val:.2f})'
                npp_label = f'NPP ({r"$\varepsilon$"}={val:.2f})'

            # Use the same color for BAP and NPP with the same value
            color = palette[(i + 2) % len(palette)]

            sns.scatterplot(data=val_data, x=x_axis, y=y_col, color=color, marker='o', label=bap_label)
            sns.scatterplot(data=val_data, x=x_axis, y=y_col_n, color=color, marker='s', label=npp_label)

            # Add connecting lines
            plt.plot(val_data[x_axis], val_data[y_col], c=color, linestyle='-', alpha=0.7)
            plt.plot(val_data[x_axis], val_data[y_col_n], c=color, linestyle='--', alpha=0.7)

    else:
        print("Invalid option. Please choose 1 or 2.")
        return

    # Adjust x-axis from 0 to max(x_axis)*1.2
    plt.xlim(data[x_axis].min() * 0.95, data[x_axis].max() * 1.02)

    # Adjust y-axis to start from min*0.9
    y_min = min(data[y_col].min(), data[y_col_n].min())
    plt.ylim(y_min * 0.95, None)

    # Ensure chi values on the x-axis are displayed as integers if x_axis is 'chi'
    if x_axis == 'chi':
        plt.xticks(np.arange(data[x_axis].min(), data[x_axis].max() + 1, 1))

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()

    # Funktion zum Formatieren der Labels
    def format_label(label):
        if 'χ=' in label:
            parts = label.split('χ=')
            value = float(parts[1])
            return f"{parts[0]}χ={int(value)}"
        return label

    # Formatiere die Labels
    labels = [format_label(label) for label in labels]

    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: ('Trend' not in x[1], 'NPP' in x[1], x[1]))
    handles, labels = zip(*sorted_handles_labels)

    # Place the legend in the upper left corner if the combination is 'undercover' and 'epsilon'
    if metric == 'undercover' and x_axis == 'epsilon':
        plt.legend(handles=handles, labels=labels, loc='upper left')
    else:
        plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.005, 1), loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()

# Example function calls
plot_data(1, 'data/data3.csv', 'undercover', x_axis='epsilon') # Epsilon on x-axis
plot_data(2, 'data/data.csv', 'undercover', x_axis='epsilon') # Epsilon on x-axis
plot_data(1, 'data/data3.csv', 'cons', x_axis='epsilon') # Epsilon on x-axis
plot_data(2, 'data/data.csv', 'cons', x_axis='epsilon') # Epsilon on x-axis
plot_data(1, 'data/data2.csv', 'undercover', x_axis='chi') # Chi on x-axis
plot_data(2, 'data/data.csv', 'undercover', x_axis='chi') # Chi on x-axis
plot_data(1, 'data/data2.csv', 'cons', x_axis='chi') # Chi on x-axis
plot_data(2, 'data/data.csv', 'cons', x_axis='chi') # Chi on x-axis