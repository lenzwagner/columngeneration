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

    if option == 1:
        # Use a colorblind-friendly palette for option 1
        palette = sns.color_palette("colorblind")

        # Sort data by the x_axis
        data_bap = data.sort_values(x_axis)
        data_npp = data.sort_values(x_axis)

        # Plot lines
        plt.plot(data_bap[x_axis], data_bap[y_col], color=palette[0], label='BAP')
        plt.plot(data_npp[x_axis], data_npp[y_col_n], color=palette[1], label='NPP')

        # Add scatter plots
        sns.scatterplot(data=data, x=x_axis, y=y_col, color=palette[0], marker='o')
        sns.scatterplot(data=data, x=x_axis, y=y_col_n, color=palette[1], marker='s')

        plt.ylabel(f'{"Total Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"}')
        plt.xlabel('Epsilon $\\varepsilon$' if x_axis == 'epsilon' else '$\\chi$')
        plt.title(f'{"Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"} vs {"Epsilon" if x_axis == "epsilon" else "χ"}')

    elif option == 2:
        # Use a husl palette for option 2
        n_colors = len(data[x_axis].unique()) * 2 + 2  # *2 for BAP and NPP, +2 for overall trends
        palette = sns.color_palette("husl", n_colors=n_colors)

        # Plot scatter plots for overall trend
        sns.scatterplot(data=data, x=x_axis, y=y_col, color=palette[0], marker='o', label='Trend-BAP')
        sns.scatterplot(data=data, x=x_axis, y=y_col_n, color=palette[1], marker='s', label='Trend-NPP')

        # Add trend lines
        sns.regplot(data=data, x=x_axis, y=y_col, scatter=False, color=palette[0])
        sns.regplot(data=data, x=x_axis, y=y_col_n, scatter=False, color=palette[1])

        plt.ylabel(f'{"Total Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"}')
        plt.xlabel('Epsilon $\\varepsilon$' if x_axis == 'epsilon' else '$\\chi$')
        plt.title(f'{"Undercoverage" if metric == "undercover" else "Ø Number of Shift Changes"} vs {"Epsilon" if x_axis == "epsilon" else "χ"} for different {"$\\chi$" if x_axis == "epsilon" else "$\\varepsilon$"} values')

        # Additional points and lines for each value
        other_axis = 'chi' if x_axis == 'epsilon' else 'epsilon'
        for i, val in enumerate(sorted(data[other_axis].unique()), start=1):
            val_data = data[data[other_axis] == val].sort_values(x_axis)
            sns.scatterplot(data=val_data, x=x_axis, y=y_col, color=palette[i*2], marker='o',
                            label=f'BAP ({other_axis}={val:.2f})')
            sns.scatterplot(data=val_data, x=x_axis, y=y_col_n, color=palette[i*2+1], marker='s',
                            label=f'NPP ({other_axis}={val:.2f})')

            # Add connecting lines
            plt.plot(val_data[x_axis], val_data[y_col], c=palette[i*2], linestyle='-', alpha=0.5)
            plt.plot(val_data[x_axis], val_data[y_col_n], c=palette[i*2+1], linestyle='--', alpha=0.5)

    else:
        print("Invalid option. Please choose 1 or 2.")
        return

    # Adjust x-axis from 0 to max(x_axis)*1.2
    plt.xlim(0, data[x_axis].max() * 1.2)

    # Adjust y-axis to start from min*0.9
    y_min = min(data[y_col].min(), data[y_col_n].min())
    plt.ylim(y_min * 0.9, None)  # Set lower limit to y_min * 0.9, upper limit remains automatic

    # Legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()

# Example function calls
plot_data(1, 'data/study.csv', 'undercover', x_axis='epsilon')  # Epsilon on x-axis
plot_data(2, 'data/study.csv', 'cons', x_axis='epsilon')  # Epsilon on x-axis