import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.legend_handler import HandlerBase

# Custom legend handler for short thick lines
class ShortThickLineHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        line = plt.Line2D([width * 0.2, width * 0.8], [height / 2, height / 2], lw=4, color=orig_handle.get_color())
        return [line]

# Custom legend handler for Pareto line
class ParetoLineHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        line = plt.Line2D([width * 0.2, width * 0.8], [height / 2, height / 2], lw=2, linestyle='--', color='red')
        return [line]

def create_plot(show_pareto=True):
    # Data initialization
    data = {
        'undercoverage1': [], 'consistency1': [], 'chi1': [], 'epsilon1': [],
        'undercoverage2': [], 'consistency2': [], 'chi2': [], 'epsilon2': []
    }

    # Reading data from CSV file
    with open('data/data_new.csv', 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data['undercoverage1'].append(float(row['undercover_norm']))
            data['undercoverage2'].append(float(row['undercover_norm_n']))
            data['consistency1'].append(float(row['cons_norm']))
            data['consistency2'].append(float(row['cons_norm_n']))
            data['chi1'].append(int(float(row['chi'])))
            data['chi2'].append(int(float(row['chi'])))
            data['epsilon1'].append(float(row['epsilon']))
            data['epsilon2'].append(float(row['epsilon']))

    # Creating DataFrames
    df1 = pd.DataFrame({
        'undercoverage': data['undercoverage1'],
        'consistency': data['consistency1'],
        'chi': data['chi1'],
        'epsilon': data['epsilon1']
    })

    df2 = pd.DataFrame({
        'undercoverage': data['undercoverage2'],
        'consistency': data['consistency2'],
        'chi': data['chi2'],
        'epsilon': data['epsilon2']
    })

    df = pd.concat([df1, df2], ignore_index=True)

    # Calculate Pareto Frontier
    def pareto_frontier(df):
        pareto_front = []
        for i, row in df.iterrows():
            dominated = False
            for j, other in df.iterrows():
                if (other['undercoverage'] <= row['undercoverage'] and other['consistency'] <= row['consistency']) and (
                        other['undercoverage'] < row['undercoverage'] or other['consistency'] < row['consistency']):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(row)
        pareto_front_df = pd.DataFrame(pareto_front)
        pareto_front_df = pareto_front_df.sort_values(by=['undercoverage'])
        return pareto_front_df

    pareto_df = pareto_frontier(df)

    # Output Pareto-Frontier points to console
    if show_pareto:
        print("Punkte auf der Pareto-Frontier:")
        print(pareto_df)

    # Determine the best position for the legend
    def determine_legend_position(df):
        x_min, x_max = df['undercoverage'].min(), df['undercoverage'].max()
        y_min, y_max = df['consistency'].min(), df['consistency'].max()

        # Define quadrants
        quadrants = {
            'upper right': (x_min + (x_max - x_min) * 0.5, y_min + (y_max - y_min) * 0.5, x_max, y_max),
            'upper left': (x_min, y_min + (y_max - y_min) * 0.5, x_min + (x_max - x_min) * 0.5, y_max),
            'lower right': (x_min + (x_max - x_min) * 0.5, y_min, x_max, y_min + (y_max - y_min) * 0.5),
            'lower left': (x_min, y_min, x_min + (x_max - x_min) * 0.5, y_min + (y_max - y_min) * 0.5)
        }

        # Count points in each quadrant
        counts = {q: 0 for q in quadrants}
        for _, row in df.iterrows():
            for q, (x1, y1, x2, y2) in quadrants.items():
                if x1 <= row['undercoverage'] <= x2 and y1 <= row['consistency'] <= y2:
                    counts[q] += 1

        # Find the quadrant with the least points
        best_quadrant = min(counts, key=counts.get)

        # Map quadrant to legend position
        position_map = {
            'upper right': 'upper right',
            'upper left': 'upper left',
            'lower right': 'lower right',
            'lower left': 'lower left'
        }

        return position_map[best_quadrant]

    # Determine the best legend position
    legend_position = determine_legend_position(df)

    # Create plot
    plt.figure(figsize=(12, 8))

    # Adjusting the color range to focus on the brighter part of the magma palette
    colors = plt.cm.magma(np.linspace(0.2, 0.8, max(len(df1), len(df2))))

    # Dictionary to store the labels to avoid duplication in the legend
    labels_dict = {}

    # Points from the first list (Circles)
    for i, row in df1.iterrows():
        label = f"$\\epsilon={row['epsilon']} / \\chi={int(row['chi'])}$"
        color_index = i % len(colors)
        if label not in labels_dict:
            labels_dict[label] = plt.scatter(row['undercoverage'], row['consistency'], color=colors[color_index],
                                             marker='o', s=100)
        else:
            plt.scatter(row['undercoverage'], row['consistency'], color=colors[color_index], marker='o', s=100)

    # Points from the second list (Squares)
    for i, row in df2.iterrows():
        label = f"$\\epsilon={row['epsilon']} / \\chi={int(row['chi'])}$"
        color_index = i % len(colors)
        if label not in labels_dict:
            labels_dict[label] = plt.scatter(row['undercoverage'], row['consistency'], color=colors[color_index],
                                             marker='s', s=100, alpha=0.8)
        else:
            plt.scatter(row['undercoverage'], row['consistency'], color=colors[color_index], marker='s', s=100,
                        alpha=0.8)

    if show_pareto:
        # Highlight Pareto-optimal points
        for i, row in pareto_df.iterrows():
            index_in_df1 = df1[(df1['undercoverage'] == row['undercoverage']) &
                               (df1['consistency'] == row['consistency']) &
                               (df1['chi'] == row['chi']) &
                               (df1['epsilon'] == row['epsilon'])].index
            index_in_df2 = df2[(df2['undercoverage'] == row['undercoverage']) &
                               (df2['consistency'] == row['consistency']) &
                               (df2['chi'] == row['chi']) &
                               (df2['epsilon'] == row['epsilon'])].index

            if not index_in_df1.empty:
                color_index = index_in_df1[0] % len(colors)
                plt.scatter(row['undercoverage'], row['consistency'], color=colors[color_index], edgecolors='black',
                            linewidths=2, alpha=0.6, s=150, marker='o')
            if not index_in_df2.empty:
                color_index = index_in_df2[0] % len(colors)
                plt.scatter(row['undercoverage'], row['consistency'], color=colors[color_index], edgecolors='black',
                            linewidths=2, alpha=0.6, s=150, marker='s')

        # Connect Pareto-optimal points with dashed line
        pareto_line, = plt.plot(pareto_df['undercoverage'], pareto_df['consistency'], linestyle='--', color='red',
                                linewidth=2, alpha=0.7)

    # Create legend
    legend_elements = [Line2D([0], [0], color=handle.get_facecolor()[0], label=label, lw=4)
                       for label, handle in labels_dict.items()]

    if show_pareto:
        pareto_line_legend = Line2D([0], [0], color='red', label='Pareto-Frontier Line', lw=2, linestyle='--')
        legend_elements.append(pareto_line_legend)

    legend_position1 = 'center right'



    # Position the legend based on the determined position
    plt.legend(handles=legend_elements,
               title='Combinations:',
               loc=legend_position1,
               ncol=1,
               handler_map={Line2D: ShortThickLineHandler(), pareto_line_legend: ParetoLineHandler()} if show_pareto else {
                   Line2D: ShortThickLineHandler()},
               fontsize=8.5)

    # Increase the font size of axis labels marginally
    plt.xlabel('Undercoverage', fontsize=14)
    plt.ylabel('Consistency (ø Shift Changes)', fontsize=14)
    plt.grid(True)

    # Set axis limits
    plt.xlim(df['undercoverage'].min() - 1, df['undercoverage'].max() + 1)
    plt.ylim(df['consistency'].min() - 1, df['consistency'].max() + 1)

    plt.tight_layout()
    plt.show()

# Aufruf der Funktion
create_plot(show_pareto=False)