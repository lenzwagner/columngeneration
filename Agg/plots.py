import numpy as np
import os
from matplotlib.transforms import offset_copy
import seaborn as sns
from matplotlib.ticker import PercentFormatter, MaxNLocator
import itertools
import plotly.express as px
import plotly.io as pio
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import gurobi_logtools as glt

def violinplots(list_cg, list_compact, name):
    file = str(name)
    file_name = f'.{os.sep}images{os.sep}{file}.png'

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    df = pd.DataFrame(list_cg, columns=['Time'])
    df1 = pd.DataFrame(list_compact, columns=['Time'])

    sns.violinplot(x=df["Time"], ax=axs[0], color=".8", bw_adjust=.5, inner_kws=dict(box_width=15, whis_width=2, color=".8"))
    sns.violinplot(x=df1["Time"], ax=axs[1], color=".8", bw_adjust=.5, inner_kws=dict(box_width=15, whis_width=2, color=".8"))

    median_cg = df["Time"].median()
    median_compact = df1["Time"].median()

    axs[0].axvline(median_cg, color='r', linestyle='--', label='Median')
    axs[0].text(median_cg, axs[0].get_ylim()[1], f'{median_cg}', ha='center', va='top', backgroundcolor='white')

    axs[1].axvline(median_compact, color='r', linestyle='--', label='Median')
    axs[1].text(median_compact, axs[1].get_ylim()[1], f'{median_compact}', ha='center', va='top', backgroundcolor='white')

    axs[0].set_title("Column Generation")
    axs[1].set_title("Compact Solver")

    plt.legend()
    plt.savefig(file_name, format='png')

    plt.show()

def optBoxplot(vals, name):
    file = str(name)
    file_name = f'.{os.sep}images{os.sep}{file}.png'

    df = pd.DataFrame(sorted(vals), columns=['Gap'])
    mean_val = np.mean(df)
    plt.axvline(x=mean_val, color='red', linestyle='--', label='Mean')
    sns.boxplot(x=df["Gap"])
    plt.title("Optimality Gap in %")
    plt.savefig(file_name, format='png')

    plt.show()

def pie_chart(optimal, name):
    file = str(name)
    file_name = f'.{os.sep}images{os.sep}{file}.png'

    zeros = sum(value == 0 for value in optimal.values())
    ones = sum(value == 1 for value in optimal.values())

    data = pd.DataFrame({'Category': ['Yes', 'No'], 'Count': [ones, zeros]})

    plt.figure(figsize=(6, 6))
    plt.pie(data['Count'], labels=data['Category'], colors=['#F18F01', '#048BA8'], startangle=90, autopct='%1.1f%%')

    plt.ylabel('')
    plt.xlabel('')
    plt.title("Optimality Distribution")
    plt.legend(labels=['Yes', 'No'], loc='lower right', bbox_to_anchor=(1.0, 0.3), title = "Optimal Solution?")
    plt.savefig(file_name, format='png')

    plt.show()

def medianplots(list_cg, list_compact, name):
    file = str(name)
    file_name = f'.{os.sep}images{os.sep}{file}.png'

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    df = pd.DataFrame(list_cg, columns=['Time'])
    df1 = pd.DataFrame(list_compact, columns=['Time'])

    sns.boxplot(x=df["Time"], ax=axs[0])
    axs[0].set_title("Column Generation")

    sns.boxplot(x=df1["Time"], ax=axs[1])
    axs[1].set_title("Compact Solver")

    median_cg = df["Time"].median()
    median_compact = df1["Time"].median()

    axs[0].axvline(median_cg, color='r', linestyle='--', label='Median')
    axs[0].text(median_cg, axs[0].get_ylim()[1], f'{median_cg}', ha='center', va='top', backgroundcolor='white')

    axs[1].axvline(median_compact, color='r', linestyle='--', label='Median')
    axs[1].text(median_compact, axs[1].get_ylim()[1], f'{median_compact}', ha='center', va='top', backgroundcolor='white')
    plt.legend()

    plt.savefig(file_name, format='png')
    plt.show()

def performancePlot(ls, days, name, anzahl_ls, eps, chi):


    # Data validation and preprocessing
    ls = np.clip(ls, 0, 1)  # Clip values between 0 and 1

    sublists = [ls[i:i + days * anzahl_ls] for i in range(0, len(ls), days * anzahl_ls)]

    # Calculate average performance for each day
    all_workers_data = np.array(sublists).reshape(-1, anzahl_ls, days)
    avg_performance = np.mean(all_workers_data, axis=1)

    grid = list(range(1, days + 1))

    # Define color palettes
    worker_palette = plt.cm.magma(np.linspace(0.15, 0.85, anzahl_ls))
    mean_palette = plt.cm.magma(np.linspace(0.5, 0.5, 1))

    for idx, sublist in enumerate(sublists):
        start_physician = idx * anzahl_ls + 1
        end_physician = start_physician + anzahl_ls - 1
        file = f"{name}_Physician_{start_physician}_to_{end_physician}_{eps}_{chi}"
        file_name = f".{os.sep}images{os.sep}perfplots{os.sep}{file}.png"

        graphs = [sublist[i:i + days] for i in range(0, len(sublist), days)]

        fig, ax = plt.subplots(figsize=(12, 8))

        lw = 1.5

        # Plot individual worker performances
        for gg, graph in enumerate(graphs):
            try:
                worker_index = gg % anzahl_ls
                trans_offset = offset_copy(ax.transData, fig=fig, x=lw * worker_index, y=lw * worker_index,
                                           units='dots')
                ax.plot(grid, graph, lw=lw, transform=trans_offset, label=gg + start_physician,
                        color=worker_palette[worker_index], alpha=0.8)
                print(f"Plotting data for Worker {gg + start_physician}: {graph}")  # Debugging output
            except Exception as e:
                print(f"Error plotting Worker {gg + start_physician}: {e}")

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Worker')
        ax.set_xlim(grid[0] - .5, grid[-1] + .5)
        ax.set_ylim(min(sublist) - 0.05, 1.05)
        ax.set_xlabel('Day')
        ax.set_ylabel('Performance')
        ax.set_title(f'Individual Performance of Workers {start_physician} to {end_physician}')
        ax.set_xticks(range(1, days + 1))

        plt.tight_layout()
        plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory

    print("All group plots generated successfully.")

    # Create a separate plot for overall average performance
    fig, ax = plt.subplots(figsize=(12, 8))
    overall_avg = np.mean(avg_performance, axis=0)
    overall_std = np.std(avg_performance, axis=0)
    ax.plot(grid, overall_avg, lw=2, color=mean_palette[0], label='Average Performance')

    # Correct the standard deviation boundaries for overall plot
    overall_lower_bound = np.maximum(overall_avg - overall_std, 0)
    overall_upper_bound = np.minimum(overall_avg + overall_std, 1)

    ax.fill_between(grid, overall_lower_bound, overall_upper_bound, alpha=0.2, color=mean_palette[0])
    ax.set_xlim(grid[0] - .5, grid[-1] + .5)
    ax.set_ylim(min(overall_lower_bound) - 0.05, 1.05)
    ax.set_xlabel('Day')
    ax.set_ylabel('Average Performance')
    ax.set_title('Overall Average Performance Across All Workers')
    ax.set_xticks(range(1, days + 1))
    ax.legend()

    plt.tight_layout()
    overall_avg_file = f".{os.sep}images{os.sep}perfplots{os.sep}{name}_Overall_Average__{eps}_{chi}.png"
    plt.savefig(overall_avg_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print("Overall average plot generated successfully.")

def plot_obj_val(objValHistRMP, name):
    file = str(name)
    file_name = f'.{os.sep}images{os.sep}{file}.png'


    sns.scatterplot(x=list(range(len(objValHistRMP[:-1]))), y=objValHistRMP[:-1], marker='o', color='#3c4cad',
                    label='Objective Value')
    sns.lineplot(x=list(range(len(objValHistRMP))), y=objValHistRMP, color='#3c4cad')
    sns.scatterplot(x=[len(objValHistRMP) - 1], y=[objValHistRMP[-1]], color='#f9c449', s=100, label='Last Point')

    plt.xlabel('Iterations')
    plt.xticks(range(0, len(objValHistRMP)))
    plt.ylabel('Objective function value')
    title = 'Optimal integer objective value: ' + str(round(objValHistRMP[-1], 2))
    plt.title(title)

    x_ticks_labels = list(range(len(objValHistRMP) - 1)) + ["Int. Solve"]
    plt.xticks(ticks=list(range(len(objValHistRMP))), labels=x_ticks_labels)

    h, l = plt.gca().get_legend_handles_labels()
    plt.legend(h[:2], l[:2] + ['Last Point'], loc='best', handletextpad=0.1, handlelength=1, fontsize='medium',
               title='Legend')

    plt.savefig(file_name, format='png')
    plt.show()

def plot_avg_rc(avg_rc_hist, name):
    file_dir = 'images'
    file_name = str(name) + '.png'
    plot_path = os.path.join(file_dir, file_name)


    sns.scatterplot(x=list(range(1, len(avg_rc_hist) + 1)), y=avg_rc_hist, marker='o', color='#3c4cad')
    sns.lineplot(x=list(range(1, len(avg_rc_hist) + 1)), y=avg_rc_hist, color='#3c4cad')
    plt.xlabel('Iterations')
    plt.xticks(range(1, len(avg_rc_hist)+1))
    plt.ylabel('Reduced Cost')
    title = 'Final reduced cost: ' + str(round(avg_rc_hist[-1], 2))
    plt.title(title)

    plt.savefig(plot_path, format='png')
    plt.show()

def optimality_plot(default_run):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=default_run["Time"], y=default_run["Incumbent"], name="Primal Bound"))
    fig.add_trace(go.Scatter(x=default_run["Time"], y=default_run["BestBd"], name="Dual Bound"))
    fig.add_trace(go.Scatter(x=default_run["Time"], y=default_run["Gap"], name="Gap"))
    fig.update_xaxes(title="Runtime")
    fig.update_yaxes(title="Obj Val")
    fig.show()

def combine_legends(*axes):
    handles = list(itertools.chain(*[ax.get_legend_handles_labels()[0] for ax in axes]))
    labels = list(
        itertools.chain(*[ax3.get_legend_handles_labels()[1] for ax3 in axes])
    )
    return handles, labels


def set_obj_axes_labels(ax):
    ax.set_ylabel("Objective value")
    ax.set_xlabel("Iterations")


def plot_obj(df, ax):
    ax.step(
        list(range(len(df))),
        df,
        where="post",
        color="b",
        label="Obj",
    )
    set_obj_axes_labels(ax)

def plot_gap(df1, ax):
    ax.step(
        list(range(len(df1))),
        df1,
        where="post",
        color="green",
        label="Gap",
    )
    ax.set_ylabel("Optimality Gap in %")
    ax.set_ylim(0, 1)
    formatter = PercentFormatter(1)
    ax.yaxis.set_major_formatter(formatter)


def optimalityplot(df, df2, last_itr, name):
    file = str(name)
    file_name = f'.{os.sep}images{os.sep}{file}.png'

    with plt.style.context("seaborn-v0_8"):
        _, ax = plt.subplots(figsize=(8, 5))

        plot_obj(df, ax)

        ax2 = ax.twinx()
        plot_gap(df2, ax2)

        ax.set_xlim(0, last_itr-2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        print(combine_legends(ax, ax2))
        ax.legend(*combine_legends(ax, ax2))

        plt.savefig(file_name, format='png')

        plt.show()

import matplotlib.pyplot as plt

def lagrangeprimal(sum_rc_hist, objValHistRMP):
    result = [x + y for x, y in zip(sum_rc_hist, objValHistRMP[:-1])]

    objValHistRMP_without_last = objValHistRMP[:-1]

    iterations = range(len(sum_rc_hist))

    iterations_objValHistRMP = range(len(objValHistRMP_without_last))

    plt.figure(figsize=(12, 6))

    plt.plot(iterations, result, label='LagrangeBound', linestyle='-', color='orange')
    plt.plot(iterations_objValHistRMP, objValHistRMP_without_last, label='PrimalMasterObj', linestyle='-', color='green')

    plt.xlabel('Iteration')
    plt.ylabel('Objective values')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.show()


def runtime_heatmap(liste1, liste2 , run):
    run_in_minutes = [round(x / 60) for x in run]

    data = pd.DataFrame({'liste1': liste1, 'liste2': liste2, 'run': run_in_minutes})

    heatmap_data = data.pivot(index='liste1', columns='liste2', values='run')

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar=True, fmt="d")
    plt.title('Model runtimes')
    plt.xlabel('Instances')
    plt.ylabel(r'$\varepsilon / \chi$-Combinations')
    plt.show()

def gap_heatmap(liste1, liste2 , gap):
    data = pd.DataFrame({'liste1': liste1, 'liste2': liste2, 'run': gap})

    heatmap_data = data.pivot(index='liste1', columns='liste2', values='gap')

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar=True, fmt="d")
    plt.title('Optimality gap in Percent')
    plt.xlabel('Instances')
    plt.ylabel(r'$\varepsilon / \chi$-Combinations')
    plt.show()


def generate_and_plot_boxplots(instances, profiles, overstaffing):
    # Data must look like this
    # instances = ['Instance1', 'Instance2', 'Instance3', 'Instance4']
    # profiles = ['Profile1', 'Profile2', 'Profile3', 'Profile4']
    # overstaffing = [[[10, 12, 14], [20, 22, 74], [30, 32, 34], [40, 42, 44]],  [[15, 17, 19], [25, 27, 2], [35, 37, 39], [45, 47, 49]], [[20, 22, 24], [30, 32, 34], [40, 42, 44], [50, 52, 54]],  [[25, 27, 29], [35, 37, 39], [45, 47, 49], [55, 57, 59]] ]

    # Ensure overstaffing is a 2D array with appropriate shape
    assert len(overstaffing) == len(instances), "Length of overstaffing must match length of instances"
    assert all(
        len(o) == len(profiles) for o in overstaffing), "Each sublist in overstaffing must match length of profiles"

    # Define the data structure
    data = {
        'instance': [],
        'profile': [],
        'runtime': []
    }

    for instance_idx, instance in enumerate(instances):
        for profile_idx, profile in enumerate(profiles):
            runtimes = overstaffing[instance_idx][profile_idx]
            data['instance'].extend([instance] * len(runtimes))
            data['profile'].extend([profile] * len(runtimes))
            data['runtime'].extend(runtimes)

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Define a grayscale palette
    palette = sns.color_palette("Greys", len(profiles))

    # Create the plot
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='instance', y='runtime', hue='profile', data=df, palette=palette)
    plt.xlabel('Instance')
    plt.ylabel('Runtime')
    plt.legend(title='Profile')
    plt.title('Runtime by Instance and Profile')
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.transforms import offset_copy
import xml.etree.ElementTree as ET


def read_xml_data(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    data_lists = []
    for iteration in root.findall('Iteration'):
        values = iteration.findall('Value')
        data_list = [float(value.text) for value in values]
        data_lists.append(data_list)

    return data_lists


def performancePlotxml(xml_file_path, days, name, anzahl_ls, liste1_name="BAP", liste2_name="NPP"):


    data_lists = read_xml_data(xml_file_path)

    print(data_lists)

    overall_min, overall_max = 0.85, 1.05  # Set fixed y-axis limits
    grid = list(range(1, days + 1))

    # Define color palettes
    palette1 = plt.cm.magma(np.linspace(0.15, 0.5, 1))
    palette2 = plt.cm.magma(np.linspace(0.65, 0.85, 1))

    for iteration, ls in enumerate(data_lists, 1):
        ls = np.clip(ls, 0, 1)  # Clip values between 0 and 1
        sublists = [ls[i:i + days * anzahl_ls] for i in range(0, len(ls), days * anzahl_ls)]
        all_workers_data = np.array(sublists).reshape(-1, anzahl_ls, days)
        avg_performance = np.mean(all_workers_data, axis=1)

        fig, ax = plt.subplots(figsize=(12, 8))

        for idx, sublist in enumerate(sublists):
            start_physician = idx * anzahl_ls + 1
            end_physician = start_physician + anzahl_ls - 1

            graphs = [sublist[i:i + days] for i in range(0, len(sublist), days)]

            # Plot list 1 ("BAP")
            group_avg1 = np.mean(graphs, axis=0)
            group_std1 = np.std(graphs, axis=0)
            ax.plot(grid, group_avg1, lw=2, color=palette1[0], label=liste1_name)

            lower_bound1 = np.maximum(group_avg1 - group_std1, 0)
            upper_bound1 = np.minimum(group_avg1 + group_std1, 1)

            ax.fill_between(grid, lower_bound1, upper_bound1, alpha=0.2, color=palette1[0])

            # Plot list 2 ("NPP")
            group_avg2 = np.mean(graphs, axis=0)
            group_std2 = np.std(graphs, axis=0)
            ax.plot(grid, group_avg2, lw=2, color=palette2[0], label=liste2_name)

            lower_bound2 = np.maximum(group_avg2 - group_std2, 0)
            upper_bound2 = np.minimum(group_avg2 + group_std2, 1)

            ax.fill_between(grid, lower_bound2, upper_bound2, alpha=0.2, color=palette2[0])

        ax.set_xlim(grid[0] - .5, grid[-1] + .5)
        ax.set_ylim(overall_min, overall_max)
        ax.set_xlabel('Day')
        ax.set_ylabel('Average Performance')
        ax.set_title(f'Comparison of {liste1_name} and {liste2_name} Performance (Iteration {iteration})')
        ax.set_xticks(range(1, days + 1))
        ax.legend()

        plt.tight_layout()
        file_name = f".{os.sep}images{os.sep}perfplots{os.sep}{name}_Iteration{iteration}_Comparison.svg"
        try:
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving file {file_name}: {e}")

        print(f"Comparison plot for iteration {iteration} generated successfully.")

    print("All comparison plots for all iterations generated successfully.")


def performancePlotAvg(ls1, ls2, days, name, anzahl_ls, eps, chi):
    # Data validation and preprocessing
    ls1 = np.clip(ls1, 0, 1)  # Clip values between 0 and 1
    ls2 = np.clip(ls2, 0, 1)  # Clip values between 0 und 1

    sublists1 = [ls1[i:i + days * anzahl_ls] for i in range(0, len(ls1), days * anzahl_ls)]
    sublists2 = [ls2[i:i + days * anzahl_ls] for i in range(0, len(ls2), days * anzahl_ls)]

    # Calculate average performance for each list
    all_workers_data1 = np.array(sublists1).reshape(-1, anzahl_ls, days)
    avg_performance1 = np.mean(all_workers_data1, axis=1)

    all_workers_data2 = np.array(sublists2).reshape(-1, anzahl_ls, days)
    avg_performance2 = np.mean(all_workers_data2, axis=1)

    grid = list(range(1, days + 1))

    # Define color palettes
    palette1 = plt.cm.magma(np.linspace(0.15, 0.5, 1))
    palette2 = plt.cm.magma(np.linspace(0.65, 0.85, 1))

    # Create a separate plot for overall average performance
    fig, ax = plt.subplots(figsize=(11, 5))

    # Plot average and std for list 1 (BAP)
    overall_avg1 = np.mean(avg_performance1, axis=0)
    overall_std1 = np.std(avg_performance1, axis=0)
    ax.plot(grid, overall_avg1, lw=2, color=palette1[0], label='Human-Scheduling Approach (HSA)')

    overall_lower_bound1 = np.maximum(overall_avg1 - overall_std1, 0)
    overall_upper_bound1 = np.minimum(overall_avg1 + overall_std1, 1)

    ax.fill_between(grid, overall_lower_bound1, overall_upper_bound1, alpha=0.2, color=palette1[0])

    # Plot average and std for list 2 (NPP)
    overall_avg2 = np.mean(avg_performance2, axis=0)
    overall_std2 = np.std(avg_performance2, axis=0)
    ax.plot(grid, overall_avg2, lw=2, color=palette2[0], label='Machine-Scheduling Approach (MSA)')

    overall_lower_bound2 = np.maximum(overall_avg2 - overall_std2, 0)
    overall_upper_bound2 = np.minimum(overall_avg2 + overall_std2, 1)

    ax.fill_between(grid, overall_lower_bound2, overall_upper_bound2, alpha=0.2, color=palette2[0])

    # Configure plot details
    ax.set_xlim(grid[0] - .5, grid[-1] + .5)
    ax.set_ylim(min(min(overall_lower_bound1), min(overall_lower_bound2)) - eps, 1 + eps)
    ax.set_xlabel('Day')
    ax.set_ylabel(r'Average Performance $\bar{p}_{id}$')
    ax.set_xticks(range(1, days + 1))
    ax.legend()

    print(f"Avg: {overall_avg1[-1], overall_avg2[-1]}")
    plt.tight_layout()
    overall_avg_file = f".{os.sep}images{os.sep}perfplots{os.sep}{name}_Overall_Average_Comparison__{eps}_{chi}.svg"
    plt.savefig(overall_avg_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print("Overall average comparison plot generated successfully.")


def visualize_schedule(dic, days, undercoverage, I, T, K):
    result = {}
    index = 0
    for i in range(1, I + 1):
        for t in range(1, T + 1):
            for k in range(1, K + 1):
                if index < len(dic):
                    result[(i, t, k)] = dic[index]
                    index += 1
                else:
                    break

    print(f"Res: {result}")

    s = pd.Series(result)

    data = (s.loc[lambda s: s == 1]
           .reset_index(-1)['level_2'].unstack(fill_value=0)
           .reindex(index=s.index.get_level_values(0).unique(),
                    columns=s.index.get_level_values(1).unique(),
                    fill_value=0
                    )
           )

    data.index = data.index.astype(int)
    data.columns = data.columns.astype(str)

    title_str = f'Physician Schedules | Total Undercoverage: {undercoverage}'
    fig = px.imshow(data[[str(i) for i in range(1, days + 1)]],
                    color_continuous_scale=[ '#E57373' , '#4B8B9F', '#DAA520' ,'#76B041'])

    fig.update(data=[{'hovertemplate': "Day: %{x}<br>"
                                       "Physician: %{y}<br>"}])

    colors = dict(thickness=35,
                    tickvals=[0, 1, 2, 3],
                    ticktext=['Off', 'Morning', 'Noon', 'Evening'],
                    title = "Shift")

    fig.update(layout_coloraxis_showscale=True, layout_coloraxis_colorbar=colors)


    x_ticks = np.arange(1, days + 1)
    day_labels = ['Day ' + str(i) for i in x_ticks]
    fig.update_xaxes(tickvals=x_ticks, ticktext=day_labels)

    y_ticks = np.arange(1, data.shape[0] + 1)
    physician_labels = ['Physician ' + str(i) for i in y_ticks]
    fig.update_yaxes(tickvals=y_ticks, ticktext=physician_labels)

    fig.update_layout(
        title={
            'text': title_str,
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        }
    )

    fig.show()
    return fig