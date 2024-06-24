import numpy as np
import os
from matplotlib.transforms import offset_copy
import seaborn as sns
from matplotlib.ticker import PercentFormatter, MaxNLocator
import itertools
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

def performancePlot(ls, days, phys_nr, name):
    sns.set(style='darkgrid')

    file = str(name)
    file_name = f'.{os.sep}images{os.sep}{file}.png'


    grid = list(range(1, days + 1))
    graphs = [ls[i:i+days] for i in range(0, len(ls), 14)]

    fig, ax = plt.subplots()

    lw = 1.5
    palette = sns.color_palette("rocket", phys_nr)
    for gg, graph in enumerate(graphs, start=1):
        trans_offset = offset_copy(ax.transData, fig=fig, x=lw * gg, y=lw * gg, units='dots')
        ax.plot(grid, graph, lw=lw, transform=trans_offset, label=gg, color=palette[gg-1], alpha = 0.6)

    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.35), title='Physician')
    # manually set the axes limits, because the transform doesn't set them automatically
    ax.set_xlim(grid[0] - .5, grid[-1] + .5)
    ax.set_ylim(min([min(g) for g in graphs]) - .02, max([max(g) for g in graphs]) + .02)

    plt.xlabel('Day')
    plt.ylabel('Performance')
    plt.title('Physician Performance over Time')
    plt.xticks(range(1, days + 1))

    plt.savefig(file_name, format='png')

    plt.show()


def plot_obj_val(objValHistRMP, name):
    file = str(name)
    file_name = f'.{os.sep}images{os.sep}{file}.png'

    sns.set(style='darkgrid')
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

    sns.set(style='darkgrid')
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
    plt.legend()
    plt.grid(True)

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