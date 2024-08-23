from demand import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "lmodern",
    "font.serif": "Computer Modern Roman",
    "font.sans-serif": "Computer Modern Sans",
    "font.monospace": "Computer Modern Typewriter",
    "axes.labelsize": 11,  # adjust as necessary
    "font.size": 11,        # adjust as necessary
    "legend.fontsize": 9,   # adjust as necessary
    "xtick.labelsize": 9,   # adjust as necessary
    "ytick.labelsize": 9,   # adjust as necessary
})

def create_dict_from_list(lst, days, shifts):
    if len(lst) != days * shifts:
        raise ValueError("Error")

    result = {}
    index = 0

    for i in range(1, days + 1):
        for j in range(1, shifts + 1):
            result[(i, j)] = lst[index]
            index += 1

    return result

import matplotlib.pyplot as plt
import numpy as np

def plot_undercover(ls, days, shifts, pt):
    lss_list = []
    colors = plt.cm.magma(np.linspace(0, 0.8, shifts))

    for day in range(1, days + 1):
        for shift in range(1, shifts + 1):
            lss_list.append(ls[(day, shift)])

    pt_in = pt / 72
    width_plt = round(pt_in)
    height_plt = round((width_plt / 16) * 9)
    plt.figure(figsize=(12,6))
    bars = plt.bar(range(len(lss_list)), lss_list)

    for i, bar in enumerate(bars):
        shift_index = i % shifts
        bar.set_color(colors[shift_index])

    plt.xticks(ticks=[(i * shifts + (shifts - 1) / 2) for i in range(days)],
               labels=[f"{i + 1}" for i in range(days)], rotation=0)

    for bar in bars:
        yval = bar.get_height()
        # Formatierung des Werts abhängig davon, ob es ein Integer ist oder nicht
        yval_str = f"{int(yval)}" if yval.is_integer() else f"{yval:.2f}"
        plt.text(bar.get_x() + bar.get_width() / 2, yval, yval_str, ha='center', va='bottom', fontsize=9)

    plt.xlabel('Day', fontsize=11)
    plt.ylabel('Undercoverage', fontsize=11)
    #plt.title('Demand Pattern', fontsize=20)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('images/undercover.svg', bbox_inches='tight')

    plt.show()

def rel_dict(a,b):
    return {key: a[key] / b[key] if key in b and b[key] != 0 else None for key in a}

def dict_reducer(data):
    result = {}

    for (key1, key2), value in data.items():
        if key1 in result:
            result[key1] += value
        else:
            result[key1] = value

    return result

def plot_undercover_d(ls, days, shifts, pt, filename_suffix=''):
    daily_undercover = []

    for day in range(1, days + 1):
        daily_sum = sum(ls.get((day, shift), 0) for shift in range(1, shifts + 1))
        daily_undercover.append(daily_sum)

    pt_in = pt / 72
    width_plt = round(pt_in)
    height_plt = round((width_plt / 16) * 9)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(1, days + 1), daily_undercover, color='blue', alpha=0.7)

    plt.xlabel('Day', fontsize=11)
    plt.ylabel('Daily Undercoverage', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(range(1, days + 1, 2))  # Show every other day on x-axis

    for bar in bars:
        yval = bar.get_height()
        yval_str = f"{int(yval)}" if yval.is_integer() else f"{yval:.2f}"
        plt.text(bar.get_x() + bar.get_width() / 2, yval, yval_str,
                 ha='center', va='bottom', fontsize=9)

    total_undercoverage = sum(daily_undercover)
    plt.text(0.95, 0.95, f'Total Undercoverage: {total_undercoverage:.2f}',
             transform=plt.gca().transAxes, ha='right', va='top',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.tight_layout()

    # Erstellen Sie den Dateinamen mit dem optionalen Suffix
    base_filename = 'daily_undercover'
    if filename_suffix:
        base_filename += f'_{filename_suffix}'

    plt.savefig(f'images/undercover/{base_filename}.svg', bbox_inches='tight')
    plt.savefig(f'images/undercover/{base_filename}.png', bbox_inches='tight')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_relative_undercover(ls1, ls2, demand_dict, days, shifts, pt, filename_suffix=''):
    daily_relative_undercover1 = []
    daily_relative_undercover2 = []

    for day in range(1, days + 1):
        # Summe des Undercoverage pro Tag für beide Listen
        daily_sum1 = sum(ls1.get((day, shift), 0) for shift in range(1, shifts + 1))
        daily_sum2 = sum(ls2.get((day, shift), 0) for shift in range(1, shifts + 1))

        # Summe des Demands pro Tag
        daily_demand_sum = sum(demand_dict.get((day, shift), 0) for shift in range(1, shifts + 1))

        # Berechnung des relativen Undercoverage für beide Listen
        if daily_demand_sum > 0:
            relative_undercover1 = daily_sum1 / daily_demand_sum
            relative_undercover2 = daily_sum2 / daily_demand_sum
        else:
            relative_undercover1 = relative_undercover2 = 0  # oder eine andere Regel, wenn der Demand 0 ist

        daily_relative_undercover1.append(relative_undercover1)
        daily_relative_undercover2.append(relative_undercover2)

    pt_in = pt / 72
    width_plt = round(pt_in)
    height_plt = round((width_plt / 16) * 9)

    plt.figure(figsize=(12, 6))

    x = np.arange(1, days + 1)
    width = 0.35

    colors = plt.cm.magma([0.2, 0.8])  # Use magma colorscheme with two distinct colors

    bars1 = plt.bar(x - width/2, daily_relative_undercover1, width, color=colors[0], alpha=0.7, label='List 1')
    bars2 = plt.bar(x + width/2, daily_relative_undercover2, width, color=colors[1], alpha=0.7, label='List 2')

    plt.xlabel('Day', fontsize=11)
    plt.ylabel('Relative Undercoverage', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(x)

    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height/2,
                     f'{height:.2%}',
                     ha='center', va='center', rotation=90, fontsize=8)

    add_value_labels(bars1)
    add_value_labels(bars2)

    avg_relative_undercoverage1 = sum(daily_relative_undercover1) / len(daily_relative_undercover1)
    avg_relative_undercoverage2 = sum(daily_relative_undercover2) / len(daily_relative_undercover2)
    plt.text(0.95, 0.95, f'Avg. Relative Undercoverage 1: {avg_relative_undercoverage1:.2%}\nAvg. Relative Undercoverage 2: {avg_relative_undercoverage2:.2%}',
             transform=plt.gca().transAxes, ha='right', va='top',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.legend()
    plt.tight_layout()

    base_filename = 'relative_undercover'
    if filename_suffix:
        base_filename += f'_{filename_suffix}'

    plt.savefig(f'images/undercover/{base_filename}.svg', bbox_inches='tight')
    plt.savefig(f'images/undercover/{base_filename}.png', bbox_inches='tight')
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_relative_undercover_dual(ls1, ls2, demand_dict, days, shifts, pt, filename_suffix=''):
    daily_relative_undercover1 = []
    daily_relative_undercover2 = []

    for day in range(1, days + 1):
        # Sum of undercoverage per day for both lists
        daily_sum1 = sum(ls1.get((day, shift), 0) for shift in range(1, shifts + 1))
        daily_sum2 = sum(ls2.get((day, shift), 0) for shift in range(1, shifts + 1))

        # Sum of demands per day
        daily_demand_sum = sum(demand_dict.get((day, shift), 0) for shift in range(1, shifts + 1))

        # Calculate relative undercoverage for both lists
        if daily_demand_sum > 0:
            relative_undercover1 = daily_sum1 / daily_demand_sum
            relative_undercover2 = daily_sum2 / daily_demand_sum
        else:
            relative_undercover1 = relative_undercover2 = 0  # Or another rule if demand is 0

        daily_relative_undercover1.append(relative_undercover1)
        daily_relative_undercover2.append(relative_undercover2)

    pt_in = pt / 72
    width_plt = round(pt_in)
    height_plt = round((width_plt / 16) * 9)

    plt.figure(figsize=(12, 6))

    x = np.arange(1, days + 1)
    width = 0.35

    colors = plt.cm.magma([0.8, 0.2])  # Use magma colorscheme with two distinct colors

    bars1 = plt.bar(x - width/2, daily_relative_undercover1, width, color=colors[0], alpha=0.7, label='Human-Scheduling Approach (HSA)')
    bars2 = plt.bar(x + width/2, daily_relative_undercover2, width, color=colors[1], alpha=0.7, label='Machine-Like Scheduling Approach (MLSA)')

    plt.xlabel('Day', fontsize=11)
    plt.ylabel('Relative Undercoverage', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(x)

    def add_value_labels(bars, color='black'):
        for bar in bars:
            height = bar.get_height()
            if height < 0.1:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{height:.2%}',
                         ha='center', va='bottom', rotation=90, fontsize=8, color='black')
            else:
                plt.text(bar.get_x() + bar.get_width()/2., height/2,
                         f'{height:.2%}',
                         ha='center', va='center', rotation=90, fontsize=8, color=color)

    # Add labels to the first bar set with white text or black for small values
    add_value_labels(bars1, color='white')
    # Add labels to the second bar set with black text
    add_value_labels(bars2)

    avg_relative_undercoverage1 = sum(daily_relative_undercover1) / len(daily_relative_undercover1)
    avg_relative_undercoverage2 = sum(daily_relative_undercover2) / len(daily_relative_undercover2)

    plt.legend()
    plt.tight_layout()

    base_filename = 'relative_undercover'
    if filename_suffix:
        base_filename += f'_{filename_suffix}'

    plt.savefig(f'images/undercover/{base_filename}.svg', bbox_inches='tight')
    #plt.savefig(f'images/undercover/{base_filename}.eps', bbox_inches='tight')
    plt.show()