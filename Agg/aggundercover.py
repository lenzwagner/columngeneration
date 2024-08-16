from demand import *

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

    plt.savefig(f'images/{base_filename}.svg', bbox_inches='tight')
    plt.savefig(f'images/{base_filename}.png', bbox_inches='tight')
    plt.show()