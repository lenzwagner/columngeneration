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
