import matplotlib.pyplot as plt
import numpy as np
import random

def plot_relative_undercover_dual(ls1, ls2, demand_dict, days, shifts):
    list1, list2 = [], []

    for day in range(1, days + 1):
        daily_sum1 = sum(ls1.get((day, shift), 0) for shift in range(1, shifts + 1))
        daily_sum2 = sum(ls2.get((day, shift), 0) for shift in range(1, shifts + 1))
        daily_demand_sum = sum(demand_dict.get((day, shift), 0) for shift in range(1, shifts + 1))

        if daily_demand_sum > 0:
            relative1 = daily_sum1 / daily_demand_sum
            relative2 = daily_sum2 / daily_demand_sum
        else:
            relative1 = relative2 = 0

        list1.append(relative1)
        list2.append(relative2)

    plt.figure(figsize=(12, 6))

    x = np.arange(1, days + 1)
    width = 0.35

    colors = plt.cm.magma([0.8, 0.2])

    bars1 = plt.bar(x - width / 2, list1, width, color=colors[0], alpha=0.7, label='Option 1')
    bars2 = plt.bar(x + width / 2, list2, width, color=colors[1], alpha=0.7, label='Option 2')

    plt.xlabel('Days', fontsize=14.5, labelpad=15)
    plt.ylabel('Relative Costs', fontsize=14.5, labelpad=15)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(x)

    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01, f'{height:.2%}',
                     ha='center', va='bottom', rotation=90, fontsize=10, color='black',
                     bbox=dict(facecolor='none', edgecolor='red', alpha= 0, pad=0.))

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Find the maximum height of all bars
    max_height = max(max(bar.get_height() for bar in bars1), max(bar.get_height() for bar in bars2))

    # Add a margin above the highest bar for the labels and legend
    legend_margin = 0.15 * max_height
    plt.ylim(top=max_height + legend_margin)

    # Erstelle die Legende mit dünnerem Rahmen
    legend = plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), ncol=1, frameon=True, edgecolor='black', facecolor='white', framealpha=1, fontsize=14.5)  # Set font size to match axis labels
    legend.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    plt.show()


def generate_random_data(days, shifts, min_value, max_value):
    return {(day, shift): round(random.uniform(min_value, max_value), 2) for day in range(1, days + 1) for shift in range(1, shifts + 1)}

demand_dict, u1, u2 = generate_random_data(28, 3, 0, 105), generate_random_data(28, 3, 0, 100), generate_random_data(28, 3, 0, 100)

plot_relative_undercover_dual(u1, u2, demand_dict, 28, 3)

