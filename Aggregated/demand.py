import random
import matplotlib.pyplot as plt
import numpy as np

def demand_dict_fifty2(num_days, prob, demand, middle_shift, fluctuation=0.25):
    base_total_demand = int(prob * demand)
    demand_dict = {}

    for day in range(1, num_days + 1):
        fluctuation_factor = 1 + (random.uniform(-fluctuation, fluctuation))
        total_demand = int(base_total_demand * fluctuation_factor)

        middle_shift_ratio = random.random()
        middle_shift_demand = round(total_demand * middle_shift_ratio)
        remaining_demand = total_demand - middle_shift_demand

        early_late_ratio = random.random()
        early_demand = round(remaining_demand * early_late_ratio)
        late_demand = remaining_demand - early_demand

        if middle_shift == 1:
            demand_dict[(day, 1)] = middle_shift_demand
            demand_dict[(day, 2)] = early_demand
            demand_dict[(day, 3)] = late_demand
        elif middle_shift == 2:
            demand_dict[(day, 1)] = early_demand
            demand_dict[(day, 2)] = middle_shift_demand
            demand_dict[(day, 3)] = late_demand
        elif middle_shift == 3:
            demand_dict[(day, 1)] = early_demand
            demand_dict[(day, 2)] = late_demand
            demand_dict[(day, 3)] = middle_shift_demand
        else:
            raise ValueError("Invalid middle_shift value")

    return demand_dict


def demand_dict_third(num_days, prob, demand):
    total_demand = int(prob * demand)
    demand_dict = {}

    for day in range(1, num_days + 1):
        z1 = random.random()
        z2 = random.random()
        z3 = random.random()

        summe = z1 + z2 + z3

        demand1 = (z1 / summe) * total_demand
        demand2 = (z2 / summe) * total_demand
        demand3 = (z3 / summe) * total_demand

        demand1_rounded = round(demand1)
        demand2_rounded = round(demand2)
        demand3_rounded = round(demand3)

        rounded_total = demand1_rounded + demand2_rounded + demand3_rounded
        rounding_difference = total_demand - rounded_total

        if rounding_difference != 0:
            shift_indices = [1, 2, 3]
            random.shuffle(shift_indices)
            for i in range(abs(rounding_difference)):
                if rounding_difference > 0:
                    if shift_indices[i] == 1:
                        demand1_rounded += 1
                    elif shift_indices[i] == 2:
                        demand2_rounded += 1
                    else:
                        demand3_rounded += 1
                else:
                    if shift_indices[i] == 1:
                        demand1_rounded -= 1
                    elif shift_indices[i] == 2:
                        demand2_rounded -= 1
                    else:
                        demand3_rounded -= 1

        demand_dict[(day, 1)] = demand1_rounded
        demand_dict[(day, 2)] = demand2_rounded
        demand_dict[(day, 3)] = demand3_rounded

    return demand_dict

def plot_demand_pattern(demands, days, shifts):
    shift_labels = ["Morning", "Noon", "Evening"]
    """
    Plots the demand pattern over shifts for a given number of days and shifts.

    Parameters:
    - demands: dict, demand values with keys as (day, shift) tuples.
    - days: int, number of days.
    - shifts: int, number of shifts per day.
    - shift_labels: list of str, labels for each shift.
    """
    plt.figure(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, days))

    for day in range(1, days + 1):
        shift_demand = [demands[(day, shift)] for shift in range(1, shifts + 1)]
        plt.plot(range(1, shifts + 1), shift_demand, marker='o', label=f'Day {day}', color=colors[day - 1])

    plt.xlabel('Shift')
    plt.ylabel('Demand')
    plt.title('Demand Pattern Over Shifts')
    plt.xticks(range(1, shifts + 1), shift_labels)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_demand_bar(demands, days, shifts):
    """
    Plots the demand pattern over shifts using a bar plot for a given number of days and shifts.

    Parameters:
    - demands: dict, demand values with keys as (day, shift) tuples.
    - days: int, number of days.
    - shifts: int, number of shifts per day.
    """
    demands_list = []
    grays = plt.cm.Greys(np.linspace(0.3, 0.7, shifts))

    for day in range(1, days + 1):
        for shift in range(1, shifts + 1):
            demands_list.append(demands[(day, shift)])

    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(demands_list)), demands_list)

    for i, bar in enumerate(bars):
        shift_index = i % shifts
        bar.set_color(grays[shift_index])

    plt.xticks(ticks=[(i * shifts + (shifts - 1) / 2) for i in range(days)],
               labels=[f"Day {i + 1}" for i in range(days)], rotation=0)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Day')
    plt.ylabel('Demand')
    plt.title('Demand Pattern Over Shifts')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def plot_demand_bar_by_day(demands, days, shifts):
    """
    Plots the demand pattern over shifts using a bar plot for a given number of days and shifts.

    Parameters:
    - demands: dict, demand values with keys as (day, shift) tuples.
    - days: int, number of days.
    - shifts: int, number of shifts per day.
    """
    demands_list = []
    colors = plt.cm.viridis(np.linspace(0, 1, days))

    for day in range(1, days + 1):
        for shift in range(1, shifts + 1):
            demands_list.append(demands[(day, shift)])

    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(demands_list)), demands_list)

    for i, bar in enumerate(bars):
        day_index = i // shifts
        bar.set_color(colors[day_index])

    plt.xticks(ticks=[(i * shifts + (shifts - 1) / 2) for i in range(days)],
               labels=[f"Day {i + 1}" for i in range(days)], rotation=0)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Day')
    plt.ylabel('Demand')
    plt.title('Demand Pattern Over Shifts')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()