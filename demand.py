import numpy as np
import matplotlib.pyplot as plt


def generate_demand(days, shifts, daily_demand):
    """
    Generates demand patterns over a specified number of days and shifts,
    adjusted to ensure the total daily demand matches the specified value.

    Parameters:
    days (int): Number of days.
    shifts (int): Number of shifts per day.
    daily_demand (int): Total demand per day.

    Returns:
    dict: A dictionary with keys as (day, shift) and values as the demand for each shift of each day.
    """
    demands = {}
    for day in range(1, days + 1):
        # Initialize demands with zeros
        shift_demands = np.zeros(shifts)
        total_demand = daily_demand

        # Randomly distribute the total demand across shifts
        for shift in range(shifts - 1):
            demand = np.random.randint(0, total_demand + 1)
            shift_demands[shift] = demand
            total_demand -= demand

        # Assign the remaining demand to the last shift
        shift_demands[shifts - 1] = total_demand

        # Randomly shuffle the demand distribution to avoid bias
        np.random.shuffle(shift_demands)

        # Store the adjusted demands in the dictionary
        for shift in range(1, shifts + 1):
            demands[(day, shift)] = shift_demands[shift - 1]

    return demands


def poisson_demand(days, shifts, lambda_daily_demand):
    demands = {}
    for day in range(1, days + 1):
        # Sample total daily demand from a Poisson distribution
        total_demand = np.random.poisson(lambda_daily_demand)

        # Initialize an array to store the demand for each shift
        shift_demands = np.zeros(shifts)

        # Distribute the total daily demand across shifts
        remaining_demand = total_demand
        for shift in range(shifts - 1):
            # Allocate a random portion of the remaining demand to the current shift
            demand = np.random.randint(0, remaining_demand + 1)
            shift_demands[shift] = demand
            remaining_demand -= demand

        # Assign the remaining demand to the last shift
        shift_demands[shifts - 1] = remaining_demand

        # Randomly shuffle the shift demands to avoid any systematic bias
        np.random.shuffle(shift_demands)

        # Store the demand values in the dictionary
        for shift in range(1, shifts + 1):
            demands[(day, shift)] = shift_demands[shift - 1]

    return demands

# Parameters
days = 19  
shifts = 3 
daily_demand = 20 

# Generate the demand pattern
demands = poisson_demand(days, shifts, daily_demand)

# Print the generated demand pattern
print("Generated Demand Pattern:")
for key, value in demands.items():
    print(f"Day {key[0]}, Shift {key[1]}: Demand {value}")

# Original line plot visualization
plt.figure(figsize=(10, 6))
for day in range(1, days + 1):
    shift_demand = [demands[(day, shift)] for shift in range(1, shifts + 1)]
    plt.plot(range(1, shifts + 1), shift_demand, marker='o', label=f'Day {day}')
plt.xlabel('Shift')
plt.ylabel('Demand')
plt.title('Demand Pattern Over Shifts')
plt.legend()
plt.grid(True)
plt.show()

# Prepare data for bar plot
demands_list = []
grays = plt.cm.Greys(np.linspace(0.3, 0.7, shifts))

for day in range(1, days + 1):
    for shift in range(1, shifts + 1):
        demands_list.append(demands[(day, shift)])

# Visualize the demand pattern using a bar plot
plt.figure(figsize=(14, 8))
bars = plt.bar(range(len(demands_list)), demands_list)

# Color bars by shift
for i, bar in enumerate(bars):
    shift_index = i % shifts
    bar.set_color(grays[shift_index])

# Add some spacing between days and adjust X-ticks
plt.xticks(ticks=[(i * shifts + (shifts - 1) / 2) for i in range(days)], labels=[f"Day {i+1}" for i in range(days)], rotation=0)

# Annotate bars with demand values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=10)

plt.xlabel('Day')
plt.ylabel('Demand')
plt.title('Demand Pattern Over Shifts')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Prepare data for bar plot
demands_list = []
colors = plt.cm.viridis(np.linspace(0, 1, days))

for day in range(1, days + 1):
    for shift in range(1, shifts + 1):
        demands_list.append(demands[(day, shift)])

# Visualize the demand pattern using a bar plot
plt.figure(figsize=(14, 8))
bars = plt.bar(range(len(demands_list)), demands_list)

# Color bars by day
for i, bar in enumerate(bars):
    day_index = i // shifts
    bar.set_color(colors[day_index])

# Add some spacing between days and adjust X-ticks
plt.xticks(ticks=[(i * shifts + (shifts - 1) / 2) for i in range(days)], labels=[f"Day {i+1}" for i in range(days)], rotation=0)

# Annotate bars with demand values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=10)

plt.xlabel('Day')
plt.ylabel('Demand')
plt.title('Demand Pattern Over Shifts')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
