import random
from collections import Counter

def process_list(input_list, T):
    sublists = [input_list[i:i + T] for i in range(0, len(input_list), T)]
    random.shuffle(sublists)  # Shuffle the sublists randomly
    flat_list = [item for sublist in sublists for item in sublist]
    return flat_list

# Example input list
input_list = [
    0, 0, 1, 0, 0, 2, 3, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 0, 1,
    2, 2, 3, 1, 0, 0, 1, 2, 2, 3, 0, 0, 0, 3, 0, 1, 0, 2, 0, 0,
    0, 1, 0, 0, 0, 2, 3, 3, 1, 0, 0, 0, 2, 3, 0, 0, 0, 1, 0, 0
]
T = 6

shuffled_flat_list = process_list(input_list, T)
print(len(input_list))
print(shuffled_flat_list)
