import numpy as np
from scipy.stats import entropy


def gini_coefficient(x):
    x = np.asarray(x)
    if len(x) <= 1:
        return 0
    sorted_x = np.sort(x)
    index = np.arange(1, len(x) + 1)
    n = len(x)
    return ((2 * np.sum(index * sorted_x)) / (n * np.sum(x))) - (n + 1) / n


def run_length_encoding(seq):
    return np.diff(np.where(np.concatenate(([True], seq[1:] != seq[:-1], [True])))[0])


def analyze_shift_changes(sequence):
    if not sequence:
        return 0, 0, 0

    sequence = np.array(sequence)

    # 1. Run length encoding analysis
    runs = run_length_encoding(sequence)
    zero_runs = runs[::2] if sequence[0] == 0 else runs[1::2]

    rle_evenness = 1 - np.std(zero_runs) / np.mean(zero_runs) if len(zero_runs) > 0 else 1

    # 2. Entropy measure
    _, counts = np.unique(sequence, return_counts=True)
    entropy_value = entropy(counts, base=2)

    max_entropy = np.log2(len(sequence))
    normalized_entropy = entropy_value / max_entropy

    # 3. Gini coefficient
    ones_indices = np.where(sequence == 1)[0]

    distances = [ones_indices[0]] if len(ones_indices) > 0 else []
    distances.extend(np.diff(ones_indices).tolist())
    if len(ones_indices) > 0:
        distances.append(len(sequence) - 1 - ones_indices[-1])

    gini = gini_coefficient(distances)

    return rle_evenness, normalized_entropy, gini


# Beispielnutzung
sequence = [0, 0, 1, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 1, 1, 0, 1, 1, 0]
rle_evenness, normalized_entropy, gini = analyze_shift_changes(sequence)

print(f"Run Length Encoding Evenness: {rle_evenness:.4f}")
print(f"Normalized Entropy: {normalized_entropy:.4f}")
print(f"Gini Coefficient: {gini:.4f}")