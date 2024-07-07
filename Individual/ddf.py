import csv
from collections import defaultdict

# CSV-Datei einlesen
data_std = []
with open('data_std_std.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data_std.append(row)

# Standardabweichungsmatrix erstellen
std_dev_matrix = defaultdict(dict)
for row in data_std:
    model = row['model']
    epsilon = float(row['epsilon'])
    chi = int(row['chi'])
    undercover_std = float(row['undercover'])
    consistency_std = float(row['consistency'])

    std_dev_matrix[(model, epsilon, chi)] = (undercover_std, consistency_std)

# Ausgabe der Standardabweichungsmatrix
for key, std_devs in std_dev_matrix.items():
    model, epsilon, chi = key
    undercover_std, consistency_std = std_devs
    print(f"({model}, {epsilon}, {chi}): ({undercover_std:.4f}, {consistency_std:.4f})")