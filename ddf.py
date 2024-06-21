import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
def plot_boxplot(file_path):
    # Lade die .xlsx Datei
    df = pd.read_excel(file_path)

    # Extrahiere die Spalte "gap"
    gap_values = df['time']

    # Erstelle eine Figur
    plt.figure(figsize=(8, 6))

    # Boxplot mit Seaborn
    sns.boxplot(y=gap_values)
    plt.title('Boxplot of Gap Values')
    plt.ylabel('Gap Values')

    # Zeige den Plot
    plt.show()

# Aufruf der Funktion mit dem Pfad zur .xlsx Datei
plot_boxplot('Individual/cg.xlsx')

import pyoverleaf
api = pyoverleaf.Api()
api.login_from_browser()


projects = api.get_projects()
project_id = projects[0].id
rootdir = api.project_get_files(project_id)
api.project_upload_file(project_id, rootdir.id, "image.jpg", open("image.jpg", "rb").read())