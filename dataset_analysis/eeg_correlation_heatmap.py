# ==========================================
# EEG Channel Correlation Heatmap (Display Only for 3 Subjects)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------
# CONFIGURATION
# --------------------------
base_path = "/Users/muskaan_garg_/GitHub/eeg_emg_fusion_ml/Dataset- 8 channel EMG, EEG upper limb gesture data/EEG_DATA/data/csv_data"

subjects = [1, 2, 3]  # Subjects to process
run = 1
gesture = 3

# --------------------------
# MAIN LOOP
# --------------------------
for subject in subjects:
    file_path = os.path.join(base_path, f"subject_{subject}", f"S{subject}_R{run}_G{gesture}.csv")

    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}")
        continue

    print(f"Displaying heatmap for Subject {subject} — Gesture {gesture}, Run {run}")

    # Load data
    df = pd.read_csv(file_path)
    data = df.values

    # Ensure shape = (samples, channels)
    if data.shape[0] < data.shape[1]:
        data = data.T

    # Compute correlation matrix
    corr_matrix = np.corrcoef(data.T)

    # Plot heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap='coolwarm',
        vmin=-1, vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation (r)'}
    )
    plt.title(f"EEG Channel Correlation Heatmap\nSubject {subject}, Gesture {gesture}, Run {run}")
    plt.xlabel("Channel Index")
    plt.ylabel("Channel Index")
    plt.tight_layout()
    plt.show()
