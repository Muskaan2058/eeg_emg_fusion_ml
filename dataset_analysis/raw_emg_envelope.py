# ==========================================
# Raw EMG + Envelope Example (Myo Armband – 200 Hz)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, sosfiltfilt

# --------------------------
# CONFIGURATION
# --------------------------
base_path = "/Users/muskaan_garg_/GitHub/eeg_emg_fusion_ml/Dataset- 8 channel EMG, EEG upper limb gesture data/EMG_DATA/data/csv_data"
subjects = [1, 2, 3]   # three subjects
gesture = 1            # one gesture for comparison
run = 1                # one trial per subject
fs = 200               # Myo sampling frequency (Hz)
window_ms = 100        # 100 ms smoothing
window = int(fs * (window_ms / 1000))  # = 20 samples

# --------------------------
# OPTIONAL: light band-pass 20–90 Hz
# --------------------------
def bandpass_filter(x, fs, low=20, high=90, order=4):
    sos = butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, x, axis=0)

# --------------------------
# MAIN LOOP
# --------------------------
for subject in subjects:
    file_path = os.path.join(base_path, f"subject_{subject}", f"S{subject}_R{run}_G{gesture}.csv")
    print(f"Processing: {file_path}")

    df = pd.read_csv(file_path)
    data = df.values
    if data.shape[0] < data.shape[1]:
        data = data.T

    # Use one representative channel (e.g., first channel)
    ch_data = data[:, 0]

    # filter for cleaner visual
    ch_data = bandpass_filter(ch_data, fs)

    # Rectify signal
    rectified = np.abs(ch_data)

    # Smooth envelope with moving average (100 ms)
    envelope = np.convolve(rectified, np.ones(window)/window, mode='same')

    # --------------------------
    # PLOT
    # --------------------------
    t = np.arange(len(ch_data)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t, ch_data, color='gray', alpha=0.6, label='Raw EMG (filtered 20–90 Hz)')
    plt.plot(t, envelope, color='red', linewidth=1.5, label='Smoothed Envelope (100 ms)')
    plt.title(f"Raw EMG + Envelope — Subject {subject}, Gesture {gesture}, Run {run}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()
    plt.tight_layout()
    plt.show()
