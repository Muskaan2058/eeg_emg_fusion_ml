# ==========================================
# 4.1.3 EEG vs EMG Frequency Distribution Comparison (3 Subjects)
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, sosfiltfilt, iirnotch, filtfilt, detrend

# --------------------------
# CONFIGURATION
# --------------------------
base_eeg = "/Users/muskaan_garg_/GitHub/eeg_emg_fusion_ml/Dataset- 8 channel EMG, EEG upper limb gesture data/EEG_DATA/data/csv_data"
base_emg = "/Users/muskaan_garg_/GitHub/eeg_emg_fusion_ml/Dataset- 8 channel EMG, EEG upper limb gesture data/EMG_DATA/data/csv_data"

subjects = [1, 2, 3]
gesture = 1
run = 1

fs_eeg = 250.0  # Hz (OpenBCI Ultracortex IV)
fs_emg = 200.0  # Hz (Myo Armband)

# --------------------------
# FILTERING FUNCTIONS
# --------------------------
def preprocess_eeg(x, fs):
    x = detrend(x, axis=0, type='constant')
    sos = butter(4, [0.5, 45], btype='bandpass', fs=fs, output='sos')
    x = sosfiltfilt(sos, x, axis=0)
    b, a = iirnotch(50, 30, fs=fs)
    x = filtfilt(b, a, x, axis=0)
    return x

def preprocess_emg(x, fs):
    x = np.abs(x)
    sos = butter(4, [20, 90], btype='bandpass', fs=fs, output='sos')
    x = sosfiltfilt(sos, x, axis=0)
    return x

# --------------------------
# PSD FUNCTION
# --------------------------
def mean_psd(data, fs):
    psds = []
    for ch in range(data.shape[1]):
        f, pxx = welch(data[:, ch], fs=fs, nperseg=min(len(data), int(fs * 2)))
        psds.append(pxx)
    return f, np.mean(psds, axis=0)

# --------------------------
# MAIN LOOP — 3 SUBJECTS
# --------------------------
for subject in subjects:
    eeg_path = os.path.join(base_eeg, f"subject_{subject}", f"S{subject}_R{run}_G{gesture}.csv")
    emg_path = os.path.join(base_emg, f"subject_{subject}", f"S{subject}_R{run}_G{gesture}.csv")

    if not (os.path.exists(eeg_path) and os.path.exists(emg_path)):
        print(f"[WARNING] Missing data for Subject {subject}")
        continue

    print(f"Processing Subject {subject}...")

    # Load files
    eeg = pd.read_csv(eeg_path).values
    emg = pd.read_csv(emg_path).values

    # Ensure (samples × channels)
    if eeg.shape[0] < eeg.shape[1]:
        eeg = eeg.T
    if emg.shape[0] < emg.shape[1]:
        emg = emg.T

    # Preprocess
    eeg_f = preprocess_eeg(eeg, fs_eeg)
    emg_f = preprocess_emg(emg, fs_emg)

    # Compute PSD
    freqs_eeg, psd_eeg = mean_psd(eeg_f, fs_eeg)
    freqs_emg, psd_emg = mean_psd(emg_f, fs_emg)

    # Convert to dB
    psd_eeg_db = 10 * np.log10(np.maximum(psd_eeg, 1e-12))
    psd_emg_db = 10 * np.log10(np.maximum(psd_emg, 1e-12))

    # --------------------------
    # VISUALISATION
    # --------------------------
    plt.figure(figsize=(9, 6))
    plt.plot(freqs_eeg, psd_eeg_db, label="EEG (250 Hz)", color='blue', linewidth=1.5)
    plt.plot(freqs_emg, psd_emg_db, label="EMG (200 Hz)", color='red', linewidth=1.5)
    plt.title(f"EEG vs EMG Frequency Distribution — Subject {subject}, Gesture {gesture}, Run {run}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB re V²/Hz)")
    plt.xlim(0, 100)  # EMG limited by Nyquist = 100 Hz
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
