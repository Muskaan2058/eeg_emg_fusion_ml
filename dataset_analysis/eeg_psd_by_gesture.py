
# EEG Power Spectral Density — Mean ± SE by Gesture

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import welch, butter, sosfiltfilt, iirnotch, filtfilt, detrend

# PATHS / CONFIG ----
base = "/Users/muskaan_garg_/GitHub/eeg_emg_fusion_ml/Dataset- 8 channel EMG, EEG upper limb gesture data/EEG_DATA/data/csv_data"
subjects = [1, 2, 3]
gestures = [1, 2, 3, 4, 5, 6, 7]
runs = [1, 2, 3, 4, 5, 6]
fs = 250.0  # EEG sampling rate (OpenBCI)

# PREPROCESSING ----
def bandpass_notch(x, fs, lo=0.5, hi=45.0, notch_f=50.0, q=30.0):
    # x: (samples, channels)
    y = detrend(x, axis=0, type='constant')
    sos_bp = butter(4, [lo, hi], btype='bandpass', fs=fs, output='sos')
    y = sosfiltfilt(sos_bp, y, axis=0)
    b, a = iirnotch(notch_f, q, fs=fs)
    y = filtfilt(b, a, y, axis=0)
    return y

for s in subjects:
    plt.figure(figsize=(10, 6))
    for g in gestures:
        psd_runs = []
        for r in runs:
            fp = os.path.join(base, f"subject_{s}", f"S{s}_R{r}_G{g}.csv")
            if not os.path.exists(fp):
                continue

            X = pd.read_csv(fp).values
            if X.shape[0] < X.shape[1]:
                X = X.T  # ensure - samples, channels

            Xf = bandpass_notch(X, fs)

            # Welch per channel → mean across channels
            ch_psd = []
            # ensure segment length is valid for short trials
            nperseg = max(128, min(int(fs * 2), Xf.shape[0]))
            noverlap = int(nperseg // 2)

            for c in range(Xf.shape[1]):
                f, Pxx = welch(Xf[:, c], fs=fs, nperseg=nperseg, noverlap=noverlap)
                ch_psd.append(Pxx)
            psd_runs.append(np.mean(ch_psd, axis=0))

        if len(psd_runs) == 0:
            continue

        A = np.vstack(psd_runs)            # runs x freqs
        m = A.mean(axis=0)
        se = A.std(axis=0, ddof=1) / np.sqrt(A.shape[0])

        # dB scale
        mdB = 10 * np.log10(np.maximum(m, 1e-12))
        sedB = 10 * np.log10(np.maximum(m + se, 1e-12)) - mdB

        plt.plot(f, mdB, label=f"G{g}")
        plt.fill_between(f, mdB - sedB, mdB + sedB, alpha=0.15)

    # PLOT ----
    plt.xlim(0, 50)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB re V²/Hz)")
    plt.title(f"EEG Power Spectral Density — Subject {s} (mean ± SE across runs)")
    plt.legend(title="Gestures", ncol=3, fontsize=9)
    plt.tight_layout()
    plt.show()

