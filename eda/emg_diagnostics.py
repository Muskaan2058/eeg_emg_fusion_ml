"""
NeBULA Dataset - EMG Diagnostics
-------------------------------------------------------------

Runs a full EDA pass on windowed EMG data:
   1. dataset summary
   2. class distribution
   3. per-subject distribution
   4. per-class mean waveforms
   5. per-class RMS / power per channel
   6. PSD per class
   7. t-SNE
   8. Fisher scores per channel
   9. subject separability
  10. simple ML baselines
"""


import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



DATA_DIR = "../preprocessed"
OUT_DIR = "./diagnostics_emg"
FS = 200  # after resampling
RANDOM_STATE = 42

CHANNEL_NAMES = [
    "EMG1", "EMG2", "EMG3", "EMG4", "EMG5",
    "EMG6", "EMG7", "EMG8", "EMG9", "EMG10", "EMG11"
]

TASK_COLORS = {0: "#E24B4A", 1: "#378ADD", 2: "#1D9E75"}
TASK_LABELS = {0: "Task 1", 1: "Task 2", 2: "Task 3"}

os.makedirs(OUT_DIR, exist_ok=True)


def load():
    X = np.load(os.path.join(DATA_DIR, "X_emg_win.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, "y_win.npy")).astype(np.int64)
    s = np.load(os.path.join(DATA_DIR, "subject_ids_win.npy")).astype(np.int64)

    if y.min() == 1:
        y = y - 1

    print("-" * 65)
    print("EMG EDA — DATA SUMMARY")
    print("=" * 70)
    print(f"X shape      : {X.shape}")
    print(f"y shape      : {y.shape}")
    print(f"subjects     : {sorted(np.unique(s).tolist())}")
    print(f"class counts : "
          f"Task0={(y == 0).sum()}  "
          f"Task1={(y == 1).sum()}  "
          f"Task2={(y == 2).sum()}")
    print(f"global mean  : {X.mean():.6f}")
    print(f"global std   : {X.std():.6f}")
    print("-" * 65)
    return X, y, s


def save_bar_class_counts(y):
    counts = [(y == t).sum() for t in [0, 1, 2]]

    plt.figure(figsize=(6, 4))
    plt.bar(
        [TASK_LABELS[t] for t in [0, 1, 2]],
        counts,
        color=[TASK_COLORS[t] for t in [0, 1, 2]]
    )
    plt.title("Class distribution")
    plt.ylabel("Number of windows")
    plt.grid(axis="y", alpha=0.3)
    path = os.path.join(OUT_DIR, "01_class_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[1] Saved class distribution -> {path}")


def save_subject_counts(s):
    subjects = np.unique(s)
    counts = [(s == sub).sum() for sub in subjects]

    plt.figure(figsize=(12, 4))
    plt.bar([str(int(sub)) for sub in subjects], counts)
    plt.title("Windows per subject")
    plt.xlabel("Subject")
    plt.ylabel("Number of windows")
    plt.grid(axis="y", alpha=0.3)
    path = os.path.join(OUT_DIR, "02_subject_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[2] Saved subject distribution -> {path}")


def print_subject_stats(X, s):
    print("\nPer-subject mean/std:")
    for sub in np.unique(s):
        idx = s == sub
        print(
            f"  Subject {int(sub):02d}: "
            f"mean={X[idx].mean():.4f}  std={X[idx].std():.4f}  n={idx.sum()}"
        )


def save_mean_waveforms(X, y):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    t = np.arange(X.shape[2]) * 1000 / FS

    for task in [0, 1, 2]:
        ax = axes[task]
        X_task = X[y == task]
        mean_sig = X_task.mean(axis=0)  # (channels, time)

        for ch in range(mean_sig.shape[0]):
            ax.plot(t, mean_sig[ch], alpha=0.7, linewidth=1)

        ax.set_title(f"{TASK_LABELS[task]} — mean waveform across channels")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (ms)")
    path = os.path.join(OUT_DIR, "03_mean_waveforms_per_class.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[3] Saved mean waveforms -> {path}")


def save_mean_channel_heatmap(X, y):
    # mean over time, by class and channel
    mat = np.zeros((3, X.shape[1]))
    for task in [0, 1, 2]:
        mat[task] = X[y == task].mean(axis=(0, 2))

    plt.figure(figsize=(10, 3))
    plt.imshow(mat, aspect="auto", cmap="RdBu_r")
    plt.yticks([0, 1, 2], [TASK_LABELS[0], TASK_LABELS[1], TASK_LABELS[2]])
    plt.xticks(range(X.shape[1]), CHANNEL_NAMES, rotation=45)
    plt.colorbar(label="Mean amplitude")
    plt.title("Mean amplitude per class and channel")
    path = os.path.join(OUT_DIR, "04_mean_channel_heatmap.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[4] Saved mean channel heatmap -> {path}")


def save_rms_power_plots(X, y):
    rms_per_class = []
    power_per_class = []

    for task in [0, 1, 2]:
        X_task = X[y == task]
        rms = np.sqrt(np.mean(X_task ** 2, axis=(0, 2)))
        power = np.mean(X_task ** 2, axis=(0, 2))
        rms_per_class.append(rms)
        power_per_class.append(power)

    rms_per_class = np.array(rms_per_class)
    power_per_class = np.array(power_per_class)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    x = np.arange(X.shape[1])
    width = 0.25

    for i, task in enumerate([0, 1, 2]):
        axes[0].bar(x + (i - 1) * width, rms_per_class[i], width=width,
                    color=TASK_COLORS[task], label=TASK_LABELS[task])
        axes[1].bar(x + (i - 1) * width, power_per_class[i], width=width,
                    color=TASK_COLORS[task], label=TASK_LABELS[task])

    axes[0].set_title("RMS per class and channel")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CHANNEL_NAMES, rotation=45)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Power per class and channel")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CHANNEL_NAMES, rotation=45)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    path = os.path.join(OUT_DIR, "05_rms_power_per_class.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[5] Saved RMS/power plots -> {path}")

    print("\nClass-level global power:")
    for task in [0, 1, 2]:
        idx = y == task
        power = np.mean(X[idx] ** 2)
        print(f"  {TASK_LABELS[task]} power: {power:.6f}")


def save_psd_per_class(X, y, channel_idx=0):
    plt.figure(figsize=(8, 4))

    for task in [0, 1, 2]:
        X_task = X[y == task, channel_idx, :]
        psds = []
        for w in X_task[: min(len(X_task), 500)]:
            f, pxx = welch(w, fs=FS, nperseg=min(64, len(w)))
            psds.append(pxx)

        if len(psds) > 0:
            mean_psd = np.mean(psds, axis=0)
            plt.plot(f, mean_psd, color=TASK_COLORS[task], linewidth=2, label=TASK_LABELS[task])

    plt.title(f"PSD by class ({CHANNEL_NAMES[channel_idx]})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power spectral density")
    plt.grid(alpha=0.3)
    plt.legend()
    path = os.path.join(OUT_DIR, f"06_psd_class_channel_{channel_idx + 1}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[6] Saved PSD plot -> {path}")


def save_tsne_plot(X, y, s, max_samples=2500):
    n = len(X)
    if n > max_samples:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
        s_sub = s[idx]
    else:
        X_sub = X
        y_sub = y
        s_sub = s

    X_flat = X_sub.reshape(len(X_sub), -1)
    X_scaled = StandardScaler().fit_transform(X_flat)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE
    )
    Z = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    for task in [0, 1, 2]:
        idx = y_sub == task
        plt.scatter(
            Z[idx, 0], Z[idx, 1],
            s=12, alpha=0.7,
            c=TASK_COLORS[task],
            label=TASK_LABELS[task]
        )

    plt.title("t-SNE of EMG windows")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.2)
    plt.legend()
    path = os.path.join(OUT_DIR, "07_tsne_by_class.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[7] Saved t-SNE plot -> {path}")

    return Z, y_sub, s_sub


def fisher_score_per_channel(X, y):
    scores = []
    for ch in range(X.shape[1]):
        feat = X[:, ch, :].mean(axis=1)

        global_mean = feat.mean()
        numerator = 0.0
        denominator = 0.0

        for cls in np.unique(y):
            feat_c = feat[y == cls]
            mu_c = feat_c.mean()
            var_c = feat_c.var()
            n_c = len(feat_c)

            numerator += n_c * (mu_c - global_mean) ** 2
            denominator += n_c * var_c

        score = numerator / (denominator + 1e-8)
        scores.append(score)

    return np.array(scores)


def save_fisher_scores(X, y):
    scores = fisher_score_per_channel(X, y)

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(scores)), scores)
    plt.xticks(range(len(scores)), CHANNEL_NAMES, rotation=45)
    plt.ylabel("Fisher score")
    plt.title("Channel discriminability (Fisher score)")
    plt.grid(axis="y", alpha=0.3)
    path = os.path.join(OUT_DIR, "08_fisher_scores.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[8] Saved Fisher scores -> {path}")

    print("\nTop discriminative channels:")
    order = np.argsort(scores)[::-1]
    for i in order[:5]:
        print(f"  {CHANNEL_NAMES[i]}: {scores[i]:.6f}")



def main():
    X, y, s = load()

    save_bar_class_counts(y)
    save_subject_counts(s)
    print_subject_stats(X, s)

    save_mean_waveforms(X, y)
    save_mean_channel_heatmap(X, y)
    save_rms_power_plots(X, y)

    # PSD for a few channels
    for ch in [0, 3, 5]:
        if ch < X.shape[1]:
            save_psd_per_class(X, y, channel_idx=ch)

    save_tsne_plot(X, y, s, max_samples=2500)
    save_fisher_scores(X, y)

    print("\nDone. All outputs saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
