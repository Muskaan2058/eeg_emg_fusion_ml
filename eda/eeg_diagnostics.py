"""
NeBULA Dataset - EEG Diagnostics
-------------------------------------------------------------
Runs a full EDA pass on windowed EEG data:

Two sections:
  A. Signal quality plots (raw data — needs .vhdr files + MNE)
     A1. EEG channel correlation heatmap
     A2. EEG power spectral density with frequency band shading

  B. EEG discriminability analysis (preprocessed windowed data)
     B1. ERP overlap — mean waveforms per class at C3
     B2. PSD per class — do tasks differ in frequency content?
     B3. t-SNE — is there any class structure in EEG space?
     B4. Fisher score per channel — which channels are most discriminative?
     B5. Per-subject separability — is EEG consistent across subjects?
     B6. Window discriminability — which time window is most useful?
     B7. Channel x window heatmap — where and when does signal exist?
     B8. Phase summary — pre / onset / execution / late comparison
     B9. All-channel Fisher — do any of the 128 channels help?

"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

import pandas as pd
from scipy import signal
from scipy.stats import zscore

from sklearn.manifold import TSNE
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

try:
    import mne
    mne.set_log_level("ERROR")
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


#  Config
# -------------------------------------------------

RAW_DATA_DIR  = "../data"  # raw BIDS data (sub-XX folders)
PREP_DATA_DIR = "../preprocessed"   # windowed arrays from epoch.py
OUT_DIR       = "./diagnostics_eeg"
os.makedirs(OUT_DIR, exist_ok=True)

FS_RAW    = 1000   # original sampling rate
FS        = 200    # target after resampling

# 15 motor channels used in modelling
MOTOR_CHANNELS = [
    "C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4",
    "C1", "C2", "C5",  "C6", "FC1", "FC2", "CP1", "CP2"
]
MOTOR_CH_SET = set(MOTOR_CHANNELS)

TASK_COLORS = {0: "#E24B4A", 1: "#378ADD", 2: "#1D9E75"}
TASK_LABELS = {0: "Task 1",  1: "Task 2",  2: "Task 3"}

# Window positions from epoch.py (sample index, onset at sample 100)
WINDOW_STARTS = np.array([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400])
WINDOW_MS     = (WINDOW_STARTS - 100) * (1000 / FS)   # ms relative to onset

CONDITION       = "free"
N_SUBJECTS_RAW  = 5     # subjects to use for section A and B9
RANDOM_STATE    = 42



#  Argument parsing
# -------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="NeBULA EEG Diagnostics")
    p.add_argument("--section",      default="both", choices=["A", "B", "both"],
                   help="Which section to run (default: both)")
    p.add_argument("--subjects",     nargs="+", default=None,
                   help="Subject IDs for section A e.g. --subjects 01 02 04")
    p.add_argument("--all-subjects", action="store_true",
                   help="Use all subjects for section A")
    p.add_argument("--condition",    default=CONDITION, choices=["free", "low", "high"])
    return p.parse_args()


def get_raw_subjects(requested=None, all_subs=False):
    found = sorted([
        d.replace("sub-", "") for d in os.listdir(RAW_DATA_DIR)
        if d.startswith("sub-") and os.path.isdir(os.path.join(RAW_DATA_DIR, d))
    ])
    # Exclude documented bad subjects (Garro et al. 2025)
    found = [s for s in found if s not in ["03", "12", "21"]]

    if requested:
        return [s for s in requested if s in found]
    if all_subs:
        return found
    return found[:N_SUBJECTS_RAW]



#  Signal processing helpers
# -------------------------------------------------

def bandpass(data, lo, hi, fs, order=4):
    nyq = fs / 2.0
    b, a = signal.butter(order, [lo/nyq, hi/nyq], btype="band")
    return signal.filtfilt(b, a, data, axis=-1)

def notch_filter(data, freq, fs, Q=30):
    b, a = signal.iirnotch(freq / (fs / 2.0), Q)
    return signal.filtfilt(b, a, data, axis=-1)

def resample(data, fs_orig, fs_target):
    n = int(data.shape[-1] * fs_target / fs_orig)
    return signal.resample(data, n, axis=-1)

def psd_welch(data, fs):
    psds = []
    for ch in range(data.shape[0]):
        f, p = signal.welch(data[ch], fs=fs, nperseg=min(512, data.shape[-1]//4))
        psds.append(p)
    return f, np.array(psds)



#  Raw data loaders
# -------------------------------------------------

def load_eeg_raw(sid, condition):
    """Load, preprocess, and return EEG for one subject."""
    if not MNE_AVAILABLE:
        return None, None
    vhdr = os.path.join(RAW_DATA_DIR, f"sub-{sid}", "eeg",
                        f"sub-{sid}_task-{condition}_eeg.vhdr")
    if not os.path.exists(vhdr):
        return None, None

    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
    available = [c for c in MOTOR_CHANNELS if c in raw.ch_names]
    if len(available) < 3:
        available = raw.ch_names[:15]
    raw.pick_channels(available)

    data = raw.get_data()
    data = signal.detrend(data, axis=-1)
    data = bandpass(data, 1, 45, FS_RAW)
    data = notch_filter(data, 50, FS_RAW)
    data -= data.mean(axis=0, keepdims=True)   # CAR
    data = resample(data, FS_RAW, FS)
    data = zscore(data, axis=-1)
    return data, available


def load_events_raw(sid, condition):
    path = os.path.join(RAW_DATA_DIR, f"sub-{sid}", "emg",
                        f"sub-{sid}_task-{condition}_events.tsv")
    ev = pd.read_csv(path, sep="\t")
    g  = ev[ev["value"].str.startswith("G", na=False)].copy()
    g["label"] = g["type"].astype(int) - 1
    return g.reset_index(drop=True)


def load_all_eeg_channels(sid, condition):
    """Load raw EEG keeping all 128 channels."""
    if not MNE_AVAILABLE:
        return None, None
    vhdr = os.path.join(RAW_DATA_DIR, f"sub-{sid}", "eeg",
                        f"sub-{sid}_task-{condition}_eeg.vhdr")
    if not os.path.exists(vhdr):
        return None, None

    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
    raw.filter(1., 45., method="iir",
               iir_params={"order": 4, "ftype": "butter"}, verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.resample(FS, verbose=False)
    return raw.get_data(), raw.ch_names


def extract_epochs_raw(data, events_df, pre=100, post=400):
    epochs, labels = [], []
    n = data.shape[1]
    for _, row in events_df.iterrows():
        onset = int(row["begsample"] * FS / 1000)
        s, e = onset - pre, onset + post
        if s < 0 or e > n:
            continue
        epochs.append(data[:, s:e])
        labels.append(int(row["label"]))
    if not epochs:
        return None, None
    return np.array(epochs), np.array(labels)



#  Windowed data loader
# -------------------------------------------------

def load_windowed():
    """Load preprocessed windowed arrays from epoch.py."""
    X = np.load(os.path.join(PREP_DATA_DIR, "X_eeg_win.npy")).astype(np.float32)
    y = np.load(os.path.join(PREP_DATA_DIR, "y_win.npy")).astype(np.int64)
    s = np.load(os.path.join(PREP_DATA_DIR, "subject_ids_win.npy")).astype(np.int64)
    trial_ids = np.load(os.path.join(PREP_DATA_DIR, "trial_ids_win.npy")).astype(np.int64)

    if y.min() == 1:
        y = y - 1

    pos        = np.arange(len(X)) % len(WINDOW_STARTS)
    win_starts = WINDOW_STARTS[pos]
    return X, y, s, trial_ids, win_starts


def get_window_subset(X, y, s, trial_ids, win_starts, start):
    mask = win_starts == start
    return X[mask], y[mask], s[mask], trial_ids[mask]



#  Fisher and LogReg utilities
# -------------------------------------------------

def fisher_score(Xf, y):
    """
    Multiclass Fisher discriminant score averaged over feature dimensions.
    Near zero = classes are not separable.
    """
    grand_mean  = Xf.mean(axis=0)
    class_stats = []
    for cls in np.unique(y):
        Xc = Xf[y == cls]
        if len(Xc) == 0: continue
        class_stats.append((len(Xc), Xc.mean(axis=0), Xc.var(axis=0) + 1e-8))
    if len(class_stats) < 2:
        return 0.0
    sb = np.zeros(Xf.shape[1], dtype=np.float64)
    sw = np.zeros(Xf.shape[1], dtype=np.float64)
    for n_c, mean_c, var_c in class_stats:
        sb += n_c * (mean_c - grand_mean) ** 2
        sw += n_c * var_c
    return float(np.mean(sb / (sw + 1e-8)))


def logreg_f1(X, y, s):
    """Cross-subject logistic regression — upper bound on linear separability."""
    Xf = X.reshape(len(X), -1)
    splitter  = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    tr, te    = next(splitter.split(Xf, y, groups=s))
    clf = Pipeline([("sc", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000))])
    clf.fit(Xf[tr], y[tr])
    pred = clf.predict(Xf[te])
    return f1_score(y[te], pred, average="macro")



#  SECTION A
#  Signal quality plots (raw data)
# -------------------------------------------------

def section_A(subjects, condition):
    print("\n" + "-"*65)
    print("  SECTION A — EEG Signal Quality (raw data)")
    print("-"*65)

    eeg_list, ch_eeg, valid = [], [], []

    for sid in subjects:
        print(f"\n  sub-{sid} ...")
        eeg, chs = load_eeg_raw(sid, condition)
        if eeg is not None:
            eeg_list.append(eeg)
            ch_eeg.append(chs)
            valid.append(sid)
        else:
            print(f"    sub-{sid} skipped — loading failed")

    if not valid:
        print("  No subjects loaded for Section A.")
        return

    # ── A1: EEG channel correlation heatmap ──
    n   = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    axes = [axes] if n == 1 else list(axes)
    fig.suptitle("A1 — EEG Channel Correlation Heatmap", fontsize=13, fontweight="bold")
    for ax, data, sid, chs in zip(axes, eeg_list, valid, ch_eeg):
        sns.heatmap(np.corrcoef(data), ax=ax, cmap="RdBu_r", vmin=-1, vmax=1,
                    xticklabels=chs, yticklabels=chs, square=True,
                    linewidths=0.3, cbar_kws={"shrink": 0.8, "label": "Pearson r"})
        ax.set_title(f"Subject {sid}", fontsize=11)
        ax.tick_params(labelsize=7, rotation=45)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "A1_eeg_correlation_heatmap.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n[A1] EEG correlation heatmap → {p}")

    # ── A2: EEG PSD ──
    bands = [("Delta 0.5–4Hz", 0.5, 4, "#AED6F1"),
             ("Theta 4–8Hz",   4,   8, "#A9DFBF"),
             ("Alpha 8–12Hz",  8,  12, "#F9E79F"),
             ("Beta 13–30Hz", 13,  30, "#F1948A")]
    fig, axes = plt.subplots(n, 1, figsize=(10, 4*n), sharex=True)
    axes = [axes] if n == 1 else list(axes)
    fig.suptitle("A2 — EEG Power Spectral Density Mean ± SE", fontsize=13, fontweight="bold")
    for ax, data, sid in zip(axes, eeg_list, valid):
        f, psds = psd_welch(data, FS)
        m = 10 * np.log10(psds.mean(axis=0) + 1e-12)
        se = psds.std(axis=0) / np.sqrt(data.shape[0])
        s_db = 10 * np.log10(se + 1e-12)
        ax.plot(f, m, color="steelblue", lw=1.5, label="Mean PSD")
        ax.fill_between(f, m - s_db, m + s_db, alpha=0.25, color="steelblue")
        for name, lo, hi, col in bands:
            ax.axvspan(lo, hi, alpha=0.12, color=col, label=name)
        ax.set_xlim(0, 45); ax.set_ylabel("Power (dB)")
        ax.set_title(f"Subject {sid}")
        ax.legend(fontsize=7, ncol=3, loc="upper right"); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Frequency (Hz)")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "A2_eeg_psd.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[A2] EEG PSD → {p}")





#  SECTION B
#  EEG discriminability analysis (windowed data)
# -------------------------------------------------

def section_B():
    print("\n" + "-"*65)
    print("  SECTION B — EEG Discriminability Analysis (preprocessed)")
    print("-"*65)

    X, y, s, trial_ids, win_starts = load_windowed()
    print(f"\n  Loaded X={X.shape}  subjects={sorted(np.unique(s).tolist())}")
    print(f"  Classes: Task0={(y==0).sum()} Task1={(y==1).sum()} Task2={(y==2).sum()}")

    # ── B1: ERP overlap ──
    # Average waveform per task at C3 (channel index 0)
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.arange(80) / FS * 1000   # ms
    for task in [0, 1, 2]:
        # Use the window at onset (-100ms = start 80)
        mask = (win_starts == 80) & (y == task)
        mean = X[mask, 0, :].mean(axis=0)
        se   = X[mask, 0, :].std(axis=0) / np.sqrt(mask.sum())
        ax.plot(t, mean, color=TASK_COLORS[task], lw=2, label=TASK_LABELS[task])
        ax.fill_between(t, mean-se, mean+se, alpha=0.2, color=TASK_COLORS[task])
    ax.axvline(0, color="k", ls="--", lw=1, label="t=0 (window centre)")
    ax.set_xlabel("Time within window (ms)")
    ax.set_ylabel("Amplitude (z-score)")
    ax.set_title("B1 — ERP: C3 mean waveform per task (onset window)\n"
                )
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "B1_erp_overlap.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n[B1] ERP overlap → {p}")

    # ── B2: PSD per class at C3 ──
    # Each window is normalised to zero-mean unit-variance before computing PSD.
    # This removes amplitude offsets caused by global z-scoring and shows only
    # relative frequency content — if tasks differ in frequency distribution,
    # the lines will separate; if they overlap, there are no frequency differences.
    from scipy.signal import welch
    fig, ax = plt.subplots(figsize=(9, 4))
    for task in [0, 1, 2]:
        mask = y == task
        psds = []
        for i in np.where(mask)[0][:200]:   # sample 200 windows per class
            win = X[i, 0, :]
            # Normalise per-window to remove amplitude offset
            win = (win - win.mean()) / (win.std() + 1e-8)
            f, p = welch(win, fs=FS, nperseg=40)
            # Keep only up to 45Hz (bandpass limit)
            keep = f <= 45
            psds.append(p[keep])
        psds = np.array(psds)
        m = 10 * np.log10(psds.mean(axis=0) + 1e-12)
        ax.plot(f[f <= 45], m, color=TASK_COLORS[task], lw=2, label=TASK_LABELS[task])
    ax.axvspan(8,  12, alpha=0.12, color="#F9E79F", label="Alpha (8–12Hz)")
    ax.axvspan(13, 30, alpha=0.12, color="#F1948A", label="Beta (13–30Hz)")
    ax.set_xlim(0, 45)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalised Power (dB)")
    ax.set_title("B2 — PSD per class at C3 (per-window normalised)\n"
                 )
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "B2_psd_per_class.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[B2] PSD per class → {p}")

    # ── B3: t-SNE ──
    # Use onset window (start=80) for t-SNE
    mask = win_starts == 80
    Xw, yw = X[mask], y[mask]
    n   = min(1500, len(Xw))
    sel = np.random.choice(len(Xw), n, replace=False)
    Xf  = StandardScaler().fit_transform(Xw[sel].reshape(n, -1))
    emb = TSNE(n_components=2, random_state=RANDOM_STATE,
               perplexity=30, max_iter=1000).fit_transform(Xf)

    fig, ax = plt.subplots(figsize=(7, 6))
    for task in [0, 1, 2]:
        m = yw[sel] == task
        ax.scatter(emb[m, 0], emb[m, 1], c=TASK_COLORS[task],
                   s=8, alpha=0.5, label=TASK_LABELS[task], rasterized=True)
    ax.set_title("B3 — t-SNE of EEG (onset window, all channels)\n"
                 )
    ax.legend(); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "B3_tsne.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[B3] t-SNE → {p}")

    # ── B4: Fisher score per channel ──
    # Use onset window
    mask = win_starts == 80
    Xw, yw = X[mask], y[mask]
    ch_scores = [fisher_score(Xw[:, ch, :], yw) for ch in range(len(MOTOR_CHANNELS))]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(MOTOR_CHANNELS, ch_scores, color="#7F77DD", alpha=0.85)
    ax.set_xlabel("EEG channel")
    ax.set_ylabel("Fisher score")
    ax.set_title("B4 — Fisher discriminability per channel (onset window)\n"
                 )
    ax.grid(axis="y", alpha=0.3)
    for i, (ch, sc) in enumerate(zip(MOTOR_CHANNELS, ch_scores)):
        ax.text(i, sc + max(ch_scores)*0.01, f"{sc:.1e}",
                ha="center", fontsize=6, rotation=45)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "B4_fisher_per_channel.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[B4] Fisher per channel → {p}")
    print(f"     Max score: {max(ch_scores):.4e}  Best channel: {MOTOR_CHANNELS[np.argmax(ch_scores)]}")

    # ── B5: Per-subject separability ──
    subj_scores = []
    for sid in np.unique(s):
        mask = s == sid
        Xs, ys = X[mask], y[mask]
        if len(np.unique(ys)) < 3:
            continue
        sc = fisher_score(Xs.reshape(len(Xs), -1), ys)
        subj_scores.append((int(sid), sc))

    subj_scores.sort(key=lambda x: x[1])
    sids  = [x[0] for x in subj_scores]
    scores = [x[1] for x in subj_scores]
    chance = 1 / 3

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(sids)), scores, color="#1D9E75", alpha=0.8)
    ax.axhline(chance, color="red", ls="--", lw=1.5, label=f"Chance ({chance:.2f})")
    ax.set_xticks(range(len(sids)))
    ax.set_xticklabels(sids, fontsize=8)
    ax.set_xlabel("Subject (sorted by Fisher score)")
    ax.set_ylabel("Fisher score")
    ax.set_title("B5 — Per-subject EEG class separability\n"
                 )
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "B5_per_subject_separability.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[B5] Per-subject separability → {p}")

    # ── B6: Window discriminability ──
    fisher_scores_w, f1s_w = [], []
    for start in WINDOW_STARTS:
        Xw, yw, sw, _ = get_window_subset(X, y, s, trial_ids, win_starts, start)
        fisher_scores_w.append(fisher_score(Xw.reshape(len(Xw), -1), yw))
        f1s_w.append(logreg_f1(Xw, yw, sw))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(WINDOW_MS, fisher_scores_w, marker="o", lw=2, color="#7F77DD",
             label="Fisher score")
    ax1.set_xlabel("Window start relative to movement onset (ms)")
    ax1.set_ylabel("Fisher score")
    ax1.axvline(0, color="k", ls="--", lw=1, label="Movement onset")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(WINDOW_MS, f1s_w, marker="^", lw=2, ls=":", color="#E24B4A",
             label="LogReg macro F1")
    ax2.axhline(1/3, color="grey", ls="--", lw=1, alpha=0.7, label="Chance")
    ax2.set_ylabel("Macro F1")
    lines  = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="best")
    ax1.set_title("B6 — EEG discriminability by window position\n"
                  )
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "B6_window_discriminability.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    best = int(np.argmax(f1s_w))
    print(f"[B6] Window discriminability → {p}")
    print(f"     Best window: {WINDOW_STARTS[best]} ({WINDOW_MS[best]:.0f}ms) "
          f"F1={f1s_w[best]:.4f}")

    # ── B7: Channel × window heatmap ──
    heat = np.zeros((len(MOTOR_CHANNELS), len(WINDOW_STARTS)))
    for j, start in enumerate(WINDOW_STARTS):
        Xw, yw, _, _ = get_window_subset(X, y, s, trial_ids, win_starts, start)
        for ch in range(len(MOTOR_CHANNELS)):
            heat[ch, j] = fisher_score(Xw[:, ch, :], yw)

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(heat, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(WINDOW_STARTS)))
    ax.set_xticklabels([f"{int(ms)}" for ms in WINDOW_MS], rotation=45)
    ax.set_yticks(np.arange(len(MOTOR_CHANNELS)))
    ax.set_yticklabels(MOTOR_CHANNELS)
    ax.set_xlabel("Window start relative to movement onset (ms)")
    ax.set_ylabel("EEG channel")
    ax.set_title("B7 — Channel × window Fisher discriminability\n"
                 )
    plt.colorbar(im, ax=ax, label="Fisher score")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "B7_channel_window_heatmap.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    ch_idx, win_idx = np.unravel_index(np.argmax(heat), heat.shape)
    print(f"[B7] Channel-window heatmap → {p}")
    print(f"     Best pair: {MOTOR_CHANNELS[ch_idx]} at {WINDOW_MS[win_idx]:.0f}ms "
          f"(score={heat[ch_idx, win_idx]:.2e})")

    # ── B8: Phase summary ──
    phases = {
        "Pre\n(-500 to -300ms)":       {0,   40},
        "Onset\n(-100 to +100ms)":     {80,  120},
        "Execution\n(+100 to +700ms)": {160, 200, 240},
        "Late\n(+700 to +1500ms)":     {280, 320, 360, 400},
    }
    names_ph, fisher_ph, f1_ph = [], [], []
    for name, starts in phases.items():
        mask = np.isin(win_starts, list(starts))
        Xp, yp, sp = X[mask], y[mask], s[mask]
        names_ph.append(name)
        fisher_ph.append(fisher_score(Xp.reshape(len(Xp), -1), yp))
        f1_ph.append(logreg_f1(Xp, yp, sp))

    x   = np.arange(len(names_ph))
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x - 0.15, fisher_ph, width=0.3, label="Fisher score", color="#7F77DD", alpha=0.85)
    ax1.set_ylabel("Fisher score")
    ax1.set_xticks(x); ax1.set_xticklabels(names_ph, fontsize=9)
    ax1.set_title("B8 — EEG discriminability by trial phase\n"
                  )
    ax1.grid(axis="y", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.15, f1_ph, width=0.3, label="LogReg F1",
            color="#E24B4A", alpha=0.7)
    ax2.axhline(1/3, color="grey", ls="--", lw=1, alpha=0.7)
    ax2.set_ylabel("Macro F1")
    lines  = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="best")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "B8_phase_summary.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    best = int(np.argmax(f1_ph))
    print(f"[B8] Phase summary → {p}")
    print(f"     Best phase: {names_ph[best].split(chr(10))[0]} F1={f1_ph[best]:.4f}")

    # ── B9: All-channel Fisher analysis ──
    if not MNE_AVAILABLE:
        print("[B9] Skipped — MNE not installed")
        return

    subjects_raw = get_raw_subjects()
    print(f"\n[B9] All-channel analysis — subjects: {subjects_raw}")
    print("     Loading raw .vhdr files...")

    all_scores, ch_names = [], None

    for sid in subjects_raw:
        try:
            events = load_events_raw(sid, CONDITION)
            data, names = load_all_eeg_channels(sid, CONDITION)
        except Exception as e:
            print(f"     sub-{sid} SKIP: {e}"); continue

        if ch_names is None:
            ch_names = names

        epochs, labels = extract_epochs_raw(data, events)
        if epochs is None or len(epochs) < 5:
            continue

        scores = np.array([
            fisher_score(epochs[:, ch, :], labels)
            for ch in range(epochs.shape[1])
        ])
        all_scores.append(scores)
        print(f"     sub-{sid}  {len(labels)} trials  "
              f"top={names[np.argmax(scores)]} ({scores.max():.2e})")

    if not all_scores:
        print("     No subjects processed.")
        return

    mean_scores = np.mean(all_scores, axis=0)
    sorted_idx  = np.argsort(mean_scores)[::-1]

    motor_idx  = [i for i, n in enumerate(ch_names) if n in MOTOR_CH_SET]
    other_idx  = [i for i, n in enumerate(ch_names) if n not in MOTOR_CH_SET]
    motor_scr  = mean_scores[motor_idx]
    other_scr  = mean_scores[other_idx]

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    ax = axes[0]
    colors = ["#E24B4A" if n in MOTOR_CH_SET else "#7F77DD" for n in ch_names]
    ax.bar(range(len(ch_names)), mean_scores, color=colors, alpha=0.8)
    ax.set_xticks(range(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=5)
    ax.set_ylabel("Fisher Score")
    ax.set_title("B9 — All 128 EEG channels — Fisher discriminability\n"
                 "(red = 15 motor channels used in modelling, purple = other channels)\n"
                 "Uniform scores ~10⁻⁶ across all regions = channel selection was appropriate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[
        Patch(color="#E24B4A", label="15 motor channels"),
        Patch(color="#7F77DD", label="Other channels"),
    ])

    ax = axes[1]
    top30 = sorted_idx[:30]
    t30_names = [ch_names[i] for i in top30]
    t30_col   = ["#E24B4A" if n in MOTOR_CH_SET else "#7F77DD" for n in t30_names]
    ax.bar(range(30), mean_scores[top30], color=t30_col, alpha=0.8)
    ax.set_xticks(range(30))
    ax.set_xticklabels(t30_names, rotation=45, fontsize=8)
    ax.set_ylabel("Fisher Score")
    ax.set_title("Top 30 most discriminative channels")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "B9_all_channel_fisher.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()

    np.save(os.path.join(OUT_DIR, "all_channel_scores.npy"), mean_scores)
    np.save(os.path.join(OUT_DIR, "all_channel_names.npy"), np.array(ch_names))

    print(f"     Motor channels:  mean={motor_scr.mean():.4e}  max={motor_scr.max():.4e}")
    print(f"     Other channels:  mean={other_scr.mean():.4e}  max={other_scr.max():.4e}")
    if other_scr.max() > motor_scr.max() * 2:
        best_other = ch_names[other_idx[int(np.argmax(other_scr))]]
        print(f"     *** Non-motor channels more discriminative: {best_other}")
    else:
        print("     Motor channel selection confirmed appropriate.")
    print(f"[B9] All-channel Fisher → {p}")



#  Main
# -------------------------------------------------

def main():
    args      = parse_args()
    condition = args.condition

    print("\n" + "-"*65)
    print("  NeBULA EEG Diagnostics")
    print("  Muskaan Garg | H00416442 | Heriot-Watt University")
    print("-"*65)
    print(f"  Output directory: {OUT_DIR}/")
    print(f"  MNE available   : {MNE_AVAILABLE}")

    if args.section in ("A", "both"):
        if not MNE_AVAILABLE:
            print("\n  [Section A] Skipped — MNE not installed (pip install mne)")
        else:
            subjects = get_raw_subjects(
                requested=args.subjects,
                all_subs=args.all_subjects
            )
            section_A(subjects, condition)

    if args.section in ("B", "both"):
        section_B()

    print("\n" + "-"*65)
    print(f"  All outputs saved to: {OUT_DIR}/")
    print("-"*65)


if __name__ == "__main__":
    main()