"""
NeBULA Dataset - Feasibility & Preprocessing Validation Script
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import zscore
from scipy.ndimage import uniform_filter1d

try:
    import mne
    mne.set_log_level('WARNING')
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("WARNING: MNE not installed. Run: pip install mne")


# ═══════════════════════════════════════════════════════════════════════════
# PATH DETECTION — finds data/ relative to script location
# ═══════════════════════════════════════════════════════════════════════════

def find_data_root(override=None):
    if override:
        if os.path.isdir(override):
            return os.path.abspath(override)
        print(f"  ERROR: Specified data dir not found: {override}")
        sys.exit(1)

    # Script directory and current working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd        = os.getcwd()

    candidates = [
        os.path.join(cwd,        "data"),          # ./data/
        os.path.join(cwd,        "..", "data"),     # ../data/ (when run from eda/)
        os.path.join(script_dir, "data"),           # <script_dir>/data/
        os.path.join(script_dir, "..", "data"),     # <script_dir>/../data/
        cwd,                                        # ./ directly
        script_dir,                                 # <script_dir>/ directly
    ]

    for path in candidates:
        if os.path.isdir(path):
            # Check it contains at least one sub-XX folder
            contents = os.listdir(path) if os.path.exists(path) else []
            if any(d.startswith('sub-') for d in contents):
                return os.path.abspath(path)

    return None


def get_subjects(data_root, requested=None, all_subs=False):
    """
    Auto-detect all subject folders, excluding sub-03 (noisy EMG).
    If requested list given, use that instead.
    """
    found = sorted([
        d.replace('sub-', '') for d in os.listdir(data_root)
        if d.startswith('sub-') and os.path.isdir(os.path.join(data_root, d))
    ])
    # Always exclude sub-03 — documented EMG noise (Garro et al. 2025)
    if '03' in found:
        found.remove('03')

    if requested:
        # Filter to only requested subjects that actually exist
        valid = [s for s in requested if s in found]
        missing = [s for s in requested if s not in found]
        if missing:
            print(f"  WARNING: Requested subjects not found: {missing}")
        return valid

    if all_subs:
        return found

    # Default: use first 3 for quick feasibility check
    return found[:3] if len(found) >= 3 else found


# ═══════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='NeBULA EEG-EMG Feasibility & Preprocessing Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python feasibility.py                          # auto-detect, first 3 subjects
  python feasibility.py --all-subjects           # run on all subjects
  python feasibility.py --subjects 01 02 04 05   # specific subjects
  python feasibility.py --condition high         # different condition
  python feasibility.py --data-dir ./data        # explicit data folder
        """
    )
    p.add_argument('--data-dir',      default=None,
                   help='Path to data folder containing sub-XX folders (auto-detected if omitted)')
    p.add_argument('--subjects',      nargs='+', default=None, metavar='ID',
                   help='Subject IDs e.g. --subjects 01 02 04')
    p.add_argument('--all-subjects',  action='store_true',
                   help='Process ALL available subjects')
    p.add_argument('--condition',     default='free',
                   choices=['free', 'low', 'high'],
                   help='Condition: free=unassisted (default), low, high')
    p.add_argument('--output',        default='./feasibility_plots',
                   help='Output directory for plots (default: ./feasibility_plots)')
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

EEG_FS   = 1000   # original sampling rate
EMG_FS   = 1000
FS_OUT   = 200    # target after resampling

MOTOR_CH = ['C3', 'C4', 'Cz', 'FC3', 'FC4', 'CP3', 'CP4',
            'C1', 'C2', 'C5', 'C6', 'FC1', 'FC2', 'CP1', 'CP2']


def bandpass(data, lo, hi, fs, order=4):
    nyq = fs / 2.0
    b, a = signal.butter(order, [lo/nyq, hi/nyq], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)

def notch(data, freq, fs, Q=30):
    b, a = signal.iirnotch(freq / (fs / 2.0), Q)
    return signal.filtfilt(b, a, data, axis=-1)

def resamp(data, fs_orig, fs_target):
    n = int(data.shape[-1] * fs_target / fs_orig)
    return signal.resample(data, n, axis=-1)

def rms_envelope(data, win_sec=0.1, fs=1000):
    win      = int(win_sec * fs)
    smoothed = uniform_filter1d(data ** 2, size=win, axis=-1)
    return np.sqrt(smoothed)

def psd_welch(data, fs):
    psds = []
    for ch in range(data.shape[0]):
        f, p = signal.welch(data[ch], fs=fs,
                            nperseg=min(512, data.shape[-1] // 4))
        psds.append(p)
    return f, np.array(psds)

def quality_report(data, label):
    flat = int((np.std(data, axis=-1) < 1e-6).sum())
    nans = int(np.isnan(data).sum())
    print(f"    [{label}]  shape={data.shape}  "
          f"mean_amp={np.mean(np.abs(data)):.4f}  "
          f"NaNs={nans}  flat_ch={flat}")


# ═══════════════════════════════════════════════════════════════════════════
# EEG LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_eeg(data_root, subject, condition):
    if not MNE_AVAILABLE:
        print("    [EEG] Skipped — MNE not installed.")
        return None, None

    vhdr = os.path.join(data_root, f"sub-{subject}", "eeg",
                        f"sub-{subject}_task-{condition}_eeg.vhdr")
    if not os.path.exists(vhdr):
        print(f"    [EEG] File not found: {vhdr}")
        return None, None

    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
    print(f"    [EEG] Loaded: {len(raw.ch_names)} ch @ "
          f"{int(raw.info['sfreq'])} Hz | Duration: {raw.times[-1]:.1f}s")

    available = [c for c in MOTOR_CH if c in raw.ch_names]
    if len(available) < 3:
        print(f"    [EEG] Motor channels not matched — using first 15.")
        available = raw.ch_names[:15]
    else:
        print(f"    [EEG] Motor channels ({len(available)}): {available}")
    raw.pick_channels(available)

    data = raw.get_data()
    data = signal.detrend(data, axis=-1)         # 1. detrend
    data = bandpass(data, 1, 45, EEG_FS)         # 2. bandpass 1–45 Hz
    data = notch(data, 50, EEG_FS)               # 3. notch 50 Hz
    data -= data.mean(axis=0, keepdims=True)     # 4. CAR re-reference
    data = resamp(data, EEG_FS, FS_OUT)          # 5. resample → 200 Hz
    data = zscore(data, axis=-1)                 # 6. z-score normalise

    quality_report(data, "EEG")
    return data, available


# ═══════════════════════════════════════════════════════════════════════════
# EMG LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_emg(data_root, subject, condition):
    csv_path = os.path.join(data_root, f"sub-{subject}", "emg",
                            f"sub-{subject}_task-{condition}_emg.csv")
    if not os.path.exists(csv_path):
        print(f"    [EMG] File not found: {csv_path}")
        return None, None

    # Muscle names from channels.tsv
    ch_tsv = os.path.join(data_root, f"sub-{subject}", "emg",
                          f"sub-{subject}_channels.tsv")
    ch_cols = (list(pd.read_csv(ch_tsv, sep='\t')['name'])
               if os.path.exists(ch_tsv) else [f"ch_{i}" for i in range(11)])

    # CSV format: (channels × samples), no header, 1000 Hz
    print(f"    [EMG] Reading CSV...")
    data = pd.read_csv(csv_path, header=None,
                       dtype=np.float32).values.astype(np.float64)
    print(f"    [EMG] Loaded: {data.shape[0]} ch × {data.shape[1]} samples @ 1000Hz")
    print(f"    [EMG] Muscles: {ch_cols}")

    # Drop flat channels
    flat = np.std(data, axis=-1) < 1e-6
    if flat.any():
        print(f"    [EMG] Removing {flat.sum()} flat channels")
        data    = data[~flat]
        ch_cols = [c for c, f in zip(ch_cols, flat) if not f]

    data = bandpass(data, 20, 400, EMG_FS)        # 1. bandpass 20–400 Hz
    data = notch(data, 50, EMG_FS)                # 2. notch 50 Hz
    data = np.abs(data)                           # 3. full-wave rectify
    data = rms_envelope(data, win_sec=0.1,
                        fs=EMG_FS)                # 4. RMS envelope
    data = resamp(data, EMG_FS, FS_OUT)           # 5. resample → 200 Hz
    data = zscore(data, axis=-1)                  # 6. z-score normalise

    quality_report(data, "EMG")
    return data, ch_cols


# ═══════════════════════════════════════════════════════════════════════════
# EVENTS INSPECTOR
# ═══════════════════════════════════════════════════════════════════════════

def inspect_events(data_root, subject, condition):
    ev_path = os.path.join(data_root, f"sub-{subject}", "emg",
                           f"sub-{subject}_task-{condition}_events.tsv")
    if not os.path.exists(ev_path):
        print(f"    [EVENTS] Not found: {ev_path}")
        return
    df = pd.read_csv(ev_path, sep='\t')
    g  = df[df['value'].astype(str).str.startswith('G', na=False)]
    print(f"    [EVENTS] {len(df)} total events | "
          f"Trials per task: {dict(g['type'].value_counts().sort_index())} "
          f"| Total trials: {len(g)}")


# ═══════════════════════════════════════════════════════════════════════════
# FEASIBILITY PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def fig1_eeg_heatmap(eeg_list, subs, ch_list, condition, out_dir):
    n    = len(eeg_list)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    axes = [axes] if n == 1 else list(axes)
    fig.suptitle(f"Fig 1 — EEG Channel Correlation Heatmap (task-{condition})",
                 fontsize=13, fontweight='bold')
    for ax, data, sid, chs in zip(axes, eeg_list, subs, ch_list):
        sns.heatmap(np.corrcoef(data), ax=ax, cmap='RdBu_r', vmin=-1, vmax=1,
                    xticklabels=chs, yticklabels=chs, square=True,
                    linewidths=0.3, cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
        ax.set_title(f"Subject {sid}", fontsize=11)
        ax.tick_params(labelsize=7, rotation=45)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig1_eeg_correlation_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}")


def fig2_eeg_psd(eeg_list, subs, condition, out_dir, fs=FS_OUT):
    fig, axes = plt.subplots(len(eeg_list), 1,
                             figsize=(10, 4*len(eeg_list)), sharex=True)
    axes = [axes] if len(eeg_list) == 1 else list(axes)
    fig.suptitle(f"Fig 2 — EEG Power Spectral Density Mean ± SE (task-{condition})",
                 fontsize=13, fontweight='bold')
    bands = [('Delta 0.5–4Hz', 0.5, 4, '#AED6F1'),
             ('Theta 4–8Hz',   4,   8, '#A9DFBF'),
             ('Alpha 8–12Hz',  8,  12, '#F9E79F'),
             ('Beta 13–30Hz', 13,  30, '#F1948A')]
    for ax, data, sid in zip(axes, eeg_list, subs):
        f, psds = psd_welch(data, fs)
        m = 10 * np.log10(psds.mean(axis=0) + 1e-12)
        s = 10 * np.log10(psds.std(axis=0) / np.sqrt(data.shape[0]) + 1e-12)
        ax.plot(f, m, color='steelblue', lw=1.5, label='Mean PSD')
        ax.fill_between(f, m-s, m+s, alpha=0.25, color='steelblue', label='±1 SE')
        for name, lo, hi, col in bands:
            ax.axvspan(lo, hi, alpha=0.12, color=col, label=name)
        ax.set_xlim(0, 45); ax.set_ylabel("Power (dB)")
        ax.set_title(f"Subject {sid}")
        ax.legend(fontsize=7, ncol=3, loc='upper right')
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Frequency (Hz)")
    plt.tight_layout()
    out = os.path.join(out_dir, "fig2_eeg_psd.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved → {out}")


def fig3_emg_envelope(emg_list, subs, condition, out_dir, fs=FS_OUT):
    fig, axes = plt.subplots(len(emg_list), 1,
                             figsize=(12, 4*len(emg_list)))
    axes = [axes] if len(emg_list) == 1 else list(axes)
    fig.suptitle(f"Fig 3 — EMG RMS Envelope (task-{condition})",
                 fontsize=13, fontweight='bold')
    for ax, data, sid in zip(axes, emg_list, subs):
        t   = np.arange(data.shape[1]) / fs
        ax.plot(t, data[0], color='steelblue', alpha=0.5, lw=0.7,
                label='Biceps (ch-0, z-scored RMS)')
        for i in range(1, min(5, data.shape[0])):
            ax.plot(t, data[i], color='grey', alpha=0.15, lw=0.5)
        win = int(0.3 * fs)
        ax.plot(t, np.convolve(np.abs(data[0]), np.ones(win)/win, mode='same'),
                color='crimson', lw=1.5, label='300ms smoothed envelope')
        ax.set_ylabel("Amplitude (z-score)")
        ax.set_title(f"Subject {sid}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    out = os.path.join(out_dir, "fig3_emg_envelope.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved → {out}")


def fig4_freq_comparison(eeg_list, emg_list, subs, condition, out_dir, fs=FS_OUT):
    n    = len(eeg_list)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4))
    axes = [axes] if n == 1 else list(axes)
    fig.suptitle(f"Fig 4 — EEG vs EMG Frequency Distribution (task-{condition})",
                 fontsize=13, fontweight='bold')
    for ax, eeg, emg, sid in zip(axes, eeg_list, emg_list, subs):
        fe, pe = psd_welch(eeg, fs); fm, pm = psd_welch(emg, fs)
        ax.plot(fe, 10*np.log10(pe.mean(axis=0)+1e-12),
                color='steelblue', lw=1.5, label='EEG mean PSD')
        ax.plot(fm, 10*np.log10(pm.mean(axis=0)+1e-12),
                color='crimson',   lw=1.5, label='EMG mean PSD')
        ax.axvspan(8,  12,    alpha=0.10, color='steelblue', label='Alpha (EEG)')
        ax.axvspan(13, 30,    alpha=0.10, color='navy',      label='Beta (EEG)')
        ax.axvspan(20, fs//2, alpha=0.06, color='salmon',    label='EMG dominant')
        ax.set_xlim(0, fs//2); ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)"); ax.set_title(f"Subject {sid}")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig4_eeg_vs_emg_frequency.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved → {out}")


def alignment_check(eeg_list, emg_list, subs):
    print(f"\n  {'Subject':<10} {'EEG samples':<16} {'EMG samples':<16} Status")
    print("  " + "─"*52)
    for eeg, emg, sid in zip(eeg_list, emg_list, subs):
        el = eeg.shape[1] if eeg is not None else 0
        ml = emg.shape[1] if emg is not None else 0
        ok = "✓ ALIGNED" if abs(el-ml) < FS_OUT else f"✗ DIFF={abs(el-ml)}"
        print(f"  sub-{sid:<6} {el:<16} {ml:<16} {ok}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args      = parse_args()
    data_root = find_data_root(args.data_dir)

    if data_root is None:
        print("\n  ERROR: Could not find data folder with sub-XX subjects.")
        print("  Make sure your data is in ./data/ or run:")
        print("    python feasibility.py --data-dir /path/to/data")
        sys.exit(1)

    subjects  = get_subjects(data_root,
                             requested=args.subjects,
                             all_subs=args.all_subjects)
    condition = args.condition
    out_dir   = args.output
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "="*65)
    print("  NeBULA — Feasibility & Preprocessing Validation")
    print("  Muskaan Garg | H00416442 | Heriot-Watt University")
    print("="*65)
    print(f"\n  Data root    : {data_root}")
    print(f"  All found    : {get_subjects(data_root, all_subs=True)}")
    print(f"  Processing   : {subjects}")
    print(f"  Condition    : task-{condition}  (free=unassisted)")
    print(f"  Output       : {out_dir}/\n")

    eeg_list, emg_list, ch_eeg, valid = [], [], [], []

    for sid in subjects:
        print(f"\n{'─'*50}")
        print(f"  Subject {sid}  |  task-{condition}")
        print(f"{'─'*50}")
        inspect_events(data_root, sid, condition)
        eeg, eeg_chs = load_eeg(data_root, sid, condition)
        emg, _       = load_emg(data_root, sid, condition)

        if eeg is not None and emg is not None:
            eeg_list.append(eeg); emg_list.append(emg)
            ch_eeg.append(eeg_chs); valid.append(sid)
        else:
            print(f"  Skipping sub-{sid} — loading failed.")

    if not valid:
        print("\n  ERROR: No subjects loaded. Check your data folder structure.")
        sys.exit(1)

    print(f"\n{'─'*50}")
    print(f"  Generating plots for {len(valid)} subjects: {valid}")
    print(f"{'─'*50}\n")

    fig1_eeg_heatmap(eeg_list, valid, ch_eeg, condition, out_dir)
    fig2_eeg_psd(eeg_list, valid, condition, out_dir)
    fig3_emg_envelope(emg_list, valid, condition, out_dir)
    fig4_freq_comparison(eeg_list, emg_list, valid, condition, out_dir)

    print(f"\n  ── Alignment Check ──")
    alignment_check(eeg_list, emg_list, valid)

    print(f"\n{'='*65}")
    print("  FINAL SUMMARY")
    print(f"{'='*65}")
    for sid, eeg, emg in zip(valid, eeg_list, emg_list):
        print(f"  sub-{sid}  EEG={eeg.shape}  EMG={emg.shape}")
    print(f"\n  Plots saved to: {out_dir}/")
    print("="*65)


if __name__ == "__main__":
    main()