"""
NeBULA Dataset - Timing Analysis Pipeline
-------------------------------------------------------------

 Estimate subject-specific temporal delay between EEG
 motor-cortex ERD onset and EMG muscle activation onset.

"""

import os
import glob
import json
import argparse
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import wilcoxon


# CONSTANTS
# ----------------------------------------------

DATA_DIR = "../preprocessed"
OUT_DIR = "./timing_results"

TARGET_FS = 200
EPOCH_LEN = 500                     # -500 ms to +2000 ms = 2.5 s
T_MIN_MS = -500
T_MAX_MS = 2000

# Index helpers for current epoch definition
ONSET_IDX = 100                     # movement onset at 0 ms
BASELINE_START_IDX = 0              # -500 ms
BASELINE_END_IDX = 80               # -100 ms

# Search EEG/EMG onset slightly before movement onset
SEARCH_START_IDX = 80               # -100 ms
SEARCH_END_IDX = 220                # +600 ms

# EEG channel order matches current epoch pipeline
EEG_CHANNEL_NAMES = [
    "C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4",
    "C1", "C2", "C5", "C6", "FC1", "FC2", "CP1", "CP2"
]
DEFAULT_EEG_ROI = ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4"]

# EMG onset detection params
EMG_THRESHOLD_K = 2.0               # baseline mean + 2*std
EMG_MIN_DURATION_MS = 25            # sustained crossing duration
EMG_SMOOTH_MS = 20                  # light smoothing only

# EEG ERD detection params
BETA_LOW = 13.0
BETA_HIGH = 30.0
ERD_DROP_FRACTION = 0.10            # 10% below baseline (more sensitive)
EEG_SMOOTH_MS = 50                  # lighter smoothing than before
EEG_MIN_DURATION_MS = 30            # shorter sustained drop

# Cross-correlation params
XCORR_WINDOW_START_IDX = 60         # -200 ms
XCORR_WINDOW_END_IDX = 220          # +600 ms

TASK_LABELS = {0: "Task1", 1: "Task2", 2: "Task3"}



def parse_args():
    parser = argparse.ArgumentParser(description="NeBULA EEG->EMG timing analysis")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument("--conditions", nargs="+", default=["free"],
                        help="Conditions to analyse, e.g. free low high")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--eeg-roi", nargs="+", default=DEFAULT_EEG_ROI)
    parser.add_argument("--emg-threshold-k", type=float, default=EMG_THRESHOLD_K)
    parser.add_argument("--erd-drop-fraction", type=float, default=ERD_DROP_FRACTION)
    parser.add_argument("--emg-min-duration-ms", type=float, default=EMG_MIN_DURATION_MS)
    parser.add_argument("--eeg-min-duration-ms", type=float, default=EEG_MIN_DURATION_MS)
    parser.add_argument("--eeg-smooth-ms", type=float, default=EEG_SMOOTH_MS)
    parser.add_argument("--save-trial-level", action="store_true")
    return parser.parse_args()


def ms_to_idx(ms):
    return int(round((ms - T_MIN_MS) * TARGET_FS / 1000.0))


def idx_to_ms(idx):
    return T_MIN_MS + (idx * 1000.0 / TARGET_FS)


def moving_average(x, win):
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, kernel, mode="same")


def robust_zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def first_sustained_crossing(signal, threshold, start_idx, end_idx, min_len, direction="above"):
    if direction == "above":
        mask = signal[start_idx:end_idx] > threshold
    elif direction == "below":
        mask = signal[start_idx:end_idx] < threshold
    else:
        raise ValueError("direction must be 'above' or 'below'")

    run = 0
    for i, flag in enumerate(mask):
        run = run + 1 if flag else 0
        if run >= min_len:
            return start_idx + i - min_len + 1
    return None


def bandpass_beta(eeg_trial):
    sos = butter(4, [BETA_LOW, BETA_HIGH], btype="bandpass", fs=TARGET_FS, output="sos")
    return sosfiltfilt(sos, eeg_trial, axis=1)


def beta_power_envelope(eeg_trial):
    beta = bandpass_beta(eeg_trial)
    analytic = hilbert(beta, axis=1)
    power = np.abs(analytic) ** 2
    return power.astype(np.float32)


def get_eeg_roi_indices(requested_roi):
    name_to_idx = {name: i for i, name in enumerate(EEG_CHANNEL_NAMES)}
    indices = [name_to_idx[ch] for ch in requested_roi if ch in name_to_idx]
    if not indices:
        raise ValueError("No valid EEG ROI channels found in requested ROI.")
    return indices


def load_condition_subject(sub_dir, sid_str, condition):
    eeg_path = os.path.join(sub_dir, f"sub-{sid_str}_{condition}_eeg.npy")
    emg_path = os.path.join(sub_dir, f"sub-{sid_str}_{condition}_emg.npy")
    y_path = os.path.join(sub_dir, f"sub-{sid_str}_{condition}_labels.npy")

    X_eeg = np.load(eeg_path).astype(np.float32)
    X_emg = np.load(emg_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    if y.min() == 1:
        y = y - 1

    return X_eeg, X_emg, y


def detect_emg_onset(emg_trial, threshold_k, min_duration_ms):
    """
    emg_trial shape: (n_emg_channels, n_samples)
    Assumes EMG is already envelope-like from preprocessing.
    """
    emg_signal = np.mean(emg_trial, axis=0)

    smooth_len = max(1, int(round(EMG_SMOOTH_MS * TARGET_FS / 1000.0)))
    emg_signal = moving_average(emg_signal, smooth_len)

    baseline = emg_signal[BASELINE_START_IDX:BASELINE_END_IDX]
    thresh = baseline.mean() + threshold_k * baseline.std()

    min_len = max(1, int(round(min_duration_ms * TARGET_FS / 1000.0)))
    onset_idx = first_sustained_crossing(
        signal=emg_signal,
        threshold=thresh,
        start_idx=SEARCH_START_IDX,
        end_idx=SEARCH_END_IDX,
        min_len=min_len,
        direction="above"
    )

    return onset_idx, emg_signal, float(thresh)


def detect_eeg_erd_onset(eeg_trial, roi_indices, erd_drop_fraction, smooth_ms, min_duration_ms):
    """
    eeg_trial shape: (n_eeg_channels, n_samples)
    Detect ERD from RELATIVE beta-power change from baseline.
    """
    power = beta_power_envelope(eeg_trial)              # (n_channels, n_samples)
    roi_power = power[roi_indices].mean(axis=0)

    smooth_len = max(1, int(round(smooth_ms * TARGET_FS / 1000.0)))
    roi_power_sm = moving_average(roi_power, smooth_len)

    baseline = roi_power_sm[BASELINE_START_IDX:BASELINE_END_IDX]
    baseline_mean = baseline.mean()

    # Relative change from baseline; ERD should be negative
    roi_rel = (roi_power_sm - baseline_mean) / (baseline_mean + 1e-8)
    threshold = -erd_drop_fraction

    min_len = max(1, int(round(min_duration_ms * TARGET_FS / 1000.0)))
    onset_idx = first_sustained_crossing(
        signal=roi_rel,
        threshold=threshold,
        start_idx=SEARCH_START_IDX,
        end_idx=SEARCH_END_IDX,
        min_len=min_len,
        direction="below"
    )

    return onset_idx, roi_rel, float(threshold), float(baseline_mean)


def crosscorr_peak_lag(eeg_erd_signal, emg_signal):
    """
    Cross-correlate EEG ERD-like signal with EMG envelope.
    Positive lag means EMG occurs later than EEG-like activity.
    """
    x = eeg_erd_signal[XCORR_WINDOW_START_IDX:XCORR_WINDOW_END_IDX]
    y = emg_signal[XCORR_WINDOW_START_IDX:XCORR_WINDOW_END_IDX]

    # ERD is negative deflection; invert to make it activation-like
    x = -robust_zscore(x)
    y = robust_zscore(y)

    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-len(x) + 1, len(x))

    best_idx = int(np.argmax(corr))
    best_lag_samples = int(lags[best_idx])
    best_lag_ms = best_lag_samples * 1000.0 / TARGET_FS
    best_corr = float(corr[best_idx])

    return best_lag_samples, float(best_lag_ms), best_corr, corr, lags


def analyze_trial(eeg_trial, emg_trial, label, sid, condition, trial_idx, roi_indices, args):
    eeg_onset_idx, eeg_rel, eeg_thresh, eeg_baseline = detect_eeg_erd_onset(
        eeg_trial=eeg_trial,
        roi_indices=roi_indices,
        erd_drop_fraction=args.erd_drop_fraction,
        smooth_ms=args.eeg_smooth_ms,
        min_duration_ms=args.eeg_min_duration_ms
    )

    emg_onset_idx, emg_signal, emg_thresh = detect_emg_onset(
        emg_trial=emg_trial,
        threshold_k=args.emg_threshold_k,
        min_duration_ms=args.emg_min_duration_ms
    )

    lag_samples, lag_ms, peak_corr, _, _ = crosscorr_peak_lag(eeg_rel, emg_signal)

    eeg_onset_ms = None if eeg_onset_idx is None else float(idx_to_ms(eeg_onset_idx))
    emg_onset_ms = None if emg_onset_idx is None else float(idx_to_ms(emg_onset_idx))

    delay_ms = None
    if eeg_onset_ms is not None and emg_onset_ms is not None:
        delay_ms = float(emg_onset_ms - eeg_onset_ms)

    return {
        "subject": int(sid),
        "condition": str(condition),
        "task": int(label),
        "trial_index": int(trial_idx),
        "eeg_erd_onset_idx": None if eeg_onset_idx is None else int(eeg_onset_idx),
        "eeg_erd_onset_ms": eeg_onset_ms,
        "eeg_baseline_mean": float(eeg_baseline),
        "eeg_threshold": float(eeg_thresh),
        "emg_onset_idx": None if emg_onset_idx is None else int(emg_onset_idx),
        "emg_onset_ms": emg_onset_ms,
        "emg_threshold": float(emg_thresh),
        "delay_ms": delay_ms,
        "xcorr_peak_lag_samples": int(lag_samples),
        "xcorr_peak_lag_ms": float(lag_ms),
        "xcorr_peak_corr": float(peak_corr),
    }


def safe_mean(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return None if len(vals) == 0 else float(np.mean(vals))


def safe_std(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return None if len(vals) == 0 else float(np.std(vals, ddof=0))


def count_valid(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return int(len(vals))


def summarize_subject_condition_task(trial_rows):
    grouped = {}
    for row in trial_rows:
        key = (row["subject"], row["condition"], row["task"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (sid, cond, task), rows in sorted(grouped.items()):
        delays = [r["delay_ms"] for r in rows]
        eegs = [r["eeg_erd_onset_ms"] for r in rows]
        emgs = [r["emg_onset_ms"] for r in rows]
        lags = [r["xcorr_peak_lag_ms"] for r in rows]

        summary.append({
            "subject": int(sid),
            "condition": str(cond),
            "task": int(task),
            "task_name": TASK_LABELS.get(task, f"Task{task+1}"),
            "n_trials": int(len(rows)),
            "n_valid_delay": count_valid(delays),
            "mean_eeg_erd_onset_ms": safe_mean(eegs),
            "std_eeg_erd_onset_ms": safe_std(eegs),
            "mean_emg_onset_ms": safe_mean(emgs),
            "std_emg_onset_ms": safe_std(emgs),
            "mean_delay_ms": safe_mean(delays),
            "std_delay_ms": safe_std(delays),
            "mean_xcorr_lag_ms": safe_mean(lags),
            "std_xcorr_lag_ms": safe_std(lags),
        })
    return summary


def summarize_condition_task(subject_summary):
    grouped = {}
    for row in subject_summary:
        key = (row["condition"], row["task"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (cond, task), rows in sorted(grouped.items()):
        delays = [r["mean_delay_ms"] for r in rows]
        eegs = [r["mean_eeg_erd_onset_ms"] for r in rows]
        emgs = [r["mean_emg_onset_ms"] for r in rows]
        lags = [r["mean_xcorr_lag_ms"] for r in rows]

        summary.append({
            "condition": str(cond),
            "task": int(task),
            "task_name": TASK_LABELS.get(task, f"Task{task+1}"),
            "n_subjects": int(len(rows)),
            "mean_delay_ms": safe_mean(delays),
            "std_delay_ms": safe_std(delays),
            "mean_eeg_erd_onset_ms": safe_mean(eegs),
            "mean_emg_onset_ms": safe_mean(emgs),
            "mean_xcorr_lag_ms": safe_mean(lags),
        })
    return summary


def run_condition_comparisons(subject_summary, out_dir):
    rows = []
    conds = sorted(list({r["condition"] for r in subject_summary}))
    if len(conds) < 2:
        return rows

    for cond_a, cond_b in combinations(conds, 2):
        for task in sorted(list({r["task"] for r in subject_summary})):
            a_map = {(r["subject"], r["task"]): r for r in subject_summary if r["condition"] == cond_a}
            b_map = {(r["subject"], r["task"]): r for r in subject_summary if r["condition"] == cond_b}

            common_keys = sorted(set(a_map.keys()) & set(b_map.keys()))
            a_vals = []
            b_vals = []

            for key in common_keys:
                a = a_map[key]["mean_delay_ms"]
                b = b_map[key]["mean_delay_ms"]
                if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                    a_vals.append(a)
                    b_vals.append(b)

            if len(a_vals) >= 3:
                try:
                    stat, p = wilcoxon(a_vals, b_vals)
                    rows.append({
                        "condition_a": cond_a,
                        "condition_b": cond_b,
                        "task": int(task),
                        "task_name": TASK_LABELS.get(task, f"Task{task+1}"),
                        "n_pairs": int(len(a_vals)),
                        "mean_delay_a_ms": float(np.mean(a_vals)),
                        "mean_delay_b_ms": float(np.mean(b_vals)),
                        "wilcoxon_stat": float(stat),
                        "p_value": float(p),
                    })
                except ValueError:
                    rows.append({
                        "condition_a": cond_a,
                        "condition_b": cond_b,
                        "task": int(task),
                        "task_name": TASK_LABELS.get(task, f"Task{task+1}"),
                        "n_pairs": int(len(a_vals)),
                        "mean_delay_a_ms": float(np.mean(a_vals)),
                        "mean_delay_b_ms": float(np.mean(b_vals)),
                        "wilcoxon_stat": None,
                        "p_value": None,
                    })
    return rows


def save_csv(path, rows):
    import csv
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def plot_delay_by_task(condition_task_summary, out_dir):
    if not condition_task_summary:
        return

    conditions = sorted(list({r["condition"] for r in condition_task_summary}))
    tasks = [0, 1, 2]
    x = np.arange(len(tasks))
    width = 0.8 / max(1, len(conditions))

    plt.figure(figsize=(10, 5))
    for i, cond in enumerate(conditions):
        vals = []
        errs = []
        for task in tasks:
            rows = [r for r in condition_task_summary if r["condition"] == cond and r["task"] == task]
            if rows:
                vals.append(rows[0]["mean_delay_ms"] if rows[0]["mean_delay_ms"] is not None else np.nan)
                errs.append(rows[0]["std_delay_ms"] if rows[0]["std_delay_ms"] is not None else 0.0)
            else:
                vals.append(np.nan)
                errs.append(0.0)

        plt.bar(
            x + (i - (len(conditions) - 1) / 2) * width,
            vals,
            width=width,
            yerr=errs,
            capsize=4,
            label=cond
        )

    plt.axhline(0, linestyle="--", linewidth=1, color="black")
    plt.xticks(x, [TASK_LABELS[t] for t in tasks])
    plt.ylabel("EEG→EMG delay (ms)")
    plt.title("Mean EEG→EMG delay by task and condition")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    path = os.path.join(out_dir, "01_delay_by_task_condition.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot -> {path}")


def plot_onset_scatter(trial_rows, out_dir):
    pts = [r for r in trial_rows if r["eeg_erd_onset_ms"] is not None and r["emg_onset_ms"] is not None]
    if not pts:
        return

    eeg = np.array([r["eeg_erd_onset_ms"] for r in pts], dtype=np.float32)
    emg = np.array([r["emg_onset_ms"] for r in pts], dtype=np.float32)
    tasks = np.array([r["task"] for r in pts], dtype=np.int64)

    plt.figure(figsize=(6, 6))
    for task in [0, 1, 2]:
        idx = tasks == task
        plt.scatter(eeg[idx], emg[idx], s=18, alpha=0.7, label=TASK_LABELS[task])

    lo = min(np.min(eeg), np.min(emg))
    hi = max(np.max(eeg), np.max(emg))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black")
    plt.xlabel("EEG ERD onset (ms)")
    plt.ylabel("EMG onset (ms)")
    plt.title("Trial-level EEG vs EMG onset")
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(out_dir, "02_trial_onset_scatter.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot -> {path}")


def plot_xcorr_vs_delay(subject_summary, out_dir):
    pts = [r for r in subject_summary if r["mean_delay_ms"] is not None and r["mean_xcorr_lag_ms"] is not None]
    if not pts:
        return

    delay = np.array([r["mean_delay_ms"] for r in pts], dtype=np.float32)
    xlag = np.array([r["mean_xcorr_lag_ms"] for r in pts], dtype=np.float32)

    plt.figure(figsize=(6, 6))
    plt.scatter(delay, xlag, s=35, alpha=0.8)
    lo = min(np.min(delay), np.min(xlag))
    hi = max(np.max(delay), np.max(xlag))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black")
    plt.xlabel("Threshold-based delay (ms)")
    plt.ylabel("Cross-correlation lag (ms)")
    plt.title("Delay agreement: threshold vs cross-correlation")
    plt.grid(alpha=0.3)
    path = os.path.join(out_dir, "03_delay_vs_xcorr.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot -> {path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    roi_indices = get_eeg_roi_indices(args.eeg_roi)
    print("-" * 65)
    print("TIMING ANALYSIS PIPELINE")
    print("-" * 65)
    print(f"Data dir             : {args.data_dir}")
    print(f"Output dir           : {args.out_dir}")
    print(f"Conditions           : {args.conditions}")
    print(f"EEG ROI              : {args.eeg_roi}")
    print(f"EMG threshold k      : {args.emg_threshold_k}")
    print(f"ERD drop fraction    : {args.erd_drop_fraction}")
    print(f"EMG min duration ms  : {args.emg_min_duration_ms}")
    print(f"EEG min duration ms  : {args.eeg_min_duration_ms}")
    print(f"EEG smoothing ms     : {args.eeg_smooth_ms}")
    print("-" * 65)

    sub_dirs = sorted(glob.glob(os.path.join(args.data_dir, "sub-*")))
    if args.max_subjects is not None:
        sub_dirs = sub_dirs[:args.max_subjects]

    if not sub_dirs:
        raise RuntimeError(f"No subject folders found in {args.data_dir}")

    trial_rows = []

    for sub_dir in sub_dirs:
        sid_str = os.path.basename(sub_dir).replace("sub-", "")
        sid = int(sid_str)
        print(f"\n--- Subject {sid:02d} ---")

        for condition in args.conditions:
            try:
                X_eeg, X_emg, y = load_condition_subject(sub_dir, sid_str, condition)
            except FileNotFoundError:
                print(f"  Skipping condition '{condition}' (files missing)")
                continue
            except Exception as e:
                print(f"  Error loading condition '{condition}': {e}")
                continue

            print(f"  Condition '{condition}': EEG={X_eeg.shape}, EMG={X_emg.shape}, y={y.shape}")

            n_trials = min(len(X_eeg), len(X_emg), len(y))
            for trial_idx in range(n_trials):
                row = analyze_trial(
                    eeg_trial=X_eeg[trial_idx],
                    emg_trial=X_emg[trial_idx],
                    label=int(y[trial_idx]),
                    sid=sid,
                    condition=condition,
                    trial_idx=trial_idx,
                    roi_indices=roi_indices,
                    args=args
                )
                trial_rows.append(row)

    if not trial_rows:
        raise RuntimeError("No trials were analyzed. Check input files and condition names.")

    subject_summary = summarize_subject_condition_task(trial_rows)
    condition_task_summary = summarize_condition_task(subject_summary)
    stats_rows = run_condition_comparisons(subject_summary, args.out_dir)

    print("\n" + "-" * 65)
    print("CONDITION / TASK SUMMARY")
    print("-" * 65)
    for row in condition_task_summary:
        print(
            f"{row['condition']:>8} | {row['task_name']:<5} | "
            f"n_sub={row['n_subjects']:02d} | "
            f"delay={row['mean_delay_ms'] if row['mean_delay_ms'] is not None else np.nan:7.2f} ms | "
            f"EEG={row['mean_eeg_erd_onset_ms'] if row['mean_eeg_erd_onset_ms'] is not None else np.nan:7.2f} ms | "
            f"EMG={row['mean_emg_onset_ms'] if row['mean_emg_onset_ms'] is not None else np.nan:7.2f} ms | "
            f"XCorr={row['mean_xcorr_lag_ms'] if row['mean_xcorr_lag_ms'] is not None else np.nan:7.2f} ms"
        )

    if stats_rows:
        print("\nPairwise Wilcoxon condition comparisons:")
        for row in stats_rows:
            print(
                f"{row['condition_a']} vs {row['condition_b']} | {row['task_name']} | "
                f"n={row['n_pairs']} | p={row['p_value']}"
            )

    plot_delay_by_task(condition_task_summary, args.out_dir)
    plot_onset_scatter(trial_rows, args.out_dir)
    plot_xcorr_vs_delay(subject_summary, args.out_dir)

    save_csv(os.path.join(args.out_dir, "subject_condition_task_summary.csv"), subject_summary)
    save_csv(os.path.join(args.out_dir, "condition_task_summary.csv"), condition_task_summary)
    if stats_rows:
        save_csv(os.path.join(args.out_dir, "condition_wilcoxon_tests.csv"), stats_rows)

    payload = {
        "meta": {
            "data_dir": args.data_dir,
            "conditions": args.conditions,
            "eeg_roi": args.eeg_roi,
            "target_fs": TARGET_FS,
            "baseline_window_ms": [idx_to_ms(BASELINE_START_IDX), idx_to_ms(BASELINE_END_IDX)],
            "search_window_ms": [idx_to_ms(SEARCH_START_IDX), idx_to_ms(SEARCH_END_IDX - 1)],
            "xcorr_window_ms": [idx_to_ms(XCORR_WINDOW_START_IDX), idx_to_ms(XCORR_WINDOW_END_IDX - 1)],
            "emg_threshold_k": args.emg_threshold_k,
            "erd_drop_fraction": args.erd_drop_fraction,
            "emg_min_duration_ms": args.emg_min_duration_ms,
            "eeg_min_duration_ms": args.eeg_min_duration_ms,
            "eeg_smooth_ms": args.eeg_smooth_ms,
            "emg_smooth_ms": EMG_SMOOTH_MS,
        },
        "subject_condition_task_summary": subject_summary,
        "condition_task_summary": condition_task_summary,
        "condition_wilcoxon_tests": stats_rows,
    }
    save_json(os.path.join(args.out_dir, "timing_analysis_summary.json"), payload)

    if args.save_trial_level:
        save_json(os.path.join(args.out_dir, "trial_level_timing_results.json"), trial_rows)

    print(f"\nDone. Outputs saved in: {args.out_dir}")


if __name__ == "__main__":
    main()