"""
NeBULA — Subject-Dependent Epoch Windowing
==========================================
EEG-EMG Fusion for Upper-Limb Movement Decoding
Muskaan Garg | H00416442 | Heriot-Watt University

PURPOSE:
  Loads the per-subject trial arrays already saved by epoch.py
  (preprocessed/sub-XX/sub-XX_free_eeg.npy etc.), applies sliding
  window, and splits chronologically by trial order into
  train / val / test for subject-dependent training.

WHY SUBJECT-DEPENDENT:
  Each model is trained and tested on the SAME subject.
  This tests how well a model learns one person's patterns,
  rather than generalising across people.
  EEG→EMG timing delay is subject-specific (RQ3), so this
  approach is also used for the timing analysis pipeline.

SPLIT (chronological by trial):
  Train : first 70% of trials  (~20 trials → ~220 windows)
  Val   : next  15% of trials  (~4  trials → ~44  windows)
  Test  : last  15% of trials  (~5  trials → ~55  windows)

  Chronological split is essential — random split would leak
  future trial information into training.

WINDOW:
  size=80 samples (400ms at 200Hz)
  step=40 samples (200ms, 50% overlap)
  → ~11 windows per trial

OUTPUT (saved to preprocessed_sub_dep/sub-XX/):
  X_eeg_train.npy, X_eeg_val.npy, X_eeg_test.npy
  X_emg_train.npy, X_emg_val.npy, X_emg_test.npy
  y_train.npy,     y_val.npy,     y_test.npy

Usage:
    python epoch_sub_dep.py
    python epoch_sub_dep.py --subjects 01 02 04
    python epoch_sub_dep.py --all-subjects
"""

import os
import sys
import argparse
import numpy as np

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

WIN_SIZE = 80
WIN_STEP = 40

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test gets the remainder


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def find_preprocessed_root(explicit=None):
    candidates = [explicit, './preprocessed', '../preprocessed']
    for c in candidates:
        if c and os.path.isdir(c):
            subs = [d for d in os.listdir(c) if d.startswith('sub-')]
            if subs:
                return c
    sys.exit("ERROR: Cannot find preprocessed/ folder. "
             "Run epoch.py first, then re-run this script.")


def get_subjects(pre_root, requested=None):
    found = sorted([
        d.replace('sub-', '') for d in os.listdir(pre_root)
        if d.startswith('sub-') and os.path.isdir(os.path.join(pre_root, d))
    ])
    if requested:
        return [s.zfill(2) for s in requested]
    return found


def apply_windowing(epochs, labels):
    """
    Slide window across each trial epoch.

    Input:  epochs (n_trials, n_channels, epoch_len)
            labels (n_trials,)
    Output: windows    (n_windows, n_channels, WIN_SIZE)
            win_labels (n_windows,)
            trial_ids  (n_windows,)
    """
    windows, win_labels, trial_ids = [], [], []
    for i, (epoch, label) in enumerate(zip(epochs, labels)):
        n = epoch.shape[1]
        for s in range(0, n - WIN_SIZE + 1, WIN_STEP):
            windows.append(epoch[:, s : s + WIN_SIZE])
            win_labels.append(label)
            trial_ids.append(i)
    return (np.array(windows),
            np.array(win_labels),
            np.array(trial_ids))


def chronological_split(n_trials):
    """
    Returns (train_idx, val_idx, test_idx) as arrays of trial indices.
    Split is strictly chronological — no shuffling.
    """
    t1 = int(n_trials * TRAIN_RATIO)
    t2 = int(n_trials * (TRAIN_RATIO + VAL_RATIO))
    train = np.arange(0, t1)
    val   = np.arange(t1, t2)
    test  = np.arange(t2, n_trials)
    return train, val, test


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Subject-dependent windowing for NeBULA')
    parser.add_argument('pre_root',      nargs='?', default=None,
                        help='Path to preprocessed/ folder')
    parser.add_argument('--subjects',    nargs='+')
    parser.add_argument('--all-subjects', action='store_true')
    parser.add_argument('--output',      default='./preprocessed_sub_dep')
    args = parser.parse_args()

    pre_root = find_preprocessed_root(args.pre_root)
    subjects = get_subjects(pre_root, args.subjects)
    os.makedirs(args.output, exist_ok=True)

    print("=" * 65)
    print("  NeBULA — Subject-Dependent Windowing")
    print("  Muskaan Garg | H00416442 | Heriot-Watt University")
    print("=" * 65)
    print(f"  Source     : {pre_root}")
    print(f"  Output     : {args.output}")
    print(f"  Subjects   : {subjects}")
    print(f"  Window     : size={WIN_SIZE} ({WIN_SIZE*5}ms), "
          f"step={WIN_STEP} ({WIN_STEP*5}ms)")
    print(f"  Split      : train={int(TRAIN_RATIO*100)}% / "
          f"val={int(VAL_RATIO*100)}% / "
          f"test={int((1-TRAIN_RATIO-VAL_RATIO)*100)}% (by trial order)")
    print()

    processed = 0

    for sid in subjects:
        sub_in  = os.path.join(pre_root, f'sub-{sid}')
        sub_out = os.path.join(args.output, f'sub-{sid}')

        # ── Check source files exist ──
        eeg_path = os.path.join(sub_in, f'sub-{sid}_free_eeg.npy')
        emg_path = os.path.join(sub_in, f'sub-{sid}_free_emg.npy')
        lbl_path = os.path.join(sub_in, f'sub-{sid}_free_labels.npy')

        if not all(os.path.exists(p) for p in [eeg_path, emg_path, lbl_path]):
            print(f"  sub-{sid}  SKIP — source arrays not found in {sub_in}")
            print(f"           Run epoch.py --all-subjects first.")
            continue

        # ── Load trial arrays ──
        eeg    = np.load(eeg_path)    # (n_trials, 15, 500)
        emg    = np.load(emg_path)    # (n_trials, 11, 500)
        labels = np.load(lbl_path)    # (n_trials,)

        n_trials = len(labels)
        train_idx, val_idx, test_idx = chronological_split(n_trials)

        print(f"  ── Subject {sid} ──────────────────────────────────────")
        print(f"    Trials: {n_trials}  "
              f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")

        # ── Window each split ──
        os.makedirs(sub_out, exist_ok=True)

        for split_name, idx in [('train', train_idx),
                                  ('val',   val_idx),
                                  ('test',  test_idx)]:
            eeg_w, y_w, _ = apply_windowing(eeg[idx], labels[idx])
            emg_w, _,  _  = apply_windowing(emg[idx], labels[idx])

            np.save(os.path.join(sub_out, f'X_eeg_{split_name}.npy'), eeg_w)
            np.save(os.path.join(sub_out, f'X_emg_{split_name}.npy'), emg_w)
            np.save(os.path.join(sub_out, f'y_{split_name}.npy'),     y_w)

            print(f"    {split_name:5s}: {len(y_w):4d} windows  "
                  f"(Task1={(y_w==1).sum()}, "
                  f"Task2={(y_w==2).sum()}, "
                  f"Task3={(y_w==3).sum()})")

        processed += 1

    print()
    print("=" * 65)
    print(f"  Subjects processed : {processed}")
    print(f"  Saved to           : {args.output}/sub-XX/")
    print(f"  Files per subject  : X_eeg_train/val/test.npy")
    print(f"                       X_emg_train/val/test.npy")
    print(f"                       y_train/val/test.npy")
    print("=" * 65)


if __name__ == '__main__':
    main()