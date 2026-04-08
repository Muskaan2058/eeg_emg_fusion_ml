"""
NeBULA Dataset - Epoching, Trial Extraction & Windowing
-------------------------------------------------------------
EEG-EMG Fusion for Upper-Limb Movement Decoding
Muskaan Garg | H00416442 | Heriot-Watt University

OUTPUT (saved to ./preprocessed/):
  Per-trial arrays (for timing analysis, subject-dependent):
    sub-XX/sub-XX_free_eeg.npy        shape (n_trials, 15, 500)
    sub-XX/sub-XX_free_emg.npy        shape (n_trials, 11, 500)
    sub-XX/sub-XX_free_labels.npy     shape (n_trials,)

  Combined windowed arrays (for cross-subject model training):
    X_eeg_win.npy      shape (n_windows_total, 15, 80)
    X_emg_win.npy      shape (n_windows_total, 11, 80)
    y_win.npy          shape (n_windows_total,)
    subject_ids_win.npy  shape (n_windows_total,)   which subject each window came from
    trial_ids_win.npy    shape (n_windows_total,)   which trial each window came from

"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

try:
    import mne
    mne.set_log_level('WARNING')
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("WARNING: MNE not installed. Run: pip install mne")

#  Constants
# ---------------------------------------

MOTOR_CH = ['C3','C4','Cz','FC3','FC4','CP3','CP4',
            'C1','C2','C5','C6','FC1','FC2','CP1','CP2']

EEG_FS     = 1000     # raw EEG sampling rate (Hz)
EMG_FS     = 1000     # raw EMG sampling rate (Hz)
TARGET_FS  = 200      # resample both to this

# Epoch window around G event (in samples at TARGET_FS)
PRE_SAMPLES  = int(0.5 * TARGET_FS)   # 100 samples = 500ms before onset
POST_SAMPLES = int(2.0 * TARGET_FS)   # 400 samples = 2000ms after onset
EPOCH_LEN    = PRE_SAMPLES + POST_SAMPLES  # 500 samples total

# Sliding window parameters
WIN_SIZE  = 80    # samples (400ms at 200Hz)
WIN_STEP  = 40    # samples (200ms, 50% overlap)

CONDITION = 'task-free'   # unassisted condition for classification


# ─────────────────────────────────────────────
#  Path helpers
# ─────────────────────────────────────────────

def find_data_root(explicit=None):
    candidates = [
        explicit,
        './data',
        '../data',
        os.path.join(os.getcwd(), 'data'),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            subs = [d for d in os.listdir(c) if d.startswith('sub-')]
            if subs:
                return c
    sys.exit("ERROR: Cannot find data folder. Pass it as first argument.")


def get_subjects(data_root, requested=None, all_subs=False):
    found = sorted([
        d.replace('sub-', '') for d in os.listdir(data_root)
        if d.startswith('sub-') and os.path.isdir(os.path.join(data_root, d))
    ])
    if requested:
        return [s.zfill(2) for s in requested]
    if all_subs or True:
        return found
    return found[:3]



#  Signal loading & preprocessing
# ---------------------------------------

def load_events(sid, data_root):
    path = os.path.join(data_root, f'sub-{sid}', 'emg',
                        f'sub-{sid}_{CONDITION}_events.tsv')
    ev = pd.read_csv(path, sep='\t')
    g  = ev[ev['value'].str.startswith('G', na=False)].copy()
    g['label'] = g['type'].astype(int)     # task type: 1, 2, or 3
    return g.reset_index(drop=True)


def load_eeg(sid, data_root):
    if not MNE_AVAILABLE:
        return None
    vhdr = os.path.join(data_root, f'sub-{sid}', 'eeg',
                        f'sub-{sid}_{CONDITION}_eeg.vhdr')
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)

    # Select motor channels
    available = [ch for ch in MOTOR_CH if ch in raw.ch_names]
    raw.pick_channels(available)

    # Preprocessing pipeline
    raw.filter(1., 45., method='iir',
               iir_params={'order': 4, 'ftype': 'butter'},
               verbose=False)                  # bandpass 1-45 Hz
    raw.notch_filter(50., verbose=False)       # 50 Hz notch
    raw.set_eeg_reference('average',
                          projection=False, verbose=False)  # CAR
    raw.resample(TARGET_FS, verbose=False)     # → 200 Hz

    data = raw.get_data()   # (n_ch, n_samples) in Volts

    # Detrend: remove slow linear drift per channel
    from scipy.signal import detrend
    data = detrend(data, axis=1)

    # Z-score per channel: mean=0, std=1 across the full recording
    data = (data - data.mean(axis=1, keepdims=True)) / \
           (data.std(axis=1, keepdims=True) + 1e-8)
    return data


def load_emg(sid, data_root):
    from scipy import signal as sp_signal
    from scipy.ndimage import uniform_filter1d

    csv = os.path.join(data_root, f'sub-{sid}', 'emg',
                       f'sub-{sid}_{CONDITION}_emg.csv')
    ch_tsv = os.path.join(data_root, f'sub-{sid}', 'emg',
                          f'sub-{sid}_channels.tsv')

    df   = pd.read_csv(csv, header=None)
    data = df.values.astype(np.float32)   # (11, n_samples)

    ch_names = pd.read_csv(ch_tsv, sep='\t')['name'].tolist()

    # Drop flat channels
    stds  = data.std(axis=1)
    valid = stds > 1e-6
    data  = data[valid]

    # Bandpass 20–400 Hz
    sos = sp_signal.butter(4, [20, 400], btype='bandpass',
                           fs=EMG_FS, output='sos')
    data = sp_signal.sosfiltfilt(sos, data, axis=1)

    # Notch 50 Hz
    b, a = sp_signal.iirnotch(50., Q=30., fs=EMG_FS)
    data = sp_signal.filtfilt(b, a, data, axis=1)

    # Rectify + RMS envelope - 100ms window = 100 samples at 1000Hz
    data = np.abs(data)
    win  = int(0.1 * EMG_FS)   # 100ms
    data = np.sqrt(uniform_filter1d(data**2, size=win, axis=1))

    # Resample to 200 Hz using integer slicing
    data = data[:, ::5]   # 1000Hz → 200Hz

    # z-score per channel
    data = (data - data.mean(axis=1, keepdims=True)) / \
           (data.std(axis=1, keepdims=True) + 1e-8)
    return data



#  Epoch extraction
# ---------------------------------------

def extract_epochs(eeg, emg, events_df):
    """
    Cut fixed windows around each G event.
    Returns arrays of shape (n_trials, n_channels, EPOCH_LEN).
    """
    eeg_epochs, emg_epochs, labels = [], [], []
    skipped = 0

    for _, row in events_df.iterrows():
        # begsample is at 1000Hz, convert to 200Hz
        onset_200 = int(row['begsample'] * TARGET_FS / EEG_FS)
        start     = onset_200 - PRE_SAMPLES
        end       = onset_200 + POST_SAMPLES

        # Skip if window falls outside the recording
        if start < 0 or end > eeg.shape[1] or end > emg.shape[1]:
            skipped += 1
            continue

        eeg_epochs.append(eeg[:, start:end])   # (15, 500)
        emg_epochs.append(emg[:, start:end])   # (11, 500)
        labels.append(int(row['label']))

    if skipped:
        print(f"      [{skipped} trials skipped — window out of bounds]")

    return (np.array(eeg_epochs),    # (n_trials, 15, 500)
            np.array(emg_epochs),    # (n_trials, 11, 500)
            np.array(labels))        # (n_trials,)


#  Windowing
# ---------------------------------------

def apply_windowing(epochs, labels, win_size=WIN_SIZE, win_step=WIN_STEP):
    """
    Slide a window across each epoch and return all windows.

    Input:  epochs shape (n_trials, n_channels, epoch_len)
            labels shape (n_trials,)

    Output: windows shape (n_windows, n_channels, win_size)
            win_labels  shape (n_windows,)     label inherited from parent trial
            trial_ids   shape (n_windows,)     which trial each window came from

    Number of windows per trial = (epoch_len - win_size) // win_step + 1
                                = (500 - 80) // 40 + 1 = 11
 """
    windows, win_labels, trial_ids = [], [], []

    for trial_idx, (epoch, label) in enumerate(zip(epochs, labels)):
        # epoch: (n_channels, epoch_len)
        n_samples = epoch.shape[1]
        starts = range(0, n_samples - win_size + 1, win_step)
        for s in starts:
            windows.append(epoch[:, s : s + win_size])
            win_labels.append(label)
            trial_ids.append(trial_idx)

    return (np.array(windows),      # (n_windows, n_channels, win_size)
            np.array(win_labels),   # (n_windows,)
            np.array(trial_ids))    # (n_windows,)



#  Main
# ---------------------------------------

def main():
    parser = argparse.ArgumentParser(description='NeBULA epoch extraction + windowing')
    parser.add_argument('data_root',    nargs='?', default=None)
    parser.add_argument('--subjects',   nargs='+')
    parser.add_argument('--all-subjects', action='store_true')
    parser.add_argument('--output',     default='./preprocessed')
    args = parser.parse_args()

    data_root = find_data_root(args.data_root)
    subjects  = get_subjects(data_root, args.subjects, args.all_subjects)
    os.makedirs(args.output, exist_ok=True)

    print("-" * 65)
    print("  NeBULA — Epoching & Windowing Pipeline")
    print("  Muskaan Garg | H00416442 | Heriot-Watt University")
    print("-" * 65)
    print(f"  Data root  : {data_root}")
    print(f"  Subjects   : {subjects}")
    print(f"  Epoch      : -{PRE_SAMPLES} to +{POST_SAMPLES} samples "
          f"({EPOCH_LEN} total = 2.5s)")
    print(f"  Window     : size={WIN_SIZE} ({WIN_SIZE*5}ms), "
          f"step={WIN_STEP} ({WIN_STEP*5}ms), "
          f"~{(EPOCH_LEN - WIN_SIZE)//WIN_STEP + 1} windows/trial")
    print()

    # Collect across all subjects for combined arrays
    all_eeg_win, all_emg_win = [], []
    all_y_win, all_sid_win, all_tid_win = [], [], []

    loaded = 0

    for sid in subjects:
        print(f"  ── Subject {sid} ──────────────────────────────────────")

        # ── Load ──
        try:
            events = load_events(sid, data_root)
            eeg    = load_eeg(sid, data_root)
            emg    = load_emg(sid, data_root)
        except Exception as e:
            print(f"    ERROR loading: {e}")
            continue

        if eeg is None:
            print("    Skipping — MNE not available")
            continue

        n_eeg = eeg.shape[1]
        n_emg = emg.shape[1]
        print(f"    EEG: {eeg.shape}  EMG: {emg.shape}  "
              f"{'ALIGNED' if n_eeg == n_emg else 'MISMATCH'}")
        print(f"    Events: {len(events)} trials  "
              f"(Task1={( events['label']==1).sum()}, "
              f"Task2={(events['label']==2).sum()}, "
              f"Task3={(events['label']==3).sum()})")

        # Make EMG match EEG length exactly
        min_len = min(n_eeg, n_emg)
        eeg = eeg[:, :min_len]
        emg = emg[:, :min_len]

        # ── Extract epochs ──
        eeg_ep, emg_ep, labels = extract_epochs(eeg, emg, events)
        if len(labels) == 0:
            print("    No valid epochs — skipping subject")
            continue

        # ── Save per-subject trial arrays (for timing analysis) ──
        sub_dir = os.path.join(args.output, f'sub-{sid}')
        os.makedirs(sub_dir, exist_ok=True)
        np.save(os.path.join(sub_dir, f'sub-{sid}_free_eeg.npy'),    eeg_ep)
        np.save(os.path.join(sub_dir, f'sub-{sid}_free_emg.npy'),    emg_ep)
        np.save(os.path.join(sub_dir, f'sub-{sid}_free_labels.npy'), labels)
        print(f"    Epochs: {eeg_ep.shape}  saved to {sub_dir}/")

        # ── Apply windowing ──
        eeg_win, y_win, t_ids = apply_windowing(eeg_ep, labels)
        emg_win, _,     _     = apply_windowing(emg_ep, labels)

        n_win = len(y_win)
        sid_int = int(sid.lstrip('0') or '0')

        all_eeg_win.append(eeg_win)
        all_emg_win.append(emg_win)
        all_y_win.append(y_win)
        all_sid_win.append(np.full(n_win, sid_int, dtype=np.int32))
        all_tid_win.append(t_ids)

        print(f"    Windows: {eeg_win.shape}  "
              f"(Task1={(y_win==1).sum()}, "
              f"Task2={(y_win==2).sum()}, "
              f"Task3={(y_win==3).sum()})")
        loaded += 1

    # ── Save combined windowed arrays ──
    if not all_eeg_win:
        print("\nERROR: No subjects loaded successfully.")
        return

    X_eeg = np.concatenate(all_eeg_win, axis=0)  # (total_windows, 15, 80)
    X_emg = np.concatenate(all_emg_win, axis=0)  # (total_windows, 11, 80)
    y     = np.concatenate(all_y_win,   axis=0)  # (total_windows,)
    sids  = np.concatenate(all_sid_win, axis=0)  # (total_windows,)
    tids  = np.concatenate(all_tid_win, axis=0)  # (total_windows,)

    np.save(os.path.join(args.output, 'X_eeg_win.npy'),       X_eeg)
    np.save(os.path.join(args.output, 'X_emg_win.npy'),       X_emg)
    np.save(os.path.join(args.output, 'y_win.npy'),            y)
    np.save(os.path.join(args.output, 'subject_ids_win.npy'),  sids)
    np.save(os.path.join(args.output, 'trial_ids_win.npy'),    tids)

    print()
    print("=" * 65)
    print(f"  Subjects processed : {loaded}")
    print(f"  EEG windows shape  : {X_eeg.shape}   "
          f"(n_windows, 15 channels, 80 samples)")
    print(f"  EMG windows shape  : {X_emg.shape}   "
          f"(n_windows, 11 channels, 80 samples)")
    print(f"  Labels             : Task1={(y==1).sum()}, "
          f"Task2={(y==2).sum()}, Task3={(y==3).sum()}")
    print(f"  Unique subjects    : {np.unique(sids).tolist()}")
    print(f"  Saved to           : {args.output}/")
    print("=" * 65)


if __name__ == '__main__':
    main()