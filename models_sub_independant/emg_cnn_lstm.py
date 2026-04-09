"""
NeBULA Dataset - EMG CNN-LSTM — Subject-Independent Classification
-------------------------------------------------------------

Hybrid model combining a CNN + BiLSTM branch (raw signal)
with a handcrafted feature branch (RMS, mean, std, waveform length).

  - Execution-phase windows only (100ms to 1100ms post-onset)
  - 3 consecutive windows concatenated per trial (240 samples)
  - Two parallel branches merged before the classifier
  - Best checkpoint selected by validation macro F1
"""

import os
import json
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# CONFIG
# ----------------------------------------------
DATA_DIR    = "../preprocessed"
RESULTS_DIR = "./results/emg_cnn_lstm"

BATCH_SIZE = 64
EPOCHS     = 160
LR         = 5e-4
DROPOUT    = 0.30
PATIENCE   = 30
SEED       = 42

# Windowing from epoch.py:
# epoch length = 500, win size = 80, step = 40
# starts = [0,40,80,120,160,200,240,280,320,360,400]
# movement onset is at sample 100 in the epoch
# Keep windows more centered on execution period:
# valid starts chosen from ~0 to +1500 ms post-onset
KEEP_WINDOW_STARTS = {120, 160, 200, 240, 280, 320}

N_CHANNELS = 11
N_CLASSES  = 3


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# FEATURE ENGINEERING
# ----------------------------------------------

def compute_emg_features(X: np.ndarray) -> np.ndarray:
    """
    X shape: (N, C, T)
    Returns handcrafted features shape: (N, 4*C)
    Features per channel:
      - RMS
      - mean
      - std
      - waveform length
    """

    # RMS captures overall activation intensity — diagnostic analysis showed
    # Task1 > Task2 > Task3 ordering is consistent across all 11 channels
    rms = np.sqrt(np.mean(X ** 2, axis=2))
    # mean reflects average activation level across the window
    mean = np.mean(X, axis=2)
    # std captures how much the signal varies within the window
    std = np.std(X, axis=2)
    # waveform length measures total signal variation — sensitive to both frequency and amplitude
    wl = np.sum(np.abs(np.diff(X, axis=2)), axis=2)

    feats = np.concatenate([rms, mean, std, wl], axis=1).astype(np.float32)
    return feats


def get_window_start_indices(n_windows: int, n_trials: int) -> np.ndarray:
    """
    Recover window start for each flattened window using trial_ids_win ordering.

    Since epoch.py appends windows in the same fixed order for every trial:
    starts = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400]

    If each trial contributes 11 windows, then for flattened window index i:
      window_position_in_trial = i % 11
    """
    starts = np.array([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400], dtype=np.int64)
    assert len(starts) == 11
    pos = np.arange(n_windows) % len(starts)
    return starts[pos]


# DATA
# ----------------------------------------------

class EMGHybridDataset(Dataset):
    def __init__(self, X_raw: np.ndarray, X_feat: np.ndarray, y: np.ndarray):
        self.X_raw = torch.tensor(X_raw, dtype=torch.float32)
        self.X_feat = torch.tensor(X_feat, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_raw[idx], self.X_feat[idx], self.y[idx]


def load_data():
    X = np.load(os.path.join(DATA_DIR, "X_emg_win.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, "y_win.npy")).astype(np.int64)
    s = np.load(os.path.join(DATA_DIR, "subject_ids_win.npy")).astype(np.int64)
    trial_ids = np.load(os.path.join(DATA_DIR, "trial_ids_win.npy")).astype(np.int64)

    if y.min() == 1:
        y = y - 1

    # Recover fixed window starts and keep only execution-phase windows
    win_starts = get_window_start_indices(n_windows=len(X), n_trials=len(np.unique(trial_ids)))
    keep_mask = np.isin(win_starts, list(KEEP_WINDOW_STARTS))

    X = X[keep_mask]
    y = y[keep_mask]
    s = s[keep_mask]
    trial_ids = trial_ids[keep_mask]
    win_starts = win_starts[keep_mask]

    X_feat = compute_emg_features(X)

    def create_window_sequences(X, X_feat, y, s, trial_ids, win_starts):
        X_seq, Xf_seq, y_seq, s_seq = [], [], [], []

        # group by (subject, trial), not just trial
        unique_keys = sorted(set(zip(s.tolist(), trial_ids.tolist())))

        for sid, tid in unique_keys:
            mask = (s == sid) & (trial_ids == tid)

            X_t = X[mask]
            Xf_t = X_feat[mask]
            y_t = y[mask]
            ws_t = win_starts[mask]

            # make sure windows are in temporal order
            order = np.argsort(ws_t)
            X_t = X_t[order]
            Xf_t = Xf_t[order]
            y_t = y_t[order]

            # build 3-window sequences
            for i in range(len(X_t) - 2):
                X_seq.append(np.concatenate(
                    [X_t[i], X_t[i + 1], X_t[i + 2]],
                    axis=1
                ))  # (11, 240)

                Xf_seq.append(np.concatenate(
                    [Xf_t[i], Xf_t[i + 1], Xf_t[i + 2]],
                    axis=0
                ))

                y_seq.append(y_t[i + 2])   # label = latest window
                # using the last window's label since it reflects the most recent state
                # all windows in the same trial have the same label anyway
                s_seq.append(sid)

        return (
            np.array(X_seq, dtype=np.float32),
            np.array(Xf_seq, dtype=np.float32),
            np.array(y_seq, dtype=np.int64),
            np.array(s_seq, dtype=np.int64),
        )

    X_raw, X_feat, y, s = create_window_sequences(X, X_feat, y, s, trial_ids, win_starts)

    return X_raw, X_feat, y, s

def subject_split(subjects):
    subjects = np.unique(subjects)
    np.random.shuffle(subjects)

    train = subjects[:25]
    val   = subjects[25:30]
    test  = subjects[30:]

    return set(train), set(val), set(test)


def make_loaders(X_raw, X_feat, y, s):
    train_s, val_s, test_s = subject_split(s)

    def get_loader(subset, shuffle):
        mask = np.array([sid in subset for sid in s])
        ds = EMGHybridDataset(X_raw[mask], X_feat[mask], y[mask])
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return (
        get_loader(train_s, True),
        get_loader(val_s, False),
        get_loader(test_s, False),
        train_s, val_s, test_s,
    )


# MODEL
# ----------------------------------------------

class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)


class RawSignalBranch(nn.Module):
    # processes the raw EMG sequence through CNN + BiLSTM + attention
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(N_CHANNELS, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),

            nn.Dropout(DROPOUT),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )

        self.attn = AttentionPool(128)

    def forward(self, x):
        x = self.cnn(x)         # (B, 64, 20)
        x = x.transpose(1, 2)   # (B, 20, 64)
        x, _ = self.lstm(x)     # (B, 20, 128)
        x = self.attn(x)        # (B, 128)
        return x


class FeatureBranch(nn.Module):
    # processes the handcrafted feature vector through a small MLP
    # provides a direct shortcut to amplitude-based class differences
    # that the CNN would otherwise need to learn from scratch
    def __init__(self, in_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(DROPOUT),

            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.mlp(x)


class EMGHybridModel(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.raw_branch = RawSignalBranch()
        self.feat_branch = FeatureBranch(feat_dim)

        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x_raw, x_feat):
        # concatenate both embeddings — raw branch learns temporal dynamics,
        # feature branch provides explicit amplitude statistics
        z_raw = self.raw_branch(x_raw)     # (B, 128)
        z_feat = self.feat_branch(x_feat)  # (B, 32)
        z = torch.cat([z_raw, z_feat], dim=1)
        return self.classifier(z)


# TRAIN
# ----------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    preds, trues = [], []

    for xb_raw, xb_feat, yb in loader:
        xb_raw = xb_raw.to(device)
        xb_feat = xb_feat.to(device)
        yb = yb.to(device)

        if is_train:
            # mild augmentation applied only during training to improve cross-subject generalisation
            # noise scaled relative to batch std so it adapts to the signal amplitude
            noise_std = 0.01 * xb_raw.std().detach()
            xb_raw = xb_raw + noise_std * torch.randn_like(xb_raw)

            # per-channel amplitude jitter — small random scaling between 0.98 and 1.02
            # prevents the model from relying on exact amplitude values

            scale = torch.empty(
                xb_raw.size(0), xb_raw.size(1), 1, device=xb_raw.device
            ).uniform_(0.98, 1.02)
            xb_raw = xb_raw * scale

        if is_train:
            optimizer.zero_grad()

        out = model(xb_raw, xb_feat)
        loss = criterion(out, yb)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(yb)
        preds.extend(out.argmax(1).detach().cpu().numpy())
        trues.extend(yb.detach().cpu().numpy())

    n = len(loader.dataset)
    mean_loss = total_loss / n
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro", zero_division=0)
    return mean_loss, acc, f1, trues, preds

def get_class_weights(y_train: np.ndarray, device):
    # inverse-frequency weighting so minority classes contribute equally to the loss
    counts = np.bincount(y_train, minlength=N_CLASSES).astype(np.float32)
    weights = counts.sum() / (N_CLASSES * counts)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights

def train():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()

    X_raw, X_feat, y, s = load_data()
    train_loader, val_loader, test_loader, train_s, val_s, test_s = make_loaders(
        X_raw, X_feat, y, s
    )

    feat_dim = X_feat.shape[1]
    model = EMGHybridModel(feat_dim=feat_dim).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("-" * 65)
    print("  EMG HYBRID CNN-LSTM — Subject-Independent NeBULA")
    print("  Raw signal branch + handcrafted feature branch")
    print("-" * 65)
    print(f"  Device         : {device}")
    print(f"  Raw input      : {X_raw.shape}")
    print(f"  Feature input  : {X_feat.shape}")
    print(f"  Kept starts    : {sorted(list(KEEP_WINDOW_STARTS))}")
    print(f"  Train subjects : {sorted([int(x) for x in train_s])}")
    print(f"  Val subjects   : {sorted([int(x) for x in val_s])}")
    print(f"  Test subjects  : {sorted([int(x) for x in test_s])}")
    print(f"  Parameters     : {n_params:,}")
    print()

    def split_stats(loader, name):
        ys = loader.dataset.y.numpy()
        print(f"  {name:5s}: {len(ys):5d} windows  "
              f"(Task0={(ys==0).sum()}, "
              f"Task1={(ys==1).sum()}, "
              f"Task2={(ys==2).sum()})")

    split_stats(train_loader, "train")
    split_stats(val_loader, "val")
    split_stats(test_loader, "test")
    print()

    y_train = train_loader.dataset.y.numpy()
    class_weights = get_class_weights(y_train, device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
    )

    best_state = None
    best_epoch = 0
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    patience_ctr = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        tl, ta, tf, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        vl, va, vf, _, _ = run_epoch(model, val_loader, criterion, None, device)

        scheduler.step(vf)

        history.append({
            "epoch": epoch,
            "train_loss": tl,
            "train_acc": ta,
            "train_f1": tf,
            "val_loss": vl,
            "val_acc": va,
            "val_f1": vf,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"  Epoch {epoch:03d} | "
            f"train loss={tl:.4f} acc={ta:.3f} f1={tf:.3f} | "
            f"val loss={vl:.4f} acc={va:.3f} f1={vf:.3f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        improved = (vf > best_val_f1) or (vf == best_val_f1 and vl < best_val_loss)

        if improved:
            best_val_f1 = vf
            best_val_loss = vl
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    model.load_state_dict(best_state)

    _, test_acc, test_f1, y_true, y_pred = run_epoch(
        model, test_loader, criterion, None, device
    )

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    print("\n" + "-" * 65)
    print("  TEST RESULTS")
    print(f"  Accuracy   : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"  F1 macro   : {test_f1:.4f}")
    print(f"  Per class  : "
          f"Task1={report['0']['f1-score']:.3f}  "
          f"Task2={report['1']['f1-score']:.3f}  "
          f"Task3={report['2']['f1-score']:.3f}")
    print(f"  Precision  : "
          f"Task1={report['0']['precision']:.3f}  "
          f"Task2={report['1']['precision']:.3f}  "
          f"Task3={report['2']['precision']:.3f}")
    print(f"  Recall     : "
          f"Task1={report['0']['recall']:.3f}  "
          f"Task2={report['1']['recall']:.3f}  "
          f"Task3={report['2']['recall']:.3f}")
    print(f"  Best epoch : {best_epoch}")
    print("  Confusion matrix:")
    print(f"    {cm[0].tolist()}  ← Task 1 predicted as")
    print(f"    {cm[1].tolist()}  ← Task 2 predicted as")
    print(f"    {cm[2].tolist()}  ← Task 3 predicted as")
    print("-" * 65)

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "emg_hybrid_cnn_lstm.pt"))
    np.save(os.path.join(RESULTS_DIR, "emg_hybrid_cnn_lstm_history.npy"),
            np.array(history, dtype=object))

    summary = {
        "model": "EMG_HYBRID_CNN_LSTM",
        "experiment": "subject_independent",
        "device": str(device),
        "n_params": n_params,
        "best_epoch": best_epoch,
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "train_subjects": sorted([int(x) for x in train_s]),
        "val_subjects": sorted([int(x) for x in val_s]),
        "test_subjects": sorted([int(x) for x in test_s]),
        "kept_window_starts": sorted([int(x) for x in KEEP_WINDOW_STARTS]),
        "config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "dropout": DROPOUT,
            "patience": PATIENCE,
        },
    }

    with open(os.path.join(RESULTS_DIR, "emg_hybrid_cnn_lstm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Model   → {RESULTS_DIR}/emg_hybrid_cnn_lstm.pt")
    print(f"  History → {RESULTS_DIR}/emg_hybrid_cnn_lstm_history.npy")
    print(f"  Summary → {RESULTS_DIR}/emg_hybrid_cnn_lstm_summary.json")


if __name__ == "__main__":
    train()

