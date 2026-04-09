"""
NeBULA Dataset - EMG CNN-LSTM — Subject-Independent Classification
-------------------------------------------------------------

 Cross-subject EEG classification using a CNN + BiLSTM hybrid.
 Onset-centred windows are selected based on EEG diagnostics.
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
RESULTS_DIR = "./results/eeg_cnn_lstm"

BATCH_SIZE = 64
EPOCHS     = 160
LR         = 1e-4
DROPOUT    = 0.30
PATIENCE   = 30
SEED       = 42
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

N_CHANNELS = 15
N_CLASSES  = 3

# All possible window start positions from epoch.py
# (epoch length = 500, window size = 80, step = 40)
WINDOW_STARTS = np.array([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400], dtype=np.int64)

# Keep only windows around movement onset (-300ms, -100ms, +100ms)
# Selected based on EEG diagnostic analysis
KEEP_WINDOW_STARTS = {40, 80, 120}



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


def recover_window_starts(n_windows: int) -> np.ndarray:
    """
    Recover window start position for each row in the windowed array.
    Assumes epoch.py appended windows in a fixed repeating order per trial.
    """
    pos = np.arange(n_windows) % len(WINDOW_STARTS)
    return WINDOW_STARTS[pos]


# DATA
# ----------------------------------------------

class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data():
    X = np.load(os.path.join(DATA_DIR, "X_eeg_win.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, "y_win.npy")).astype(np.int64)
    s = np.load(os.path.join(DATA_DIR, "subject_ids_win.npy")).astype(np.int64)
    trial_ids = np.load(os.path.join(DATA_DIR, "trial_ids_win.npy")).astype(np.int64)

    if y.min() == 1:
        y = y - 1

    # only keep the three windows closest to movement onset
    win_starts = recover_window_starts(len(X))
    keep_mask = np.isin(win_starts, list(KEEP_WINDOW_STARTS))

    X = X[keep_mask]
    y = y[keep_mask]
    s = s[keep_mask]
    trial_ids = trial_ids[keep_mask]
    win_starts = win_starts[keep_mask]

    return X, y, s, trial_ids, win_starts


def subject_split(subjects):
    """
    70/15/15 subject-level split.
    np.random.seed is set before this call so the split is deterministic.
    """
    subjects = np.unique(subjects)
    np.random.shuffle(subjects)

    train = subjects[:25]
    val   = subjects[25:30]
    test  = subjects[30:]

    return set(train), set(val), set(test)


def make_loaders(X, y, s):
    train_s, val_s, test_s = subject_split(s)

    def get_loader(subset, shuffle):
        mask = np.array([sid in subset for sid in s])
        ds = EEGDataset(X[mask], y[mask])
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
    """Soft attention over the LSTM timestep dimension."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)


class EEGCnnLstm(nn.Module):
    """
    EEG CNN-LSTM classifier.

    Architecture:
      - Conv2d temporal filter (1 x 7) across time
      - Conv2d spatial filter (15 x 1) across all channels
      - Second Conv2d block for higher-level features
      - BiLSTM over 20 timesteps with attention readout
      - Linear classifier

    Input:  (B, 15, 80)
    Output: (B, 3)
    """
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, 32, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),

            # Spatial convolution across all 15 channels
            nn.Conv2d(32, 32, kernel_size=(N_CHANNELS, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),   # 80 -> 40
            nn.Dropout(DROPOUT),

            # Second temporal block
            nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),   # 40 -> 20
            nn.Dropout(DROPOUT),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )

        self.attn = AttentionPool(128)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x):
        # input is (B, 15, 80) — add channel dim for Conv2d which expects (B, C, H, W)
        x = x.unsqueeze(1)         # (B, 1, 15, 80)
        x = self.cnn(x)            # (B, 64, 1, 20)
        x = x.squeeze(2)           # (B, 64, 20)
        x = x.transpose(1, 2)      # (B, 20, 64)
        x, _ = self.lstm(x)        # (B, 20, 128)
        x = self.attn(x)           # (B, 128)
        return self.classifier(x)


# TRAIN
# ----------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    preds, trues = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad()

        out = model(xb)
        loss = criterion(out, yb)

        if is_train:
            # clip gradients to prevent exploding gradients in the LSTM
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


def train():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()

    X, y, s, trial_ids, win_starts = load_data()
    train_loader, val_loader, test_loader, train_s, val_s, test_s = make_loaders(X, y, s)

    model = EEGCnnLstm().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("-" * 65)
    print("  EEG CNN-LSTM — Subject-Independent NeBULA Classification")
    print("-" * 65)
    print(f"  Device         : {device}")
    print(f"  Input shape    : {X.shape}")
    print(f"  Window starts  : {sorted(list(KEEP_WINDOW_STARTS))}")
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

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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

        # save checkpoint if val F1 improved, or if F1 tied but loss is lower
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

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "eeg_cnn_lstm.pt"))
    np.save(os.path.join(RESULTS_DIR, "eeg_cnn_lstm_history.npy"),
            np.array(history, dtype=object))

    summary = {
        "model": "EEG_CNN_LSTM",
        "experiment": "subject_independent",
        "device": str(device),
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
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
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
        },
    }

    with open(os.path.join(RESULTS_DIR, "eeg_cnn_lstm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Model   → {RESULTS_DIR}/eeg_cnn_lstm.pt")
    print(f"  History → {RESULTS_DIR}/eeg_cnn_lstm_history.npy")
    print(f"  Summary → {RESULTS_DIR}/eeg_cnn_lstm_summary.json")


if __name__ == "__main__":
    train()